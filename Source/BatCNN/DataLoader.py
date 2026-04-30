import random
import time
import multiprocessing as mp
import torch
import numpy as np
from pathlib import Path
from typing import List

from scipy.signal.windows import hann, blackmanharris

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import Window, WindowNorm
from BatSpec.Core.Record import loadRecord, correctDC, resampleRecord
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import makeLogDB, DSPContext, makeRobustSpec
from BatSpec.Core.Spectral.Statistic import noiseZNormByFreq

import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType
from BatCNN.Config import HardwareConfig

# ==========================================
# 1. ЖЕСТКАЯ ГЕНЕРАЦИЯ ОКОН (Safe for Multiprocessing)
# ==========================================
# Функции-обертки нужны вместо lambda, чтобы multiprocessing не падал с ошибкой pickling
def _get_hann(n: int):
    return hann(n, sym=False)

def _get_blha(n: int):
    return blackmanharris(n, sym=False)

# ИДЕАЛЬНОЕ ВРЕМЯ:
# При bins=256, makeRobustSpec делает fft_length = (256-1)*2 = 510.
# Значит, окно должно быть ровно 510 сэмплов.
# time = 510 / 256000 = 0.0019921875 сек.
EXACT_WINDOW_TIME = 510 / 256000

FIXED_HANN = Window(func=_get_hann, time=EXACT_WINDOW_TIME, norm=WindowNorm.ENERGY)
FIXED_BLHA = Window(func=_get_blha, time=EXACT_WINDOW_TIME, norm=WindowNorm.ENERGY)


# ==========================================
# 2. CPU ПРОЦЕСС (MATH WORKER)
# ==========================================
def cpu_worker(wav_paths: List[Path], queue: mp.Queue, target_sr: int, target_bins: int):
    """Фоновый процесс: читает аудио, делает STFT и нормализацию, кладет Numpy-матрицы в очередь."""
    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    
    while True:
        try:
            path = random.choice(wav_paths)
            
            # Тайм-домен (Ресемплинг + Калибровка)
            record = loadRecord(path)
            record = resampleRecord(record, target_sr) # SR жестко 256000
            record = correctDC(record)
            record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
            
            # Частотный домен (STFT)
            with DSPContext(ctx_numpy_cpu):
                spec_robust = makeRobustSpec(
                    record,
                    window=(FIXED_HANN, FIXED_BLHA),
                    overlap=0.8,
                    bins=target_bins, 
                    shifts=(-1, 1)  
                ).to_context(ctx_numpy_cpu)
            
            # Нормализация
            zspec = noiseZNormByFreq(spec_robust)
            log_spec = makeLogDB(zspec, add_one=True)
            
            # Извлекаем сырой numpy array: [C, F, T]
            tensor_np = ma.to_numpy(log_spec.values[0])
            if tensor_np.ndim == 2:
                tensor_np = np.expand_dims(tensor_np, axis=0) 
                
            tensor_np = tensor_np.astype(np.float32)
            
            # Кладем в очередь (заблокируется, если RAM очередь полна)
            queue.put(tensor_np)
            
        except Exception as e:
            print(f"[CPU Worker Error] Ошибка при обработке {path}: {e}")
            time.sleep(1)


# ==========================================
# 3. GPU МЕНЕДЖЕР
# ==========================================
class DualProcessDataLoader:
    def __init__(self, wav_paths: List[Path], config: HardwareConfig):
        self.cfg = config
        self.gpu_pool: List[torch.Tensor] = []
        
        # 'spawn' критичен для PyTorch (предотвращает зависания CUDA)
        ctx = mp.get_context('spawn') 
        self.queue = ctx.Queue(maxsize=self.cfg.CPU_QUEUE_SIZE)
        
        self.worker_process = ctx.Process(
            target=cpu_worker, 
            args=(wav_paths, self.queue, self.cfg.TARGET_SR, self.cfg.FREQ_BINS),
            daemon=True
        )
        self.worker_process.start()
        print("[DataLoader] Фоновый CPU Worker запущен.")

    def _sync_queue_to_gpu(self):
        """Переносит всё готовое из RAM-очереди в VRAM-пул без блокировки."""
        while not self.queue.empty():
            try:
                tensor_np = self.queue.get_nowait()
                # Перенос на GPU
                tensor_gpu = torch.tensor(tensor_np, device=self.cfg.device, dtype=torch.float32)
                self.gpu_pool.append(tensor_gpu)
                
                # Очистка старых данных (FIFO)
                if len(self.gpu_pool) > self.cfg.GPU_POOL_SIZE:
                    old_tensor = self.gpu_pool.pop(0)
                    del old_tensor 
            except mp.queues.Empty:
                break

    def get_batch(self) -> torch.Tensor:
        """Нарезает кропы из пула GPU с умным акцентом на сигналы."""
        self._sync_queue_to_gpu()
        
        if len(self.gpu_pool) == 0:
            print("[DataLoader] Ожидание первых данных от процессора...")
            tensor_np = self.queue.get(block=True)
            tensor_gpu = torch.tensor(tensor_np, device=self.cfg.device, dtype=torch.float32)
            self.gpu_pool.append(tensor_gpu)
            
        batch = torch.empty((self.cfg.BATCH_SIZE, 1, self.cfg.FREQ_BINS, self.cfg.CHUNK_SIZE), device=self.cfg.device)
        
        for i in range(self.cfg.BATCH_SIZE):
            t = random.choice(self.gpu_pool)
            T = t.shape[-1]
            
            if T <= self.cfg.CHUNK_SIZE:
                pad_len = self.cfg.CHUNK_SIZE - T
                batch[i] = torch.nn.functional.pad(t, (0, pad_len))
            else:
                # 75% шанс: Фокусируемся на полезном сигнале
                # 25% шанс: Случайный кроп (чтобы сеть видела чистый шум)
                if random.random() < 0.95:
                    # Ищем "яркие" участки. Берём максимум энергии по всем частотам (схлопываем в 1D)
                    # Так как данные Z-нормализованы, сигналы будут иметь большие положительные значения.
                    energy_profile = t[0].max(dim=0).values  # Shape: (T,)
                    
                    # Зануляем отрицательный шум и возводим в квадрат, чтобы усилить пики
                    weights = torch.clamp(energy_profile, min=0.0) ** 2
                    
                    if weights.sum() > 1e-5:
                        # Сэмплируем 1 индекс на основе распределения энергии
                        center_idx = torch.multinomial(weights, 1).item()
                        
                        # Преобразуем центр в начало окна
                        start = center_idx - (self.cfg.CHUNK_SIZE // 2)
                        # Защита от выхода за границы
                        start = max(0, min(start, T - self.cfg.CHUNK_SIZE))
                    else:
                        # Если сигнал сплошной шум (нет ярких участков) - fallback на рандом
                        start = random.randint(0, T - self.cfg.CHUNK_SIZE)
                else:
                    # Случайный кроп (Uniform distribution)
                    start = random.randint(0, T - self.cfg.CHUNK_SIZE)
                    
                batch[i] = t[:, :, start : start + self.cfg.CHUNK_SIZE]
                
        return batch

    def __del__(self):
        if hasattr(self, 'worker_process') and self.worker_process.is_alive():
            self.worker_process.terminate()