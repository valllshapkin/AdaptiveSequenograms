import warnings
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch

# --- Импорты BatSpec и MultiArray ---
import MultiArray as ma
from MultiArray.Core import ArrayContext, Framework, DeviceType
from BatSpec.Core.Functions import SpecFunc, TimeFunc
from BatSpec.Core.Physical.Units import UREG
# --- Ключевой импорт для решения проблемы ---
from BatSpec.Core.Spectral import interpolateToShape

# --- Импорты архитектуры сети ---
from BatCNN.Model import SlidingPerceptronAutoencoder
from BatCNN.Config import HardwareConfig

class BatCNNInference:
    """
    Инференс-модуль для модели SlidingPerceptronAutoencoder.
    Автоматически адаптируется под контекст входных данных (NumPy/Torch) и железо.
    """
    def __init__(self, weights_path: Path | str = None):
        self.config = HardwareConfig()
        self.device = torch.device(self.config.device)
        self.model = SlidingPerceptronAutoencoder().to(self.device)
        
        if weights_path is None:
            weights_path = Path(__file__).parent / "batcnn_weights.pt"
        else:
            weights_path = Path(weights_path)
            
        if not weights_path.exists():
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
            
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[*] Модель BatCNN успешно загружена на {self.device}")

        # Целевой контекст для нейросети
        dev_enum = DeviceType.GPU if self.device.type == 'cuda' else DeviceType.CPU
        self.torch_ctx = ArrayContext(Framework.TORCH, dev_enum, None)

    @torch.no_grad()
    def process(self, spec: SpecFunc) -> Tuple[SpecFunc, List[TimeFunc]]:
        """
        Принимает спектрограмму, прогоняет через Autoencoder и возвращает:
        1. Восстановленный SpecFunc (такой же размерности и контекста как оригинал).
        2. Список из 32 TimeFunc, представляющих активации латентного слоя во времени.
        """
        orig_ctx = spec.context
        orig_freq_bins = spec.freq[0].shape[-1]
        orig_time_frames = spec.time[0].shape[-1]

        # 1. СТАНДАРТИЗАЦИЯ ВХОДА: Интерполируем спектрограмму к размеру сети
        if orig_freq_bins != self.config.FREQ_BINS:
            warnings.warn(f"Сеть ожидает {self.config.FREQ_BINS} частотных бинов, получено {orig_freq_bins}. "
                          f"Входные данные будут интерполированы.")
            spec_standard = interpolateToShape(spec, target_shape=(self.config.FREQ_BINS, orig_time_frames))
        else:
            spec_standard = spec

        # 2. Переносим стандартизированные данные в PyTorch
        spec_torch = spec_standard.to_context(self.torch_ctx)
        mat_t, unit = spec_torch.values
        
        if mat_t.ndim > 2: mat_t = mat_t[0]
            
        # 3. Формируем тензор для сети: (Batch=1, Channels=1, Freq, Time)
        x = mat_t.unsqueeze(0).unsqueeze(0).to(torch.float32)

        # 4. Проход через сеть
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
            reconstructed, latent = self.model(x, return_latent=True)
            
        # 5. Постобработка: создаем SpecFunc с укороченной осью времени
        recon_mat_torch = reconstructed[0, 0, :, :]
        output_time_frames = recon_mat_torch.shape[-1]

        time_axis_sliced = spec_torch.time[0][:output_time_frames]

        recon_spec_standard = SpecFunc(
            matrix=(recon_mat_torch, unit),
            freq=spec_torch.freq[0],
            time=time_axis_sliced
        )

        # 6. ОБРАТНАЯ ИНТЕРПОЛЯЦИЯ: Возвращаем к оригинальной частотной размерности
        if orig_freq_bins != self.config.FREQ_BINS:
            interp_spec = interpolateToShape(
                recon_spec_standard,
                target_shape=(orig_freq_bins, output_time_frames)
            )
        else:
            interp_spec = recon_spec_standard

        # 7. ФИНАЛЬНОЕ ВЫРАВНИВАНИЕ: Дополняем (pad) матрицу, чтобы она точно соответствовала оригиналу
        final_mat_unpadded, final_unit = interp_spec.values
        mat_in_orig_ctx = ma.convert_to(final_mat_unpadded, orig_ctx)

        if mat_in_orig_ctx.shape[-1] < orig_time_frames:
            pad_len = orig_time_frames - mat_in_orig_ctx.shape[-1]
            pad_widths = [(0, 0)] * (mat_in_orig_ctx.ndim - 1) + [(0, pad_len)]
            
            if orig_ctx.isTorch():
                final_mat = torch.nn.functional.pad(mat_in_orig_ctx, (0, pad_len), "constant", 0)
            else:
                final_mat = orig_ctx.fw.pad(mat_in_orig_ctx, pad_widths, mode='constant')
        else:
            final_mat = mat_in_orig_ctx

        final_recon_spec = SpecFunc(
            matrix=(final_mat, final_unit),
            freq=spec.freq[0],
            time=spec.time[0]
        )

        # 8. Постобработка латентного пространства
        latent_mat = latent[0, :, 0, :]
        channels, t_latent = latent_mat.shape
        dt_orig, _ = spec_torch.dt
        dt_latent = float(dt_orig) * 10.0
        
        latent_time_np = np.arange(t_latent) * dt_latent
        axis_ctx = ArrayContext(orig_ctx._framework, orig_ctx._device, None)
        latent_time_axis = ma.convert_to(latent_time_np, axis_ctx)

        latent_funcs = []
        for i in range(channels):
            chan_activations = latent_mat[i]
            tf = TimeFunc(
                values=(chan_activations, UREG.dimensionless),
                axis=latent_time_axis
            ).to_context(orig_ctx)
            latent_funcs.append(tf)

        return final_recon_spec, latent_funcs