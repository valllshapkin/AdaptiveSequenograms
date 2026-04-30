import torch
import torch.nn as nn
import time

# --- 1. АВТОЭНКОДЕР (Архитектура та же самая) ---
class Audio2DAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Энкодер (сжимает частоту 256->1, время жмет в 8 раз)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)), nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 2), dilation=(1, 2)), nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(4, 3), stride=(2, 1), padding=(1, 4), dilation=(1, 4)), nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=(4, 3), stride=(2, 1), padding=(1, 8), dilation=(1, 8)), nn.GELU(),
            nn.Conv2d(256, 512, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)), nn.GELU(),
        )
        # Декодер (восстанавливает обратно)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 3), stride=(2, 1), padding=(0, 1)), nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)), nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)), nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), nn.GELU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- 2. УМНЫЙ РЕЗЧИК (Инференс) ---
@torch.no_grad() # Отключаем градиенты НАВСЕГДА для этой функции
def process_long_spectrogram(model, long_tensor, max_vram_mb=500):
    """
    max_vram_mb - сколько мегабайт видеопамяти пользователь разрешил использовать на обработку.
    """
    model.eval() # Переводим модель в режим предсказания
    
    # 1. СЧИТАЕМ ОПТИМАЛЬНЫЙ РАЗМЕР ЧАНКА
    # Эмпирическое правило для этой модели: 512 пикселей в режиме no_grad жрут ~35-40 МБ.
    # Значит 1 МБ памяти вмещает примерно 12 пикселей времени.
    safe_margin = 0.8 # Оставляем 20% памяти про запас
    pixels_per_mb = 12 
    
    target_chunk = int(max_vram_mb * safe_margin * pixels_per_mb)
    
    # Делаем размер кратным 512 (чтобы математика сверток всегда была идеальной)
    chunk_size = max(512, (target_chunk // 512) * 512)
    overlap = 256 # Перекрытие всегда оставляем фиксированным
    
    print(f"[*] Лимит VRAM: {max_vram_mb} MB -> Вычислен размер чанка: {chunk_size} пикселей")

    # 2. НАРЕЗКА И СКЛЕЙКА
    B, C, F, Total_T = long_tensor.shape
    output_tensor = torch.zeros_like(long_tensor)
    step = chunk_size - overlap
    
    # Паддинг краев
    padded_input = torch.nn.functional.pad(long_tensor, (overlap//2, overlap//2))
    
    chunks_processed = 0
    for start in range(0, Total_T, step):
        end = start + chunk_size
        if end > padded_input.shape[-1]:
            break
            
        chunk = padded_input[:, :, :, start:end]
        out_chunk = model(chunk)
        
        valid_out = out_chunk[:, :, :, overlap//2 : -overlap//2]
        output_tensor[:, :, :, start : start + step] = valid_out
        chunks_processed += 1
        
    return output_tensor, chunks_processed


# --- 3. ТЕСТ: ОБУЧЕНИЕ И ИНФЕРЕНС ---
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {DEVICE}")

    model = Audio2DAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Симуляция гигантского файла
    Total_Time = 37640 
    dummy_input = torch.randn(1, 1, 256, Total_Time, device=DEVICE)
    
    # ==========================================
    # ЭТАП 1: Симуляция ОБУЧЕНИЯ (с градиентами)
    # ==========================================
    print("\n--- ЭТАП 1: ОБУЧЕНИЕ (Train Step) ---")
    model.train() # Включаем обучение
    optimizer.zero_grad()
    
    # Вырезаем случайный МАЛЕНЬКИЙ кусок для обучения (чтобы не взорвать память)
    train_chunk_size = 512
    start_idx = torch.randint(0, Total_Time - train_chunk_size, (1,)).item()
    train_chunk = dummy_input[:, :, :, start_idx : start_idx + train_chunk_size]
    
    # Прогоняем и считаем градиенты
    out_train = model(train_chunk)
    loss = torch.nn.functional.mse_loss(out_train, train_chunk)
    loss.backward() # Теперь это работает, потому что нет no_grad!
    optimizer.step()
    print(f"Шаг обучения выполнен успешно. Loss: {loss.item():.4f}")
    
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats() # Сбрасываем счетчик памяти
        
    # ==========================================
    # ЭТАП 2: Симуляция ИНФЕРЕНСА (обработка всего файла)
    # ==========================================
    print("\n--- ЭТАП 2: ИНФЕРЕНС (Обработка всего трека) ---")
    
    # Сценарий А: Пользователь с RTX 3060 (выделил 2000 МБ)
    start_t = time.time()
    out_big, chunks_big = process_long_spectrogram(model, dummy_input, max_vram_mb=2000)
    print(f"Время: {time.time() - start_t:.2f} сек. Обработано чанков: {chunks_big}")
    
    # Сценарий Б: Пользователь со старым ноутбуком (выделил 300 МБ)
    start_t = time.time()
    out_small, chunks_small = process_long_spectrogram(model, dummy_input, max_vram_mb=300)
    print(f"Время: {time.time() - start_t:.2f} сек. Обработано чанков: {chunks_small}")

    if DEVICE == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\nПиковое потребление VRAM за инференс: {peak_mem_mb:.2f} МБ")