import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. АВТО-НАСТРОЙКА ПОД ЖЕЛЕЗО (Colab / Local)
# ==========================================
def auto_configure_env():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if DEVICE == "cuda":
        # Узнаем объем VRAM в Гигабайтах
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        name = torch.cuda.get_device_name(0)
        print(f"[*] Обнаружена GPU: {name} (VRAM: {vram_gb:.1f} GB)")
        
        # Подбираем batch_size в зависимости от памяти
        # (Наши чанки 512x256 весят мало, так что можно брать большие батчи)
        if vram_gb >= 14:     # Colab Tesla T4 / P100 (15-16 GB)
            batch_size = 64
        elif vram_gb >= 8:    # Средние игровые карты (8-12 GB)
            batch_size = 32
        else:                 # Бюджетные карты (<8 GB)
            batch_size = 16
            
        chunk_size = 512 # Оставляем константой ради архитектуры сверток
    else:
        print("[*] Обнаружен CPU. Обучение будет медленным.")
        batch_size = 4
        chunk_size = 512
        
    print(f"[*] Установлены параметры: Batch Size = {batch_size}, Chunk Size = {chunk_size}\n")
    return DEVICE, batch_size, chunk_size

# ==========================================
# 2. УМНАЯ ФУНКЦИЯ ПОТЕРЬ ДЛЯ СПЕКТРОГРАММ
# ==========================================
class SparseSpectrogramLoss(nn.Module):
    def __init__(self, peak_weight=10.0):
        super().__init__()
        self.peak_weight = peak_weight

    def forward(self, pred, target):
        # 1. Считаем базовый L1 Loss (он дает более резкие границы, чем MSE)
        l1_diff = torch.abs(pred - target)
        
        # 2. Создаем маску: где в оригинале звук громче, там вес ошибки больше
        # Предполагаем, что фоновый шум болтается около 0 (до 0.5), а сигналы > 1.0
        # Всё, что громче 0.5, будет умножаться на peak_weight
        weights = torch.ones_like(target)
        weights[target > 0.5] = self.peak_weight
        
        # 3. Применяем веса к ошибке
        weighted_loss = l1_diff * weights
        
        return weighted_loss.mean()

# ==========================================
# 3. СИНТЕТИКА И ДАТАСЕТ
# ==========================================
def generate_bat_spectrogram(F=256, T=5000):
    spec = torch.zeros(1, F, T)
    num_chirps = int(T / 150) 
    
    for _ in range(num_chirps):
        start_t = random.randint(0, T - 100)
        duration = random.randint(15, 30)
        end_t = start_t + duration
        start_f = random.randint(150, 240)
        end_f = random.randint(40, 100)
        
        # Чирп
        for t in range(start_t, end_t):
            progress = (t - start_t) / duration
            f = int(start_f - progress * (start_f - end_f))
            f_min, f_max = max(0, f - 1), min(F, f + 2)
            spec[0, f_min:f_max, t] += 2.0 # Яркий сигнал
            
        # Эхо
        echo_delay = random.randint(15, 40)
        if end_t + echo_delay < T:
            for t in range(start_t, end_t):
                progress = (t - start_t) / duration
                f = int(start_f - progress * (start_f - end_f))
                f_min, f_max = max(0, f - 1), min(F, f + 2)
                spec[0, f_min:f_max, t + echo_delay] += 0.8 
                
    noise = torch.randn_like(spec) * 0.15 # Тихий шум
    return spec + noise

class BatDataset(Dataset):
    def __init__(self, data_list, chunk_size, samples_per_epoch=1000):
        self.data = data_list
        self.chunk_size = chunk_size
        self.samples = samples_per_epoch

    def __len__(self): return self.samples

    def __getitem__(self, idx):
        tensor = random.choice(self.data)
        start = random.randint(0, tensor.shape[-1] - self.chunk_size)
        return tensor[:, :, start : start + self.chunk_size]

# ==========================================
# 4. АВТОЭНКОДЕР
# ==========================================
class Audio2DAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
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
    def forward(self, x): return self.decoder(self.encoder(x))

# ==========================================
# 5. ГЛАВНЫЙ ЦИКЛ (Colab Ready)
# ==========================================
if __name__ == "__main__":
    # Настраиваем среду
    DEVICE, BATCH_SIZE, CHUNK_SIZE = auto_configure_env()
    
    print("Создаем синтетический датасет (5 длинных записей)...")
    database = [generate_bat_spectrogram(T=random.randint(10000, 20000)).to(DEVICE) for _ in range(5)]
    
    # 800 кусков за эпоху — достаточно, чтобы модель хорошо поучилась
    dataset = BatDataset(database, chunk_size=CHUNK_SIZE, samples_per_epoch=800)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = Audio2DAutoencoder().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Наша новая умная Loss-функция (штрафует за "мыло" на пиках в 10 раз сильнее)
    criterion = SparseSpectrogramLoss(peak_weight=10.0)
    
    test_chunk = database[0][:, :, 1000:1512].unsqueeze(0).clone()
    
    print("\n--- НАЧИНАЕМ ОБУЧЕНИЕ ---")
    model.train()
    epochs = 20 # Увеличил до 20, так как батчи стали больше и обучение пойдет быстрее
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Эпоха [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # ==========================================
    # 6. ВЫВОД РЕЗУЛЬТАТОВ (plt.show для Colab)
    # ==========================================
    model.eval()
    with torch.no_grad():
        reconstructed_chunk = model(test_chunk)
        
    orig_img = test_chunk.squeeze().cpu().numpy()
    recon_img = reconstructed_chunk.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].imshow(orig_img, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title("Оригинал: Чирпы, Эхо и Шум")
    axes[0].set_ylabel("Частота")
    
    axes[1].imshow(recon_img, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title("Восстановление (Loss: L1 + Peak Weight)")
    axes[1].set_ylabel("Частота")
    axes[1].set_xlabel("Время")
    
    plt.tight_layout()
    plt.show() # Идеально для Google Colab