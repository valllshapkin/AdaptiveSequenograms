from pathlib import Path
import os
os.chdir(Path(__file__).parent)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОЙ СПЕКТРОГРАММЫ
# ==========================================
def generate_synthetic_spectrogram(size=(128, 128)):
    # 99% - Базовый Гауссовский шум (амплитуда ~0.1 - 0.2)
    noise = np.abs(np.random.normal(0.1, 0.1, size))
    signal = np.zeros(size)
    x = np.arange(size[1])
    
    # Сигнал 1: Жирная яркая кривая (Основное эхо, амплитуда ~ 5.0)
    y1 = np.clip(np.sin(x / 10.0) * 20 + 64, 0, size[0]-1).astype(int)
    for xi, yi in enumerate(y1):
        signal[max(0, yi-2):min(size[0], yi+3), xi] = 5.0 
        
    # Сигнал 2: Слабое размытое эхо (Вторичное эхо, амплитуда ~ 0.5)
    y2 = np.clip(np.sin(x / 15.0 + 2) * 25 + 90, 0, size[0]-1).astype(int)
    for xi, yi in enumerate(y2):
        signal[max(0, yi-6):min(size[0], yi+7), xi] = 0.5 

    # Сглаживаем сигнал, чтобы он был похож на спектральные паттерны
    signal = gaussian_filter(signal, sigma=1.5)
    spectrogram = noise + signal
    
    # Возвращаем тензор [B, C, H, W]
    return torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ==========================================
# 2. ПРОСТАЯ МОДЕЛЬ (АВТОЭНКОДЕР)
# ==========================================
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2), nn.ReLU() 
            # ReLU гарантирует, что предсказания >= 0 (спектрограмма не бывает < 0)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

# ==========================================
# 3. ФУНКЦИИ ОШИБКИ И КАСТОМНЫЙ ГРАДИЕНТ
# ==========================================
def loss_mse(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

def loss_l1(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))

def loss_itakura_saito(y_pred, y_true, eps=1e-4):
    y_pred = torch.clamp(y_pred, min=eps)
    y_true = torch.clamp(y_true, min=eps)
    ratio = y_true / y_pred
    return torch.mean(ratio - torch.log(ratio) - 1)

def loss_multi_res(y_pred, y_true, eps=1e-4):
    lin_loss = torch.mean(torch.abs(y_pred - y_true))
    log_loss = torch.mean(torch.abs(torch.log(y_pred + eps) - torch.log(y_true + eps)))
    return lin_loss + log_loss

# НАШ КАСТОМНЫЙ ГРАДИЕНТ (Глушит шум, учит только структуру)
class SNR_Gated_Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_pred, y_true):
        ctx.save_for_backward(y_pred, y_true)
        # Возвращаем "пустышку", чтобы PyTorch мог построить граф от y_pred
        return torch.sum(y_pred * 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors
        eps = 1e-4
        
        # Маска SNR: если пиксель ярче 0.4, это сигнал (вес ~1), если меньше - шум (вес -> 0)
        # В реальной задаче здесь можно использовать локальную дисперсию
        mask = torch.sigmoid((y_true - 0.4) * 20.0)
        
        # Градиент относительной ошибки, умноженный на маску
        grad = mask * (y_pred - y_true) / (y_true + eps)
        
        return grad * grad_output, None

def loss_custom_gated(y_pred, y_true):
    return SNR_Gated_Gradient.apply(y_pred, y_true)

# ==========================================
# 4. ЦИКЛ ОБУЧЕНИЯ
# ==========================================
def train_model(model_name, loss_fn, target, epochs=150):
    print(f"Обучение: {model_name}...")
    torch.manual_seed(42) # Одинаковая инициализация для честности
    model = SimpleAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    frames =[]
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(target)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        
        # Сохраняем кадр каждые 3 эпохи
        if epoch % 3 == 0:
            frames.append(pred.detach().numpy()[0, 0].copy())
            
    return frames

# ==========================================
# 5. ГЕНЕРАЦИЯ И РЕНДЕР АНИМАЦИИ
# ==========================================
if __name__ == "__main__":
    target_spec = generate_synthetic_spectrogram()
    
    # Словарь экспериментов
    experiments = {
        "1. MSE Loss\n(Размытие)": loss_mse,
        "2. L1 Loss\n(Теряет слабые)": loss_l1,
        "3. Itakura-Saito\n(Нестабилен, но видит все)": loss_itakura_saito,
        "4. Multi-Res (Lin+Log)\n(Индустриальный стандарт)": loss_multi_res,
        "5. Custom SNR-Gated\n(Игнорирует шум)": loss_custom_gated
    }
    
    all_frames = {}
    for name, fn in experiments.items():
        all_frames[name] = train_model(name, fn, target_spec, epochs=150)
        
    num_frames = len(list(all_frames.values())[0])
    
    # Настройка графика
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    axes = axes.flatten()
    
    # Отображаем Target (Ожидание)
    target_img = target_spec.numpy()[0, 0]
    axes[0].imshow(target_img, cmap='magma', origin='lower', vmin=0, vmax=5)
    axes[0].set_title("Target (Ожидание)\n1% эхо, 99% шум")
    axes[0].axis('off')
    
    # Подготовка холстов для анимации
    im_plots =[]
    for i, name in enumerate(experiments.keys()):
        ax = axes[i+1]
        im = ax.imshow(np.zeros_like(target_img), cmap='magma', origin='lower', vmin=0, vmax=5)
        ax.set_title(name)
        ax.axis('off')
        im_plots.append(im)
        
    fig.suptitle("Обучение автоэнкодера спектрограмм: Битва Loss'ов", fontsize=16)

    def update(frame_idx):
        for i, name in enumerate(experiments.keys()):
            im_plots[i].set_data(all_frames[name][frame_idx])
        return im_plots

    print("Рендеринг GIF (это займет секунд 10-20)...")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
    
    # Сохраняем как GIF с помощью Pillow (не требует установки ffmpeg)
    ani.save('spectrogram_training.gif', writer='pillow', fps=10)
    print("Готово! Откройте файл 'spectrogram_training.gif'")