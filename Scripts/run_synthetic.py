import sys
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from AdaptiveSequenograms.model import fft_conv1d, BlindEchoDeconv
from AdaptiveSequenograms.trainer import train_model

# Настройки
torch.manual_seed(42)
T = 15000
F_bins = 300
K = 2
W_len = 150  # Сделали паттерны шире
E_len = 400  # Сделали эхо длиннее
bounds = [(50, 150), (180, 260)]

print("1. Generating synthetic ground truth...")
H_true = torch.zeros(K, T)
# Делаем редкие, но четкие пики
H_true[0, [2000, 8000, 13000]] = torch.tensor([1.0, 0.8, 0.9])
H_true[1, [4000, 10000]] = torch.tensor([0.9, 1.0])

W_true = []
# Шаблон 1: Восходящий частотный свип (Up-chirp)
f_h0 = bounds[0][1] - bounds[0][0]
tt0 = torch.linspace(0, 1, W_len).view(1, -1)
ff0 = torch.linspace(0, 1, f_h0).view(-1, 1)
W_true.append(torch.exp(-((ff0 - tt0)**2) * 40))

# Шаблон 2: Нисходящий свип с гармоникой (Down-chirp + harmonic)
f_h1 = bounds[1][1] - bounds[1][0]
ff1 = torch.linspace(0, 1, f_h1).view(-1, 1)
w1_main = torch.exp(-((ff1 - (1 - tt0))**2) * 40)
w1_harm = 0.6 * torch.exp(-((ff1 - (0.7 - tt0))**2) * 40)
W_true.append(w1_main + w1_harm)

# Адаптивное эхо: высокие частоты затухают гораздо быстрее
E_true = torch.zeros(F_bins, E_len)
time_axis = torch.arange(E_len).float()
for f in range(F_bins):
    decay_rate = 0.005 + 0.02 * (f / F_bins)**2
    E_true[f, :] = torch.exp(-decay_rate * time_axis)

# Сборка чистой спектрограммы
V_clean = torch.zeros(F_bins, T)
for k in range(K):
    h_k = H_true[k].unsqueeze(0).expand(W_true[k].shape[0], -1)
    V_clean[bounds[k][0]:bounds[k][1], :] += fft_conv1d(h_k, W_true[k], centered=True)

# Добавляем статичный фоновый шум (0.2) + гауссовский шум
bg_noise_true = 0.2
S_data = F.relu(fft_conv1d(V_clean, E_true, centered=False) + bg_noise_true + torch.randn(F_bins, T) * 0.05)

print("2. Training model...")
model = BlindEchoDeconv(W_true, bounds, T, F_bins, E_len)
# Увеличили epochs до 300, lr до 0.03, чуть ослабили L1 для роста пиков
model = train_model(model, S_data, epochs=300, lr=0.03, lambda_l1=0.2)

print("3. Plotting and saving results...")
with torch.no_grad():
    H_pred = model.H.detach().cpu()
    S_pred = model().detach().cpu()
    E_pred = model.E.detach().cpu()

fig, axs = plt.subplots(6, 1, figsize=(12, 22))

# 0. Таргет
im0 = axs[0].imshow(S_data.numpy(), aspect='auto', origin='lower', cmap='magma', vmax=2.0)
axs[0].set_title("Target Spectrogram (Signal + Echo + Noise)")
fig.colorbar(im0, ax=axs[0], pad=0.01)

# 1. Реконструкция
im1 = axs[1].imshow(S_pred.numpy(), aspect='auto', origin='lower', cmap='magma', vmax=2.0)
axs[1].set_title("Reconstructed Spectrogram")
fig.colorbar(im1, ax=axs[1], pad=0.01)

# 2. Секвенограмма 1
axs[2].plot(H_true[0].numpy(), label="True Seq 1 (Up-Chirp)", lw=3, color='blue', alpha=0.6)
axs[2].plot(H_pred[0].numpy(), '--', label="Pred Seq 1", color='red', lw=2)
axs[2].set_title("Sequenogram 1: Template Activation")
axs[2].grid(True, alpha=0.3)
axs[2].legend()

# 3. Секвенограмма 2
axs[3].plot(H_true[1].numpy(), label="True Seq 2 (Down-Chirp)", lw=3, color='green', alpha=0.6)
axs[3].plot(H_pred[1].numpy(), '--', label="Pred Seq 2", color='orange', lw=2)
axs[3].set_title("Sequenogram 2: Template Activation")
axs[3].grid(True, alpha=0.3)
axs[3].legend()

# 4. Срезы эха (True vs Pred)
f1, f2 = 50, 250
axs[4].plot(E_true[f1].numpy(), label=f"True Echo (Low Freq={f1})", lw=3, color='royalblue', alpha=0.6)
axs[4].plot(E_pred[f1].numpy(), '--', label=f"Pred Echo (Low Freq={f1})", color='navy', lw=2)
axs[4].plot(E_true[f2].numpy(), label=f"True Echo (High Freq={f2})", lw=3, color='lightcoral', alpha=0.6)
axs[4].plot(E_pred[f2].numpy(), '--', label=f"Pred Echo (High Freq={f2})", color='darkred', lw=2)
axs[4].set_title("Adaptive Echo Profiles (1D Frequency Slices)")
axs[4].grid(True, alpha=0.3)
axs[4].legend()

# 5. Двумерное окно эха (Learned Matrix E)
im5 = axs[5].imshow(E_pred.numpy(), aspect='auto', origin='lower', cmap='viridis')
axs[5].set_title("Learned 2D Echo Matrix (Frequency vs Time-Delay)")
axs[5].set_xlabel("Time Delay (frames)")
axs[5].set_ylabel("Frequency Bin")
fig.colorbar(im5, ax=axs[5], pad=0.01)

plt.tight_layout()

os.makedirs("Results", exist_ok=True)
save_path = "Results/synthetic_results.png"
plt.savefig(save_path, dpi=150)
print(f"✅ Success! Graphs saved to {save_path}")
