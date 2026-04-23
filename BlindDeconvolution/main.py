import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
os.chdir(Path(__file__).parent)

# =========================================================
# 0. УСТРОЙСТВО И НАСТРОЙКИ
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

ECHO_FRAMES = 500  # Длина окна эха

# =====================================================================
# 1. ДВИЖОК СВЕРТКИ И СЛОЙ ЭХА С ПИКОМ (ИНТЕГРАЦИЯ ПРОИЗВОДНЫХ)
# =====================================================================
class PeakedEchoConv(nn.Module):
    """Слой эха с пиком. Интегрирует строго положительные и отрицательные производные из центра."""
    def __init__(self, F_bins: int, E_frames: int, peak_ratio: float = 0.25):
        super().__init__()
        self.F_bins = F_bins
        self.E_frames = E_frames
        self.peak_idx = int(E_frames * peak_ratio) # Индекс максимума (для 500 фреймов = 125)
        
        # Инициализируем производные (скорости затухания/роста)
        # -4.6 после softplus даст шаг ~0.01
        
        # Производные для левой части (рост к пику)
        init_left = torch.full((F_bins, self.peak_idx), -4.6, dtype=torch.float32)
        self.raw_left = nn.Parameter(init_left)
        
        # Производные для правой части (спад после пика)
        init_right = torch.full((F_bins, E_frames - self.peak_idx - 1), -4.6, dtype=torch.float32)
        self.raw_right = nn.Parameter(init_right)

    def get_kernel(self) -> Tensor:
        # 1. ПРАВАЯ ЧАСТЬ (Интегрируем вправо от пика)
        step_right = torch.nn.functional.softplus(self.raw_right) # Строго > 0
        drop_right = torch.cumsum(step_right, dim=-1)             # Монотонно растет
        A_right = -drop_right                                     # Монотонно убывает в минуса
        
        # 2. ЛЕВАЯ ЧАСТЬ (Интегрируем влево от пика)
        step_left = torch.nn.functional.softplus(self.raw_left)   # Строго > 0
        # Разворачиваем, суммируем от пика к краю, разворачиваем обратно
        drop_left = torch.cumsum(torch.flip(step_left, dims=[-1]), dim=-1)
        A_left = -torch.flip(drop_left, dims=[-1])                # Растет к нулю в сторону пика
        
        # 3. ЦЕНТР (Сам пик)
        A_peak = torch.zeros(self.F_bins, 1, device=step_right.device)
        
        # 4. Собираем массив псевдо-логарифмов амплитуды: [..., -0.5, -0.2, 0.0, -0.1, -0.6, ...]
        A_total = torch.cat([A_left, A_peak, A_right], dim=-1)
        
        # 5. Экспонента: в нуле будет 1, на краях плавное затухание
        k_unnorm = torch.exp(A_total)
        
        # 6. Нормализация, чтобы энергия не меняла масштаб V
        k_norm = k_unnorm / (k_unnorm.sum(dim=-1, keepdim=True) + 1e-8)
        return k_norm

    def forward(self, signal: Tensor) -> Tensor:
        kernel = self.get_kernel()
        T, W = signal.shape[-1], self.E_frames
        N = T + W - 1
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(kernel, n=N, dim=-1)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        
        # Якорная обрезка (Anchoring):
        # Поскольку пик ядра находится не на 0, а на self.peak_idx, 
        # эхо "опережает" сигнал. Мы сдвигаем окно чтения, чтобы выровнять V и S.
        return out[..., self.peak_idx : self.peak_idx + T]

# =========================================================
# 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =========================================================
file_name = "TestSpec.npy"

if os.path.exists(file_name):
    print(f"Загрузка {file_name}...")
    matrix_np = np.load(file_name)
else:
    print(f"Файл {file_name} не найден. Генерирую тестовую заглушку...")
    matrix_np = np.random.rand(14998, 200).astype(np.float32) * 0.1
    matrix_np[5000:5010, 50:150] += 5.0 # Фейковый пик

# (T, F) -> (F, T)
matrix_np = matrix_np.T  
F_BINS, TIME_STEPS = matrix_np.shape

# Энергия и Нормализация (Критически важно!)
matrix_energy = matrix_np ** 2
max_val = matrix_energy.max()
matrix_energy = matrix_energy / max_val

S_target = torch.tensor(matrix_energy, dtype=torch.float32, device=device)

# =========================================================
# 3. ПОДГОТОВКА МОДЕЛИ И ШТРАФОВ
# =========================================================
V_param = nn.Parameter(S_target.clone())

# Подключаем наш новый слой ядра (пик будет на ~25%, то есть индекс 125)
E_layer = PeakedEchoConv(F_BINS, ECHO_FRAMES, peak_ratio=0.25).to(device)

optimizer = torch.optim.Adam([
    {'params': V_param, 'lr': 0.05},       
    {'params': E_layer.raw_left, 'lr': 0.05},  # Обучаем левый склон
    {'params': E_layer.raw_right, 'lr': 0.05}  # Обучаем правый склон
])

EPOCHS = 400

# ГИПЕРПАРАМЕТРЫ
LAMBDA_SPARSE = 0.02     
LAMBDA_SMOOTH_F = 500000.0 

# =========================================================
# 4. ЦИКЛ ОБУЧЕНИЯ
# =========================================================
print("Начало оптимизации...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    V_pos = torch.nn.functional.relu(V_param)
    S_hat_linear = E_layer(V_pos)

    # 1. Log-MSE Loss
    eps = 1e-6
    log_S_hat = torch.log(S_hat_linear + eps)
    log_S_target = torch.log(S_target + eps)
    loss_rec = torch.nn.functional.mse_loss(log_S_hat, log_S_target)

    # 2. Sparsity Loss
    loss_sparse = torch.mean(torch.sqrt(V_pos + 1e-8))

    # 3. 1D Smoothness Loss (По частоте)
    kernel = E_layer.get_kernel()
    diff_f = kernel[1:, :] - kernel[:-1, :]
    loss_smooth_f = torch.mean(diff_f**2)

    # Общий лосс
    loss = loss_rec + LAMBDA_SPARSE * loss_sparse + LAMBDA_SMOOTH_F * loss_smooth_f

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | "
              f"LogMSE: {loss_rec.item():.4f} | Sparse: {loss_sparse.item():.4f} | "
              f"Sm_F: {loss_smooth_f.item()*LAMBDA_SMOOTH_F:.4f}")

print("Обучение завершено!\n")

# =========================================================
# 5. ВИЗУАЛИЗАЦИЯ
# =========================================================
print("Рендеринг результатов...")
with torch.no_grad():
    V_final = torch.nn.functional.relu(V_param)
    S_final = E_layer(V_final)
    E_final = E_layer.get_kernel()
    
    S_target_db = 10 * np.log10(S_target.cpu().numpy() * max_val + 1e-5)
    S_final_db  = 10 * np.log10(S_final.cpu().numpy() * max_val + 1e-5)
    V_final_db  = 10 * np.log10(V_final.cpu().numpy() * max_val + 1e-5)
    E_final_cpu = E_final.cpu().numpy()

# --- ПОИСК МАКСИМАЛЬНОГО ПИКА ---
f_max_idx, t_max_idx = np.unravel_index(np.argmax(S_target_db), S_target_db.shape)

ZOOM_TIME_FRAMES = 400
ZOOM_FREQ_BINS = 100

t_start = max(0, t_max_idx - ZOOM_TIME_FRAMES // 4)
t_end   = min(TIME_STEPS, t_start + ZOOM_TIME_FRAMES)
f_start = max(0, f_max_idx - ZOOM_FREQ_BINS // 2)
f_end   = min(F_BINS, f_start + ZOOM_FREQ_BINS)

S_target_zoom = S_target_db[f_start:f_end, t_start:t_end]
S_final_zoom  = S_final_db[f_start:f_end, t_start:t_end]
V_final_zoom  = V_final_db[f_start:f_end, t_start:t_end]

extent_zoom = [t_start, t_end, f_start, f_end]
extent_echo = [0, ECHO_FRAMES, 0, F_BINS]

# --- ОТРИСОВКА ---
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1.5], hspace=0.35)

ax_S_true = fig.add_subplot(gs[0, 0])
ax_V_pred = fig.add_subplot(gs[1, 0], sharex=ax_S_true, sharey=ax_S_true)
ax_E_pred = fig.add_subplot(gs[2, 0]) 

ax_S_pred = fig.add_subplot(gs[0, 1], sharex=ax_S_true, sharey=ax_S_true)
ax_E_slice = fig.add_subplot(gs[2, 1])

vmin_db = np.percentile(S_target_zoom, 5) 
vmax_db = np.max(S_target_zoom)

ax_S_true.set_title(f"1A. Исходная S_target (ЗУМ) [дБ]")
ax_S_true.imshow(S_target_zoom, aspect='auto', origin='lower', cmap='magma', vmin=vmin_db, vmax=vmax_db, extent=extent_zoom)
ax_S_true.set_ylabel("Частотные бины")

ax_S_pred.set_title("1B. Реконструкция (V * E) [дБ]")
ax_S_pred.imshow(S_final_zoom, aspect='auto', origin='lower', cmap='magma', vmin=vmin_db, vmax=vmax_db, extent=extent_zoom)

ax_V_pred.set_title("2. Очищенный сигнал V_pred [дБ]")
ax_V_pred.imshow(V_final_zoom, aspect='auto', origin='lower', cmap='magma', vmin=vmin_db, vmax=vmax_db, extent=extent_zoom)
ax_V_pred.set_ylabel("Частотные бины")
ax_V_pred.set_xlabel("Абсолютные фреймы времени")

# --- Обновленные графики эха ---
ax_E_pred.set_title("3A. Эхо с Пиком (E_pred) [Линейный]")
im4 = ax_E_pred.imshow(E_final_cpu, aspect='auto', origin='lower', cmap='viridis', extent=extent_echo)
ax_E_pred.axvline(x=E_layer.peak_idx, color='red', linestyle='--', alpha=0.5, label='Пик') # Покажем где пик
ax_E_pred.legend()
ax_E_pred.set_ylabel("Все частотные бины")
ax_E_pred.set_xlabel("Фреймы задержки (эхо)")
fig.colorbar(im4, ax=ax_E_pred)

ax_E_slice.set_title("3B. Срезы эха (Интегрирование производных)")
freq_indices = [F_BINS//4, F_BINS//2, 3*F_BINS//4]
for f_idx in freq_indices:
    ax_E_slice.plot(E_final_cpu[f_idx, :], label=f'Бин {f_idx}', lw=2)
ax_E_slice.axvline(x=E_layer.peak_idx, color='red', linestyle='--', alpha=0.5)
ax_E_slice.legend()
ax_E_slice.set_xlabel("Фреймы задержки")
ax_E_slice.grid(True, alpha=0.3)

plt.show()