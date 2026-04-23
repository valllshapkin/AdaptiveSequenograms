from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

# =====================================================================
# 1. Свертка: 1D Сигнал * 1D Окно -> 1D Сигнал
# =====================================================================
class AnchoredConv1D_1DKernel(nn.Module):
    def __init__(self, data: Float[Tensor, "... W"], zero_index: int):
        super().__init__()
        if isinstance(data, nn.Parameter):
            self.data = data 
        else:
            self.register_buffer('data', data)
        self.zero_index = zero_index

    def forward(self, signal: Float[Tensor, "... T"]) -> Float[Tensor, "... T"]:
        T, W = signal.shape[-1], self.data.shape[-1]
        N = T + W - 1
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        Z = self.zero_index
        return out[..., Z : Z + T]

# =====================================================================
# 2. Свертка: 1D Сигнал * 2D Окно -> 2D Спектрограмма
# =====================================================================
class AnchoredConv1D_2DKernel(nn.Module):
    def __init__(self, data: Float[Tensor, "... F W"], zero_index: int):
        super().__init__()
        if isinstance(data, nn.Parameter):
            self.data = data 
        else:
            self.register_buffer('data', data)
        self.zero_index = zero_index

    def forward(self, signal: Float[Tensor, "... T"]) -> Float[Tensor, "... F T"]:
        T, W = signal.shape[-1], self.data.shape[-1]
        N = T + W - 1
        sig_expanded = signal.unsqueeze(-2)
        Sig_f = torch.fft.rfft(sig_expanded, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        Z = self.zero_index
        return out[..., Z : Z + T]

# =====================================================================
# 3. Свертка: 2D Сигнал * 2D Окно -> 2D Спектрограмма (Используется для генерации)
# =====================================================================
class AnchoredConv2D_2DKernel(nn.Module):
    def __init__(self, data: Float[Tensor, "... F W"], zero_index: int):
        super().__init__()
        if isinstance(data, nn.Parameter):
            self.data = data 
        else:
            self.register_buffer('data', data)
        self.zero_index = zero_index

    def forward(self, signal: Float[Tensor, "... F T"]) -> Float[Tensor, "... F T"]:
        F_sig, T = signal.shape[-2], signal.shape[-1]
        F_ker, W = self.data.shape[-2], self.data.shape[-1]
        N = T + W - 1
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        Z = self.zero_index
        return out[..., Z : Z + T]

# =====================================================================
# НОВОЕ: 4. Слой эха с ГАРАНТИРОВАННЫМ монотонным убыванием и нормализацией
# =====================================================================
class MonotonicEchoConv(nn.Module):
    def __init__(self, F_bins: int, E_frames: int, zero_index: int = 0):
        super().__init__()
        self.F_bins = F_bins
        self.E_frames = E_frames
        self.zero_index = zero_index
        
        # Обучаемый параметр: логарифм скорости затухания на каждом шаге.
        # Инициализируем небольшим значением (-4.6 в softplus дает ~0.01)
        self.raw_decay = nn.Parameter(torch.full((F_bins, E_frames - 1), -4.6))

    def get_kernel(self) -> Tensor:
        # 1. Затухание должно быть строго положительным
        decay = torch.nn.functional.softplus(self.raw_decay)
        
        # 2. Добавляем нулевое затухание для t=0 (чтобы exp(0) = 1)
        zero_pad = torch.zeros(self.F_bins, 1, device=decay.device)
        decay_padded = torch.cat([zero_pad, decay], dim=-1)
        
        # 3. Кумулятивная сумма гарантирует монотонный рост показателя степени
        exponent = torch.cumsum(decay_padded, dim=-1)
        
        # 4. Берем экспоненту (строго монотонное убывание)
        E_unnorm = torch.exp(-exponent)
        
        # 5. Строгая нормализация на 1 для каждого частотного бина
        E_norm = E_unnorm / (E_unnorm.sum(dim=-1, keepdim=True) + 1e-12)
        return E_norm

    def forward(self, signal: Float[Tensor, "... F T"]) -> Float[Tensor, "... F T"]:
        kernel = self.get_kernel()
        T, W = signal.shape[-1], self.E_frames
        N = T + W - 1
        
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(kernel, n=N, dim=-1)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        return out[..., self.zero_index : self.zero_index + T]


# =====================================================================
# Вспомогательные функции для генерации данных
# =====================================================================
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from dataclasses import dataclass

@dataclass
class Trace:
    t: list | np.ndarray
    f: list | np.ndarray
    i: list | np.ndarray
    def __post_init__(self):
        self.t, self.f, self.i = map(lambda x: np.asarray(x, dtype=float), (self.t, self.f, self.i))

def render_template_nmf(traces, f_axis, dt, template_frames, sigma_t, sigma_f):
    df = f_axis[1] - f_axis[0]
    t_min, t_max = min(np.min(tr.t) for tr in traces), max(np.max(tr.t) for tr in traces)
    t_axis = np.arange(t_min - 0.05, t_max + 0.05, dt)
    canvas = np.zeros((len(t_axis), len(f_axis)), dtype=np.float32)

    rad_t, rad_f = max(1, int(3 * sigma_t / dt)), max(1, int(3 * sigma_f / df))
    PT, PF = np.meshgrid(np.arange(-rad_t, rad_t + 1) * dt, np.arange(-rad_f, rad_f + 1) * df, indexing='ij')
    base_patch = np.exp(-0.5 * ((PT / sigma_t)**2 + (PF / sigma_f)**2))

    for trace in traces:
        if len(trace.t) < 2: continue
        tck, _ = splprep([trace.t, trace.f, trace.i], s=0, k=min(3, len(trace.t) - 1))
        t_dense, f_dense, i_dense = splev(np.linspace(0, 1, int((t_max - t_min) / dt) * 4), tck)

        for t_val, f_val, i_val in zip(t_dense, f_dense, i_dense):
            if i_val <= 0: continue
            idx_t, idx_f = int(round((t_val - t_axis[0]) / dt)), int(round((f_val - f_axis[0]) / df))
            mt_start, mt_end = max(0, idx_t - rad_t), min(canvas.shape[0], idx_t + rad_t + 1)
            mf_start, mf_end = max(0, idx_f - rad_f), min(canvas.shape[1], idx_f + rad_f + 1)
            if mt_start >= mt_end or mf_start >= mf_end: continue
            pt_start, pf_start = mt_start - (idx_t - rad_t), mf_start - (idx_f - rad_f)
            scaled = base_patch[pt_start:pt_start+(mt_end-mt_start), pf_start:pf_start+(mf_end-mf_start)] * i_val
            canvas[mt_start:mt_end, mf_start:mf_end] = np.maximum(canvas[mt_start:mt_end, mf_start:mf_end], scaled)

    mass_per_t = canvas.sum(axis=1)
    center_idx = int(np.average(np.arange(len(t_axis)), weights=mass_per_t)) if mass_per_t.sum() > 0 else len(t_axis)//2
    out_matrix = np.zeros((template_frames, len(f_axis)), dtype=np.float32)
    start_t, end_t = center_idx - template_frames // 2, center_idx + template_frames // 2
    src_start, src_end = max(0, start_t), min(len(t_axis), end_t)
    dst_start = max(0, -start_t)
    if src_start < src_end:
        out_matrix[dst_start:dst_start+(src_end-src_start), :] = canvas[src_start:src_end, :]
    return out_matrix.T


# =========================================================
# 0. ОПРЕДЕЛЕНИЕ УСТРОЙСТВА
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")


# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ
# ==========================================
F_BINS, F_MAX = 300, 120_000.0
f_axis = np.linspace(0, F_MAX, F_BINS)
dt_step = 0.000_1
TEMPLATE_FRAMES = 64
TIME_STEPS = 100_000 // 2
K_TEMPLATES = 2
ECHO_FRAMES = 2000

# --- ШАГ A: Шаблоны (W) ---
W_1_np = render_template_nmf([Trace([0.677, 0.678, 0.679, 0.681, 0.6825], [85000, 65000, 38000, 25000, 22000], [0.2, 0.6, 1.0, 0.7, 0.1])], f_axis, dt_step, TEMPLATE_FRAMES, 0.0001, 1500.0)
W_2_np = render_template_nmf([Trace([0.677, 0.678, 0.679, 0.681, 0.6825], [42500, 36000, 27000, 15500, 11000], [0.2, 0.55, 0.85, 0.65, 0.1]), Trace([0.678, 0.679, 0.680], [59000, 42000, 30000], [0.04, 0.13, 0.04])], f_axis, dt_step, TEMPLATE_FRAMES, 0.0001, 1500.0)
W_tensor = torch.tensor(np.stack([W_1_np, W_2_np]), dtype=torch.float32).to(device)

# --- ШАГ B: Идеальные Секвенограммы (H) ---
H_tensor = torch.zeros((K_TEMPLATES, TIME_STEPS), dtype=torch.float32).to(device)
events_t1 = list(i//2 for i in [5_000, 15_000, 27_000, 38_000, 42_000, 51_000, 68_000, 75_000, 88_000, 95_000])
events_t2 = list(i//2 for i in [8_000, 22_000, 39_500, 40_500, 50_000, 60_000, 72_000, 85_000, 92_000])
for idx in events_t1: H_tensor[0, idx] = 1.0
for idx in events_t2: H_tensor[1, idx] = 1.0

# --- ШАГ C: Чистая спектрограмма (V) ---
conv_W_gen = AnchoredConv1D_2DKernel(W_tensor, zero_index=TEMPLATE_FRAMES // 2).to(device)
V_tensor = conv_W_gen(H_tensor).sum(dim=0)

# --- ШАГ D: Эхо-профиль (E) ---
E_np = np.zeros((F_BINS, ECHO_FRAMES), dtype=np.float32)
t_echo = np.arange(ECHO_FRAMES) * dt_step
tau_low_freq, tau_high_freq = 0.05, 0.005
taus = np.linspace(tau_low_freq, tau_high_freq, F_BINS)
for f in range(F_BINS):
    decay = np.exp(-t_echo / taus[f])
    decay /= decay.sum()
    E_np[f, :] = decay
E_tensor = torch.tensor(E_np, dtype=torch.float32).to(device)

# --- ШАГ E: Реверберирующая спектрограмма (S) ---
conv_E_gen = AnchoredConv2D_2DKernel(E_tensor, zero_index=0).to(device)
S_tensor = conv_E_gen(V_tensor)


# =========================================================
# 2. ПОДГОТОВКА МОДЕЛИ
# =========================================================
print("Инициализация слепой деконволюции (ConvNMF)...")

conv_W = AnchoredConv1D_2DKernel(data=W_tensor, zero_index=TEMPLATE_FRAMES // 2).to(device)

torch.manual_seed(42)
H_init = torch.rand((K_TEMPLATES, TIME_STEPS), dtype=torch.float32, device=device) * 0.1
H_param = nn.Parameter(H_init)

# ИСПОЛЬЗУЕМ НОВЫЙ СЛОЙ ЭХА
E_layer = MonotonicEchoConv(F_BINS, ECHO_FRAMES, zero_index=0).to(device)

# =========================================================
# 3. СТРУКТУРНЫЕ ШТРАФЫ
# =========================================================
TAU_W_MS = 50.0
tau_w_frames = int((TAU_W_MS / 1000.0) / dt_step)
w_iso_data = torch.ones(1, tau_w_frames * 2 + 1, device=device)
w_iso_data[0, tau_w_frames] = 0.0
w_iso_layer = AnchoredConv1D_1DKernel(data=w_iso_data, zero_index=tau_w_frames).to(device)

# =========================================================
# 4. ОБУЧЕНИЕ
# =========================================================
# Обучаем H и сырые параметры затухания эха. Для E можно поставить LR чуть выше.
optimizer = torch.optim.Adam([
    {'params': H_param, 'lr': 0.01},
    {'params': E_layer.raw_decay, 'lr': 0.05}
])
BETA_REG = 1.0 
EPOCHS = 400 # Чуть больше эпох для надежности сходимости

print("Начало оптимизации (на GPU)...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    # Прямой проход: слой эха автоматически генерирует строго монотонное, отнормированное ядро
    V_hat = conv_W(H_param).sum(dim=0)
    S_hat = E_layer(V_hat)
    
    eps_noise = 1e-5
    S_safe = S_tensor + eps_noise
    S_hat_safe = S_hat + eps_noise
    L_rec = torch.sum(S_safe * torch.log(S_safe / S_hat_safe) - S_safe + S_hat_safe) / F_BINS
    
    L_sharp = torch.sum(H_param * (1.0 - H_param))
    density = w_iso_layer(H_param)
    L_iso = 0.5 * torch.sum(H_param * density)
    L_reg = torch.sqrt(L_sharp**2 + L_iso**2 + 1e-4)
    
    loss = L_rec + BETA_REG * L_reg
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_([H_param, E_layer.raw_decay], max_norm=1.0)
    optimizer.step()

    # Жесткое ограничение + туннелирование через барьер 0.5
    with torch.no_grad():
        H_param.clamp_(0.0, 1.0)
        
        # Вероятность туннелировать = exp(-k * |H - 0.5|)
        # Чем ближе к 0.5 — тем выше шанс отразиться на другую сторону
        TUNNEL_K = 25  # Крутизна барьера (больше = туже барьер)
        dist_to_barrier = torch.abs(H_param - 0.5)
        p_tunnel = -4*H_param*(H_param-1)/2
        
        mask = torch.bernoulli(p_tunnel**4).bool()
        H_param[mask] = 1.0 - H_param[mask]
        H_param.clamp_(0.0, 1.0)

    if epoch % 15 == 0 or epoch == EPOCHS - 1:
        theta_deg = np.degrees(torch.atan2(L_iso, L_sharp).item())
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
              f"KL/F: {L_rec.item():.4f} | Sharp: {L_sharp.item():.4f} | Iso: {L_iso.item():.4f}")

print("Обучение завершено!\n")

# =========================================================
# 5. ВИЗУАЛИЗАЦИЯ (ИСПРАВЛЕННАЯ)
# =========================================================
print("Рендеринг результатов...")
with torch.no_grad():
    V_final = conv_W(H_param).sum(dim=0)
    S_final = E_layer(V_final)
    
    E_final_np = E_layer.get_kernel().cpu().numpy()
    H_final_np = H_param.cpu().numpy()
    S_tensor_cpu = S_tensor.cpu().numpy()
    S_final_cpu = S_final.cpu().numpy()
    V_final_cpu = V_final.cpu().numpy()
    V_true_cpu = conv_W(H_tensor).sum(0).cpu().numpy()
    H_tensor_cpu = H_tensor.cpu().numpy()
    E_tensor_cpu = E_tensor.cpu().numpy()

# Задаем нужные границы (можете менять, теперь ничего не сломается)
T_START, T_END = 35_000, 45_000

# Физические рамки (extent) для спектрограмм [x_min, x_max, y_min, y_max]
extent_spec = [T_START * dt_step, T_END * dt_step, 0, F_MAX / 1000]
# Физические рамки для эха (время в миллисекундах)
extent_echo = [0, ECHO_FRAMES * dt_step * 1000, 0, F_MAX / 1000]

t_axis = np.arange(T_START, T_END) * dt_step

# Создаем кастомную сетку графиков, чтобы Эхо (E) имело независимую ось X
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 0.8, 1.5, 1.5], hspace=0.3)

# Колонки True
ax_S_true = fig.add_subplot(gs[0, 0])
ax_H_true = fig.add_subplot(gs[1, 0], sharex=ax_S_true)
ax_V_true = fig.add_subplot(gs[2, 0], sharex=ax_S_true)
ax_E_true = fig.add_subplot(gs[3, 0]) # Ось X независима!

# Колонки Pred
ax_S_pred = fig.add_subplot(gs[0, 1], sharex=ax_S_true, sharey=ax_S_true)
ax_H_pred = fig.add_subplot(gs[1, 1], sharex=ax_S_true, sharey=ax_H_true)
ax_V_pred = fig.add_subplot(gs[2, 1], sharex=ax_S_true, sharey=ax_V_true)
ax_E_pred = fig.add_subplot(gs[3, 1], sharex=ax_E_true, sharey=ax_E_true)

vmax_spec = S_tensor_cpu[:, T_START:T_END].max() * 1.2
vmax_echo = E_tensor_cpu.max() * 1.2

# --- 1. Наблюдаемые спектрограммы (S) ---
ax_S_true.set_title("1A. Истинная наблюдаемая спектрограмма (S)")
ax_S_true.imshow(S_tensor_cpu[:, T_START:T_END], aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec, extent=extent_spec)
ax_S_true.set_ylabel("Частота (кГц)")

ax_S_pred.set_title("1B. Реконструированная спектрограмма (S_hat)")
ax_S_pred.imshow(S_final_cpu[:, T_START:T_END], aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec, extent=extent_spec)

# --- 2. Секвенограммы (H) ---
ax_H_true.set_title("2A. Истинные Секвенограммы (H_true)")
ax_H_true.plot(t_axis, H_tensor_cpu[0, T_START:T_END], color='red', label='H1')
ax_H_true.plot(t_axis, H_tensor_cpu[1, T_START:T_END], color='blue', alpha=0.7, label='H2')
ax_H_true.set_ylim(0, 1.1)
ax_H_true.set_ylabel("Амплитуда")

ax_H_pred.set_title("2B. Извлеченные Секвенограммы (H_pred)")
ax_H_pred.plot(t_axis, H_final_np[0, T_START:T_END], color='red', label='H1_pred')
ax_H_pred.plot(t_axis, H_final_np[1, T_START:T_END], color='blue', alpha=0.7, label='H2_pred')

# --- 3. Чистые спектрограммы (V) ---
ax_V_true.set_title("3A. Истинная чистая спектрограмма (V_true)")
ax_V_true.imshow(V_true_cpu[:, T_START:T_END], aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec, extent=extent_spec)
ax_V_true.set_xlabel("Абсолютное время (с)")
ax_V_true.set_ylabel("Частота (кГц)")

ax_V_pred.set_title("3B. Очищенная спектрограмма (V_pred)")
ax_V_pred.imshow(V_final_cpu[:, T_START:T_END], aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec, extent=extent_spec)
ax_V_pred.set_xlabel("Абсолютное время (с)")

# --- 4. Профили эха (E) ---
ax_E_true.set_title("4A. Истинный профиль Эха (E_true)")
ax_E_true.imshow(E_tensor_cpu, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax_echo, extent=extent_echo)
ax_E_true.set_xlabel("Относительная задержка (мс)")
ax_E_true.set_ylabel("Частота (кГц)")

ax_E_pred.set_title("4B. Изученный профиль Эха (E_pred) - Без Пиков!")
ax_E_pred.imshow(E_final_np, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax_echo, extent=extent_echo)
ax_E_pred.set_xlabel("Относительная задержка (мс)")

plt.show()