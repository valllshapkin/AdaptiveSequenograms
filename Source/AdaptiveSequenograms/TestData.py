import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from dataclasses import dataclass

# Импорт ваших кастомных FFT-сверток
from AdaptiveSequenograms.Tools import AnchoredConv1D_1DKernel, AnchoredConv1D_2DKernel, AnchoredConv2D_2DKernel

# ==========================================
# 0. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (Генерация W)
# ==========================================
@dataclass
class Trace:
    t: list | np.ndarray
    f: list | np.ndarray
    i: list | np.ndarray

    def __post_init__(self):
        self.t = np.asarray(self.t, dtype=float)
        self.f = np.asarray(self.f, dtype=float)
        self.i = np.asarray(self.i, dtype=float)

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


# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ (Выполняется при импорте)
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

W_tensor = torch.tensor(np.stack([W_1_np, W_2_np]), dtype=torch.float32)

# --- ШАГ B: Идеальные Секвенограммы (H) ---
H_tensor = torch.zeros((K_TEMPLATES, TIME_STEPS), dtype=torch.float32)

# Расширенный набор пиков (15 событий на 10 секунд)
events_t1 = list(i//2 for i in [5_000, 15_000, 27_000, 38_000, 42_000, 51_000, 68_000, 75_000, 88_000, 95_000])
events_t2 = list(i//2 for i in [8_000, 22_000, 39_500, 40_500, 50_000, 60_000, 72_000, 85_000, 92_000])

for idx in events_t1: H_tensor[0, idx] = 1.0
for idx in events_t2: H_tensor[1, idx] = 1.0

# --- ШАГ C: Чистая спектрограмма (V = Sum(H * W)) ---
conv_W = AnchoredConv1D_2DKernel(W_tensor, zero_index=TEMPLATE_FRAMES // 2)
V_tensor = conv_W(H_tensor).sum(dim=0)

# --- ШАГ D: Эхо-профиль (E) ---
E_tensor = torch.zeros((F_BINS, ECHO_FRAMES), dtype=torch.float32)
t_echo = np.arange(ECHO_FRAMES) * dt_step
tau_low_freq, tau_high_freq = 0.05, 0.005  # Затухание от 50мс до 5мс
taus = np.linspace(tau_low_freq, tau_high_freq, F_BINS)

for f in range(F_BINS):
    decay = np.exp(-t_echo / taus[f])
    decay /= decay.sum()  # Нормировка на 1 для каждого частотного бина!
    E_tensor[f, :] = torch.tensor(decay, dtype=torch.float32)

# --- ШАГ E: Реверберирующая спектрограмма с шумом (S = V * E + Noise) ---
conv_E = AnchoredConv2D_2DKernel(E_tensor, zero_index=0)
S_clean = conv_E(V_tensor)

# # Добавляем гауссов шум (5% от максимальной амплитуды сигнала)
# noise_level = 0.05 * S_clean.max()
# noise = torch.randn_like(S_clean) * noise_level

# # Для NMF спектрограмма должна быть неотрицательной, поэтому обрезаем отрицательный шум
# S_tensor = torch.clamp(S_clean + noise, min=0.0)
S_tensor = S_clean


# ==========================================
# 2. БЛОК ВИЗУАЛИЗАЦИИ (Только при прямом запуске)
# ==========================================
if __name__ == "__main__":
    print(f"Экспорт модулей готов:")
    print(f"  - W_tensor: {W_tensor.shape}")
    print(f"  - H_tensor: {H_tensor.shape}")
    print(f"  - V_tensor: {V_tensor.shape}")
    print(f"  - E_tensor: {E_tensor.shape}")
    print(f"  - S_tensor: {S_tensor.shape}")

    # Выбираем отрезок с плотным перекрытием сигналов
    T_START_PLOT, T_END_PLOT = 25_000, 45_000
    t_axis_plot = np.arange(T_START_PLOT, T_END_PLOT) * dt_step
    f_extent = [t_axis_plot[0], t_axis_plot[-1], 0, 120]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2, 1, 2]})

    axes[0].plot(t_axis_plot, H_tensor[0, T_START_PLOT:T_END_PLOT].numpy(), label="H_1 (MAIN_CALL)", color='red')
    axes[0].plot(t_axis_plot, H_tensor[1, T_START_PLOT:T_END_PLOT].numpy(), label="H_2 (SECOND_CALL)", color='blue', alpha=0.7)
    axes[0].set_title("Секвенограммы (H): Разреженные активации (фрагмент с перекрытием)")
    axes[0].set_ylabel("Амплитуда")
    axes[0].set_xlim(t_axis_plot[0], t_axis_plot[-1])
    axes[0].legend(loc="upper right")

    axes[1].imshow(V_tensor[:, T_START_PLOT:T_END_PLOT].numpy(), aspect='auto', origin='lower', cmap='magma', extent=f_extent)
    axes[1].set_title("Чистая спектрограмма (V = Sum(H * W))")
    axes[1].set_ylabel("Частота (кГц)")

    axes[2].imshow(E_tensor.numpy(), aspect='auto', origin='lower', cmap='viridis', extent=[0, ECHO_FRAMES * dt_step, 0, 120])
    axes[2].set_title("1D профиль Эха (E): Экспоненциальное затухание")
    axes[2].set_xlabel("Время эха (сек)")
    axes[2].set_ylabel("Частота (кГц)")

    axes[3].imshow(S_tensor[:, T_START_PLOT:T_END_PLOT].numpy(), aspect='auto', origin='lower', cmap='magma', extent=f_extent)
    axes[3].set_title("Наблюдаемая спектрограмма с реверберацией и шумом (S = V * E + Noise)")
    axes[3].set_xlabel("Глобальное время (сек)")
    axes[3].set_ylabel("Частота (кГц)")

    plt.tight_layout()
    plt.show()