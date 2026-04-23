from typing import Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import NMF
from tqdm.auto import tqdm
from NewSpec.Core.Functions import SpecFunc, TimeFunc
from NewSpec.Core.Units import UREG
# =====================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================================

def _functional_conv1d_fft_2dkernel(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет 1D свертку 2D сигнала и 2D ядра ТОЛЬКО по временной оси (последней).
    Ось частот должна совпадать.
    """
    assert signal.shape[-2] == kernel.shape[-2], "Размерность частот сигнала и ядра должна совпадать"
    
    T_sig, T_ker = signal.shape[-1], kernel.shape[-1]
    N_t = T_sig + T_ker - 1

    Sig_f = torch.fft.rfft(signal, n=N_t, dim=-1)
    Ker_f = torch.fft.rfft(kernel, n=N_t, dim=-1)
    
    out_f = Sig_f * Ker_f
    out = torch.fft.irfft(out_f, n=N_t, dim=-1)

    pad_t_start = (T_ker - 1) // 2
    return out[..., pad_t_start : pad_t_start + T_sig]

def _functional_match1d_fft_2dkernel(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет 1D кросс-корреляцию 2D сигнала и 2D ядра по временной оси.
    """
    # Для корреляции через свертку нужно отзеркалить ядро по оси свертки
    flipped_kernel = torch.flip(kernel, dims=(-1,))
    return _functional_conv1d_fft_2dkernel(signal, flipped_kernel)

def _total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Заставляет паттерн быть гладким, штрафуя резкие перепады."""
    tv_h = torch.sum(torch.abs(img[..., :-1, :] - img[..., 1:, :]))
    tv_w = torch.sum(torch.abs(img[..., :, :-1] - img[..., :, 1:]))
    return tv_h + tv_w

# =====================================================================
# ОСНОВНАЯ ФУНКЦИЯ ПОИСКА ПАТТЕРНА
# =====================================================================
def discover_best_pattern_with_callback(
    spec: SpecFunc,
    window_duration_ms: float = 100.0,
    anchor_freq_percent: float = 50.0,
    anchor_time_percent: float = 50.0,
    epochs: int = 500,
    learning_rate: float = 0.05,
    top_k_peaks: int = 20,
    lambda_tv: float = 0.01,
    lambda_center: float = 0.1,
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[SpecFunc, TimeFunc], None]] = None,
    callback_interval_percent: Optional[float] = 2.0
) -> Tuple[SpecFunc, TimeFunc]:
    """
    Находит один "идеальный" 2D паттерн, который наилучшим образом описывает 
    повторяющиеся события в спектрограмме, используя 1D свертку по времени.

    Args:
        spec: Входная спектрограмма.
        window_duration_ms: Длительность искомого паттерна в миллисекундах.
        anchor_freq_percent: Ожидаемый центр масс по частоте (в % от высоты).
        anchor_time_percent: Ожидаемый центр масс по времени (в % от ширины).
        epochs: Количество эпох обучения.
        learning_rate: Скорость обучения.
        top_k_peaks: Сколько самых сильных откликов использовать для расчета лосса.
        lambda_tv: Вес регуляризации гладкости (Total Variation).
        lambda_center: Общий вес штрафа за отклонение от якорей.
        device: Устройство для вычислений.
        callback: Функция для визуализации прогресса.
        callback_interval_percent: Как часто вызывать callback (в процентах от эпох).

    Returns:
        Кортеж из (найденный_паттерн, 1D_временная_карта_корреляций).
    """
    # --- 1. Подготовка данных ---
    S, S_unit = spec.values
    freq_axis, _ = spec.freq
    time_axis, _ = spec.time
    
    if device is None: device = S.device
    dtype = S.dtype
    target_spec = S.unsqueeze(0) if S.dim() == 2 else S
    target_spec = target_spec.to(device, dtype)
    
    dt_val, _ = spec.dt
    sr = 1.0 / dt_val
    num_freqs, _ = target_spec.shape[-2], target_spec.shape[-1]
    win_width = max(3, int(round((window_duration_ms / 1000.0) * sr)))
    
    # --- 2. Инициализация ---
    W_raw = nn.Parameter(torch.randn(1, num_freqs, win_width, device=device, dtype=dtype))
    optimizer = torch.optim.Adam([W_raw], lr=learning_rate)
    
    # Предрасчет для лосса центрирования
    target_center_freq = (num_freqs - 1) * (anchor_freq_percent / 100.0)
    target_center_time = (win_width - 1) * (anchor_time_percent / 100.0)
    freq_indices = torch.arange(num_freqs, device=device, dtype=dtype)
    time_indices = torch.arange(win_width, device=device, dtype=dtype)
    
    print(f"Поиск паттерна: окно {num_freqs}x{win_width}. Целевой центр (F,T): ({target_center_freq:.1f}, {target_center_time:.1f})")
    
    def _invoke_callback(current_W_raw: torch.Tensor, correlation_timeline: torch.Tensor):
        with torch.no_grad():
            W_norm = current_W_raw - current_W_raw.mean()
            W_norm /= (torch.norm(W_norm) + 1e-8)
            
            pattern_time_axis = torch.arange(win_width, device='cpu', dtype=dtype) * dt_val
            
            found_pattern_spec = SpecFunc(
                (W_norm.squeeze(0).cpu(), S_unit), 
                freq_axis.cpu(), pattern_time_axis)
            
            correlation_timeline_func = TimeFunc(
                (correlation_timeline.squeeze(0).cpu(), UREG.dimensionless),
                time_axis.cpu())
            
            if callback:
                callback(found_pattern_spec, correlation_timeline_func)
        return found_pattern_spec, correlation_timeline_func

    # --- 3. Цикл обучения ---
    pbar = tqdm(range(epochs), desc="Поиск паттерна")
    latest_results = None
    eps = 1e-8

    for epoch in pbar:
        optimizer.zero_grad()
        
        # --- ШАГ A: Дифференцируемая нормализация паттерна ---
        W_zero_mean = W_raw - W_raw.mean()
        W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
        
        # --- ШАГ B: Дифференцируемый ZNCC (с 1D сверткой по времени) ---
        numerator = _functional_match1d_fft_2dkernel(target_spec, W)
        
        U = torch.ones_like(W); N = W.numel()
        Sum_S = _functional_match1d_fft_2dkernel(target_spec, U)
        Sum_S2 = _functional_match1d_fft_2dkernel(target_spec**2, U)
        
        local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
        denominator = torch.sqrt(local_variance) + eps
        
        # Карта корреляций все еще 2D (F, T)
        zncc_map_2d = numerator / denominator
        # Схлопываем по частоте, чтобы получить 1D временную функцию
        correlation_timeline = zncc_map_2d.sum(dim=-2)
        
        # --- ШАГ C: Вычисление Loss-функции ---
        # 1. Лосс "заметности" для 1D временной функции
        top_peaks, _ = torch.topk(correlation_timeline.flatten(), k=top_k_peaks)
        loss_match = -torch.mean(top_peaks) 
        
        # 2. Лосс гладкости (TV Loss)
        loss_tv = lambda_tv * _total_variation_loss(W_raw)
        
        # 3. Лосс центрирования по двум осям
        energy_density = W_raw**2
        total_mass = energy_density.sum() + eps
        
        # Центр масс по времени
        time_density = energy_density.sum(dim=-2)
        com_time = (time_density * time_indices).sum() / total_mass
        loss_center_time = F.mse_loss(com_time, torch.full_like(com_time, target_center_time))
        
        # Центр масс по частоте
        freq_density = energy_density.sum(dim=-1)
        com_freq = (freq_density * freq_indices).sum() / total_mass
        loss_center_freq = F.mse_loss(com_freq, torch.full_like(com_freq, target_center_freq))
        
        loss_center = lambda_center * (loss_center_time + loss_center_freq)
        
        total_loss = loss_match + loss_tv + loss_center
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{total_loss.item():.3f}", Match=f"{-loss_match.item():.3f}", Cntr=f"{loss_center.item():.3f}")

        # ... (Callback и код после цикла)
        is_last_epoch = (epoch == epochs - 1)
        if callback_interval_percent > 0 and ((epoch + 1) % int(epochs * callback_interval_percent / 100.0) == 0 or is_last_epoch):
            latest_results = _invoke_callback(W_raw, correlation_timeline)

    if latest_results is None:
        with torch.no_grad():
            W_zero_mean = W_raw - W_raw.mean()
            W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
            numerator = _functional_match1d_fft_2dkernel(target_spec, W)
            U = torch.ones_like(W); N = W.numel()
            Sum_S = _functional_match1d_fft_2dkernel(target_spec, U)
            Sum_S2 = _functional_match1d_fft_2dkernel(target_spec**2, U)
            local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
            denominator = torch.sqrt(local_variance) + eps
            final_zncc_map_2d = numerator / denominator
            final_timeline = final_zncc_map_2d.sum(dim=-2)
            latest_results = _invoke_callback(W_raw, final_timeline)

    return latest_results

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    F_BINS, TIME_STEPS = 128, 4000
    spec_tensor = torch.randn(F_BINS, TIME_STEPS) * 0.1 # Фоновый шум

    # Создадим "идеальный" паттерн (птичий чирк)
    pattern_width = 150
    y_coords = torch.arange(F_BINS).unsqueeze(1)
    x_coords = torch.arange(pattern_width).unsqueeze(0)
    center_freq = F_BINS * 0.7 # Паттерн будет в верхней части спектра
    chirp = torch.exp(-((y_coords - (center_freq + 20 * torch.sin(x_coords * 0.15)))**2) / 25)
    
    # Вставим паттерн 3 раза
    spec_tensor[:, 500:500+pattern_width] += chirp * 5.0
    spec_tensor[:, 1800:1800+pattern_width] += chirp * 3.0
    spec_tensor[:, 3000:3000+pattern_width] += chirp * 8.0

    # Упакуем в классы
    freq_axis = torch.arange(F_BINS)
    time_axis = torch.arange(TIME_STEPS) * 0.01 # dt = 10ms
    test_spec = SpecFunc((spec_tensor, UREG.dimensionless), freq_axis, time_axis)

    # Callback для визуализации
    def plot_callback(found_pattern: SpecFunc, correlation_timeline: TimeFunc):
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, figsize=(15, 7), gridspec_kw={'height_ratios': [2, 1]})
            
            pattern_img, _ = found_pattern.values
            im = axs[0].imshow(pattern_img.numpy(), aspect='auto', origin='lower', cmap='viridis')
            axs[0].set_title(f'Найденный паттерн (W) {pattern_img.shape}')
            fig.colorbar(im, ax=axs[0])
            
            timeline_data, _ = correlation_timeline.values
            axs[1].plot(correlation_timeline.time.numpy(), timeline_data.numpy())
            axs[1].set_title('1D Временная карта корреляций')
            axs[1].grid(True)
            axs[1].set_xlabel('Время, с')
            axs[1].set_ylabel('Суммарная корреляция')
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib не установлен, визуализация пропущена.")

    print("\n--- Запуск поиска самого характерного паттерна (с 1D сверткой по времени) ---")
    found_pattern, final_corr_map = discover_best_pattern_with_callback(
        spec=test_spec,
        window_duration_ms=1500.0, # 150 сэмплов при dt=10ms
        anchor_freq_percent=70.0,  # Ожидаем его вверху
        anchor_time_percent=50.0,  # Ожидаем его в центре окна
        epochs=150,
        learning_rate=0.05,
        top_k_peaks=15,
        lambda_tv=0.001,
        lambda_center=0.2, # Увеличим вес, чтобы якоря работали надежнее
        callback=plot_callback,
        callback_interval_percent=20.0
    )