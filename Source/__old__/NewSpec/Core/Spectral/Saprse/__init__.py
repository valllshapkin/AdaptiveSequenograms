from typing import Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import NMF
from tqdm.auto import tqdm

from NewSpec.Core.Functions import SpecFunc, TimeFunc
from NewSpec.Core.Units import UREG

class _FFTSpecConvDecomposer(nn.Module):
    def __init__(self, num_patterns: int, num_freqs: int, win_width: int, seq_len: int, dtype: torch.dtype):
        super().__init__()
        self.num_patterns = num_patterns
        self.W = seq_len
        self.W_win = win_width
        self.basis = nn.Parameter(torch.abs(torch.randn(num_patterns, num_freqs, win_width, dtype=dtype)) * 0.1)
        self.seq = nn.Parameter(torch.abs(torch.randn(1, num_patterns, seq_len, dtype=dtype)) * 0.01)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        active_seq = F.relu(self.seq)
        fft_size = self.W + self.W_win - 1
        seq_f = torch.fft.rfft(active_seq, n=fft_size)
        basis_f = torch.fft.rfft(self.basis, n=fft_size)
        canvas_f = torch.einsum('b n f, n h f -> b h f', seq_f, basis_f)
        canvas = torch.fft.irfft(canvas_f, n=fft_size)
        return canvas[..., :self.W], active_seq


def extract_spec_patterns_with_callback(
    spec: SpecFunc,
    num_patterns: int = 6,
    num_learn: int = 12,
    window_duration_ms: float = 100.0,
    epochs: int = 2000,
    learning_rate: float = 0.05,
    lambda_l1: float = 0.15,
    lambda_ortho: float = 1.0,
    lambda_center: float = 0.1,  # НОВЫЙ ПАРАМЕТР: вес лосса центрирования
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[SpecFunc, TimeFunc, SpecFunc, SpecFunc], None]] = None,
    callback_interval_percent: Optional[float] = 1.0
) -> Tuple[SpecFunc, TimeFunc, SpecFunc, SpecFunc]:
    
    # --- 1. Подготовка данных ---
    S_tensor, S_unit = spec.values
    freq_axis, _ = spec.freq
    time_axis, _ = spec.time
    
    if device is None:
        device = S_tensor.device
        
    dtype = S_tensor.dtype
    target_canvas = S_tensor.unsqueeze(0) if S_tensor.dim() == 2 else S_tensor
    target_canvas = target_canvas.to(device)
    
    dt_val, _ = spec.dt
    sr = 1.0 / dt_val
    num_freqs, seq_len = target_canvas.shape[1], target_canvas.shape[2]
    win_width = max(3, int(round((window_duration_ms / 1000.0) * sr)))
    
    model = _FFTSpecConvDecomposer(num_learn, num_freqs, win_width, seq_len, dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    identity = torch.eye(num_learn, dtype=dtype, device=device)

    # Предрасчет для лосса центрирования
    target_center = (win_width - 1) / 2.0
    time_indices = torch.arange(win_width, dtype=dtype, device=device)

    callback_epoch_interval = 0
    if callback and callback_interval_percent is not None and callback_interval_percent > 0 and epochs > 0:
        callback_epoch_interval = max(1, int(epochs * (callback_interval_percent / 100.0)))

    print(f"Запуск разложения: {num_patterns} паттернов (через {num_learn}). Окно: {win_width} сэмплов.")
        
    def _post_process_and_invoke_callback(
        current_model: _FFTSpecConvDecomposer, 
        reconstructed_canvas: torch.Tensor, 
        activations: torch.Tensor
    ):
        current_model.eval()
        with torch.no_grad():
            seq_np = activations[0].cpu().numpy()
            
            # --- Сжатие через NMF ---
            nmf = NMF(n_components=num_patterns, init='nndsvda', max_iter=500)
            nmf_projections = nmf.fit_transform(seq_np.T)
            nmf_timelines = torch.tensor(nmf_projections.T, dtype=dtype, device='cpu')

            nmf_weights = nmf.components_ 
            flat_bases_np = current_model.basis.view(num_learn, -1).cpu().numpy()
            nmf_bases_flat = np.dot(nmf_weights, flat_bases_np)
            
            nmf_bases = torch.tensor(
                nmf_bases_flat.reshape(num_patterns, num_freqs, win_width),
                dtype=dtype, device='cpu'
            )

            # --- Покомпонентная реконструкция (БЕЗ ручного сдвига, модель отцентровала всё сама) ---
            fft_size = seq_len + win_width - 1
            time_f = torch.fft.rfft(nmf_timelines, n=fft_size)  # (num_patterns, freq_bins)
            basis_f = torch.fft.rfft(nmf_bases, n=fft_size)     # (num_patterns, num_freqs, freq_bins)
            
            # Свертка (Умножение в частотной области: Таймлайн * Базис)
            canvas_components_f = torch.einsum('n f, n h f -> n h f', time_f, basis_f)
            
            # Возвращаемся во временную область, обрезаем хвост свертки
            components_canvas = torch.fft.irfft(canvas_components_f, n=fft_size)[..., :seq_len]
            
            # Общая реконструкция — сумма всех компонентов
            total_reconstructed = components_canvas.sum(dim=0) 
            
            # --- Упаковка в классы ---
            cpu_freq_axis, cpu_time_axis = freq_axis.cpu(), time_axis.cpu()
            basis_time_axis = torch.arange(win_width, dtype=dtype, device='cpu') * dt_val
            
            timelines_func = TimeFunc((nmf_timelines, UREG.dimensionless), cpu_time_axis)
            bases_spec = SpecFunc((nmf_bases, S_unit), cpu_freq_axis, basis_time_axis)
            reconstructed_spec = SpecFunc((total_reconstructed, S_unit), cpu_freq_axis, cpu_time_axis)
            
            # Слой компонентов
            components_spec = SpecFunc((components_canvas, S_unit), cpu_freq_axis, cpu_time_axis)
            
            if callback:
                callback(bases_spec, timelines_func, reconstructed_spec, components_spec)

        current_model.train()
        return bases_spec, timelines_func, reconstructed_spec, components_spec
            
    # --- 3. Цикл обучения ---
    pbar = tqdm(range(epochs), desc="Оптимизация базисов")
    latest_results = None

    for epoch in pbar:
        optimizer.zero_grad()
        
        with torch.no_grad():
            model.basis.data.clamp_(min=0.0) # Запрет на отрицательные значения
            norms = torch.norm(model.basis, p=2, dim=(1, 2), keepdim=True)
            model.basis.data.div_(norms + 1e-8)

        pred_canvas, active_seq = model.forward()
        
        # 1. Лосс реконструкции
        loss_recon = F.mse_loss(pred_canvas, target_canvas)
        
        # 2. Лосс спарсовости (редкости) таймлайнов
        seq_l2_norms = torch.sqrt(torch.sum(active_seq**2, dim=1) + 1e-8)
        loss_sparse = lambda_l1 * torch.mean(seq_l2_norms)
        
        # 3. Лосс ортогональности базисов (чтобы они не были похожи друг на друга)
        flat_basis = model.basis.view(num_learn, -1)
        gram_matrix = torch.mm(flat_basis, flat_basis.T)
        loss_ortho = lambda_ortho * F.mse_loss(gram_matrix, identity)

        # 4. НОВЫЙ ЛОСС: Центр масс (CoM) каждого базиса должен быть в центре окна
        # Считаем плотность энергии по времени: (num_learn, win_width)
        density = model.basis.sum(dim=1) 
        mass = density.sum(dim=1) + 1e-8
        # Считаем центр масс: sum(time_idx * density) / mass
        com = (density * time_indices).sum(dim=1) / mass
        # Штрафуем отклонение от target_center
        loss_center = lambda_center * F.mse_loss(com, torch.full_like(com, target_center))
        
        loss = loss_recon + loss_sparse + loss_ortho + loss_center
        loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{loss.item():.3f}", Rec=f"{loss_recon.item():.3f}", Cntr=f"{loss_center.item():.3f}")

        is_last_epoch = (epoch == epochs - 1)
        if callback_epoch_interval > 0 and ((epoch + 1) % callback_epoch_interval == 0 or is_last_epoch):
            latest_results = _post_process_and_invoke_callback(model, pred_canvas, active_seq)

    if latest_results is None:
        latest_results = _post_process_and_invoke_callback(model, *model.forward())

    return latest_results


# =====================================================================
# НОВЫЙ БЛОК: Поиск самого характерного паттерна через ZNCC и градиентный спуск
# =====================================================================

def _functional_match2d_fft(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет 2D кросс-корреляцию через FFT. Ядро НЕ должно быть отзеркалено заранее.
    Функциональная версия для дифференцируемых вычислений.
    """
    flipped_kernel = torch.flip(kernel, dims=(-2, -1))
    
    F_sig, T_sig = signal.shape[-2], signal.shape[-1]
    F_ker, T_ker = flipped_kernel.shape[-2], flipped_kernel.shape[-1]
    
    N_f = F_sig + F_ker - 1
    N_t = T_sig + T_ker - 1

    Sig_f = torch.fft.rfft2(signal, s=(N_f, N_t), dim=(-2, -1))
    Ker_f = torch.fft.rfft2(flipped_kernel, s=(N_f, N_t), dim=(-2, -1))
    
    out_f = Sig_f * Ker_f
    out = torch.fft.irfft2(out_f, s=(N_f, N_t), dim=(-2, -1))

    pad_f_start = (F_ker - 1) // 2
    pad_t_start = (T_ker - 1) // 2
    
    return out[..., pad_f_start : pad_f_start + F_sig, pad_t_start : pad_t_start + T_sig]

def _total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Заставляет паттерн быть гладким, штрафуя резкие перепады."""
    tv_h = torch.sum(torch.abs(img[..., :-1, :] - img[..., 1:, :]))
    tv_w = torch.sum(torch.abs(img[..., :, :-1] - img[..., :, 1:]))
    return tv_h + tv_w

def discover_best_pattern_with_callback(
    spec: SpecFunc,
    window_duration_ms: float = 100.0,
    anchor_percent: float = 50.0,  ### ИЗМЕНЕНИЕ: Добавляем якорь положения ###
    epochs: int = 500,
    learning_rate: float = 0.05,
    top_k_peaks: int = 20,
    lambda_tv: float = 0.01,
    lambda_center: float = 0.1,  ### ИЗМЕНЕНИЕ: Вес для нового лосса ###
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[SpecFunc, SpecFunc], None]] = None,
    callback_interval_percent: Optional[float] = 2.0
) -> Tuple[SpecFunc, SpecFunc]:
    """
    Находит один "идеальный" паттерн...
    
    Args:
        ...
        anchor_percent (float): Ожидаемое положение центра масс паттерна по времени
                                (в процентах от ширины окна). 50.0 - центр.
        ...
        lambda_center (float): Вес штрафа за отклонение от якоря.
        ...
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
    
    ### ИЗМЕНЕНИЕ: Предрасчет для лосса центрирования ###
    target_center_index = (win_width - 1) * (anchor_percent / 100.0)
    time_indices = torch.arange(win_width, device=device, dtype=dtype)
    
    print(f"Поиск паттерна: окно {win_width} сэмплов. Целевой центр: {target_center_index:.1f}")
    
    callback_epoch_interval = 0
    if callback and callback_interval_percent is not None and callback_interval_percent > 0:
        callback_epoch_interval = max(1, int(epochs * (callback_interval_percent / 100.0)))

    def _invoke_callback(current_W_raw: torch.Tensor, zncc_map_tensor: torch.Tensor):
        with torch.no_grad():
            W_norm = current_W_raw - current_W_raw.mean()
            W_norm /= (torch.norm(W_norm) + 1e-8)
            
            # Упаковка в SpecFunc
            pattern_time_axis = torch.arange(win_width, device='cpu', dtype=dtype) * dt_val
            
            found_pattern_spec = SpecFunc(
                (W_norm.squeeze(0).cpu(), S_unit), 
                freq_axis.cpu(), 
                pattern_time_axis
            )
            
            correlation_map_spec = SpecFunc(
                (zncc_map_tensor.squeeze(0).cpu(), UREG.dimensionless),
                freq_axis.cpu(),
                time_axis.cpu()
            )
            
            if callback:
                callback(found_pattern_spec, correlation_map_spec)
        return found_pattern_spec, correlation_map_spec

    # --- 3. Цикл обучения ---

    pbar = tqdm(range(epochs), desc="Поиск паттерна")
    latest_results = None
    eps = 1e-8

    for epoch in pbar:
        optimizer.zero_grad()
        
        # --- ШАГ A: Нормализация паттерна (без изменений) ---
        W_zero_mean = W_raw - W_raw.mean()
        W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
        
        # --- ШАГ B: ZNCC (без изменений) ---
        # ... (код для расчета zncc_map) ...
        numerator = _functional_match2d_fft(target_spec, W)
        U = torch.ones_like(W); N = W.numel()
        Sum_S = _functional_match2d_fft(target_spec, U)
        Sum_S2 = _functional_match2d_fft(target_spec**2, U)
        local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
        denominator = torch.sqrt(local_variance) + eps
        zncc_map = numerator / denominator

        # --- ШАГ C: Вычисление Loss-функции ---
        # 1. Лосс "заметности" (без изменений)
        top_peaks, _ = torch.topk(zncc_map.flatten(), k=top_k_peaks)
        loss_match = -torch.mean(top_peaks) 
        
        # 2. Лосс гладкости (без изменений)
        loss_tv = lambda_tv * _total_variation_loss(W_raw)
        
        ### ИЗМЕНЕНИЕ: Добавляем лосс центрирования ###
        # 3. Лосс центра масс (Center of Mass Loss)
        # Мы хотим, чтобы энергия в W_raw была сконцентрирована вокруг target_center_index.
        # Используем квадрат W_raw, чтобы работать с "энергией" и сделать ее положительной.
        energy_density = W_raw**2
        
        # Плотность энергии по времени: (1, win_width)
        time_density = energy_density.sum(dim=-2)
        total_mass = time_density.sum() + eps
        
        # Считаем центр масс: sum(time_idx * density) / mass
        current_com = (time_density * time_indices).sum() / total_mass
        
        # Штрафуем отклонение от целевого центра
        loss_center = lambda_center * F.mse_loss(
            current_com, 
            torch.full_like(current_com, target_center_index)
        )
        
        total_loss = loss_match + loss_tv + loss_center
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(Loss=f"{total_loss.item():.3f}", Match=f"{-loss_match.item():.3f}", Cntr=f"{loss_center.item():.3f}")


        # Вызов callback для визуализации
        is_last_epoch = (epoch == epochs - 1)
        if callback_epoch_interval > 0 and ((epoch + 1) % callback_epoch_interval == 0 or is_last_epoch):
            latest_results = _invoke_callback(W_raw, zncc_map)

    # Гарантированный вызов в конце, если не было в цикле
    if latest_results is None:
        # Пересчитаем карту с финальными весами
        with torch.no_grad():
            W_zero_mean = W_raw - W_raw.mean()
            W = W_zero_mean / (torch.norm(W_zero_mean) + eps)
            numerator = _functional_match2d_fft(target_spec, W)
            U = torch.ones_like(W); N = W.numel()
            Sum_S = _functional_match2d_fft(target_spec, U)
            Sum_S2 = _functional_match2d_fft(target_spec**2, U)
            local_variance = (Sum_S2 - (Sum_S**2) / N).clamp_(min=0.0)
            denominator = torch.sqrt(local_variance) + eps
            final_zncc_map = numerator / denominator
            latest_results = _invoke_callback(W_raw, final_zncc_map)

    return latest_results

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    # --- Сначала протестируем старый код, чтобы убедиться, что ничего не сломалось ---
    try:
        F_BINS, TIME_STEPS, TEMPLATES_COUNT, TEMPLATES_TIME = 64, 1000, 4, 21
        ANYBATCH = (10, )
        H = nn.Parameter(torch.rand(*ANYBATCH, TEMPLATES_COUNT, TIME_STEPS))
        templates_data = torch.rand(*ANYBATCH, TEMPLATES_COUNT, F_BINS, TEMPLATES_TIME)
        # и т.д. ... (ваш старый __main__ блок)
        print("Старый код __main__ работает.")
    except Exception as e:
        print(f"Ошибка в старом коде: {e}")
        
    # --- Теперь новый тест для discover_best_pattern ---
    F_BINS, TIME_STEPS = 64, 2000
    spec_tensor = torch.randn(F_BINS, TIME_STEPS) * 0.1 # Фоновый шум

    # Создадим "идеальный" паттерн (например, птичий чирк - синусоида)
    pattern_width = 100
    y_coords = torch.arange(F_BINS).unsqueeze(1)
    x_coords = torch.arange(pattern_width).unsqueeze(0)
    center_freq = F_BINS / 2
    chirp = torch.exp(-((y_coords - (center_freq + 15 * torch.sin(x_coords * 0.2)))**2) / 10)
    
    # Вставим этот паттерн в три места на спектрограмме
    spec_tensor[:, 200:200+pattern_width] += chirp * 5.0
    spec_tensor[:, 900:900+pattern_width] += chirp * 3.0
    spec_tensor[:, 1500:1500+pattern_width] += chirp * 8.0

    # Упакуем в ваши классы
    freq_axis = torch.arange(F_BINS)
    time_axis = torch.arange(TIME_STEPS) * 0.01 # dt = 10ms
    
    test_spec = SpecFunc((spec_tensor, UREG.dimensionless), freq_axis, time_axis)

    # Определим callback для визуализации
    def plot_callback(found_pattern, correlation_map):
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            
            pattern_img, _ = found_pattern.values
            axs[0].imshow(pattern_img.numpy(), aspect='auto', origin='lower', cmap='viridis')
            axs[0].set_title('Найденный паттерн (W)')
            
            corr_img, _ = correlation_map.values
            axs[1].imshow(corr_img.numpy(), aspect='auto', origin='lower', cmap='coolwarm', vmin=-1, vmax=1)
            axs[1].set_title('Карта корреляций ZNCC')
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib не установлен, визуализация пропущена.")

    print("\n--- Запуск поиска самого характерного паттерна ---")
    found_pattern, final_corr_map = discover_best_pattern_with_callback(
        spec=test_spec,
        window_duration_ms=1000.0, # 100 сэмплов при dt=10ms
        epochs=100,
        learning_rate=0.05,
        top_k_peaks=10,
        lambda_tv=0.001,
        callback=plot_callback,
        callback_interval_percent=25.0
    )

    print("\nПоиск завершен!")
    print("Форма найденного паттерна:", found_pattern.values[0].shape)
    print("Форма итоговой карты корреляций:", final_corr_map.values[0].shape)