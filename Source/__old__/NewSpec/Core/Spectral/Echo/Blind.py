from typing import Tuple, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from NewSpec.Core.Functions import SpecFunc
from NewSpec.Core.Units import UREG


class _FreqDomainDeconvFilter(nn.Module):
    """
    Обучаемый деконволюционный фильтр G в частотной области.
    Модель: V_clean = S_observed * G.
    Инициализируется как обратный к предполагаемому профилю эха (1/H_guess),
    что эквивалентно фильтру Винера при нулевом шуме.
    """
    def __init__(self, num_freqs: int, seq_len: int, tau_samples: float, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.seq_len = seq_len
        self.n_bins = seq_len // 2 + 1
        
        with torch.no_grad():
            # 1. Создаем временной профиль ПРЕДПОЛАГАЕМОГО эха H_guess
            t_indices = torch.arange(seq_len, dtype=dtype, device=device)
            decay = torch.exp(-t_indices / max(tau_samples, 1.0))
            decay = decay / decay.sum() # Нормируем (сумма = 1), чтобы H_guess[0] в частотной области был ≈ 1
            
            # 2. Переводим его в Фурье-пространство
            h_guess_time = decay.unsqueeze(0).repeat(num_freqs, 1)
            h_guess_freq = torch.fft.rfft(h_guess_time, n=seq_len, dim=-1)
            
            # 3. Инициализируем наш фильтр G как обратный к H_guess
            # Добавляем эпсилон для стабильности деления
            g_init_freq = 1.0 / (h_guess_freq + 1e-8)
        
        # Обучаемые параметры - это реальная и мнимая части G
        self.real_p = nn.Parameter(g_init_freq.real.clone())
        self.imag_p = nn.Parameter(g_init_freq.imag.clone())

    def get_complex_G(self) -> torch.Tensor:
        """Собирает комплексный тензор деконволюционного фильтра G."""
        real = self.real_p.clone()
        imag = self.imag_p.clone()
        
        # Жесткое ограничение: DC компонента G должна быть 1, т.к. H[0]=1.
        # Это сохраняет общую энергию/громкость сигнала.
        real[..., 0] = 1.0
        imag[..., 0] = 0.0

        # Nyquist компонента также должна быть вещественной для RFFT
        if self.seq_len % 2 == 0:
            imag[..., -1] = 0.0

        return torch.complex(real, imag)
        
    def get_implicit_echo_time(self) -> torch.Tensor:
        """
        Возвращает физический профиль эха H, соответствующий текущему фильтру G.
        Вычисляется как H = 1/G. Используется для наложения физических ограничений (регуляризации).
        """
        G_freq = self.get_complex_G()
        # H_freq = 1 / G_freq
        H_freq = 1.0 / (G_freq + 1e-8)
        return torch.fft.irfft(H_freq, n=self.seq_len, dim=-1)


def deconvolve_echo_wiener_style(
    spec: SpecFunc,
    echo_duration_limit_ms: float = 150.0,
    epochs: int = 200,
    learning_rate: float = 0.005,
    lambda_sparsity: float = 0.01,
    lambda_echo_decay: float = 1.0,
    lambda_echo_positivity: float = 0.5,
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[SpecFunc, SpecFunc], None]] = None,
    callback_interval_percent: Optional[float] = 10.0
) -> Tuple[SpecFunc, SpecFunc]:
    """
    Выполняет слепую деконволюцию, обучая обратный фильтр в стиле Винера.
    Цель (loss) - максимизация разреженности (sparsity) восстановленного сигнала.
    
    Args:
        spec: Исходная спектрограмма с эхом.
        echo_duration_limit_ms: Предполагаемая длительность эха для инициализации и регуляризации.
        epochs: Количество эпох оптимизации.
        learning_rate: Скорость обучения.
        lambda_sparsity: Вес L1-штрафа для чистого сигнала (основная движущая сила).
        lambda_echo_decay: Вес штрафа за слишком длинный "хвост" у подразумеваемого эха H.
        lambda_echo_positivity: Вес штрафа за отрицательные значения у H.
        device: Устройство для вычислений (cpu/cuda).
        callback: Функция для визуализации промежуточных результатов.
        callback_interval_percent: Как часто вызывать callback (в процентах от общего числа эпох).

    Returns:
        Кортеж из двух SpecFunc: (очищенная_спектрограмма, профиль_эха).
    """
    
    # --- 1. Подготовка данных ---
    S_tensor, S_unit = spec.values
    freq_axis, _ = spec.freq
    time_axis, _ = spec.time
    
    if device is None: device = S_tensor.device
    dtype = S_tensor.dtype
    
    # S_observed в частотной области - это константа, которую мы будем фильтровать
    S_observed_freq = torch.fft.rfft(S_tensor.to(device), dim=-1)
    
    dt_val, _ = spec.dt
    sr = 1.0 / dt_val
    num_freqs, seq_len = S_tensor.shape[-2], S_tensor.shape[-1]
    
    # --- 2. Инициализация модели ---
    limit_samples = (echo_duration_limit_ms / 1000.0) * sr
    init_tau_samples = limit_samples / 3.0 # tau, при котором к лимиту сигнал упадет в e^3 раз
    
    deconv_model = _FreqDomainDeconvFilter(num_freqs, seq_len, init_tau_samples, dtype, device)
    
    optimizer = torch.optim.Adam(deconv_model.parameters(), lr=learning_rate)

    # --- Маска для регуляризации эха H ---
    decay_start_idx = int(round(limit_samples))
    time_indices = torch.arange(seq_len, device=device, dtype=dtype)
    decay_mask = torch.clamp(time_indices - decay_start_idx, min=0.0) / seq_len
    decay_mask = decay_mask.unsqueeze(0) 

    callback_epoch_interval = 0
    if callback and callback_interval_percent is not None and callback_interval_percent > 0 and epochs > 0:
        callback_epoch_interval = max(1, int(epochs * (callback_interval_percent / 100.0)))
        
    print(f"Запуск деконволюции (стиль Винера). Tau: {init_tau_samples:.1f} сэмплов.")
    
    def _invoke_callback(current_model: _FreqDomainDeconvFilter):
        with torch.no_grad():
            G_freq = current_model.get_complex_G()
            V_clean_freq = S_observed_freq * G_freq
            V_clean_time = torch.fft.irfft(V_clean_freq, n=seq_len, dim=-1).cpu()

            H_time = current_model.get_implicit_echo_time().cpu()
            vis_width = min(seq_len, max(decay_start_idx * 3, 50))
            H_vis = H_time[..., :vis_width]
            
            clean_spec = SpecFunc((V_clean_time, S_unit), freq_axis.cpu(), time_axis.cpu())
            
            echo_time_axis = torch.arange(vis_width, dtype=dtype, device='cpu') * dt_val
            echo_spec = SpecFunc((H_vis, UREG.dimensionless), freq_axis.cpu(), echo_time_axis)
            
            if callback:
                callback(clean_spec, echo_spec)
                
        return clean_spec, echo_spec

    # --- 3. Цикл обучения ---
    pbar = tqdm(range(epochs), desc="Оптимизация фильтра G")
    latest_results = None

    for epoch in pbar:
        optimizer.zero_grad()
        
        # --- Прямой проход: фильтрация для получения V_clean ---
        G_freq = deconv_model.get_complex_G()
        V_clean_freq = S_observed_freq * G_freq
        V_clean_time = torch.fft.irfft(V_clean_freq, n=seq_len, dim=-1)
        
        # --- Основной Loss: разреженность (sparsity) результата ---
        # Мы хотим, чтобы очищенный сигнал был "резким", без хвостов.
        loss_sparse = lambda_sparsity * torch.mean(torch.abs(V_clean_time))
        
        # --- Регуляризация: физичность подразумеваемого эха H = 1/G ---
        H_time = deconv_model.get_implicit_echo_time()
        
        # Штраф за длинный хвост у H
        loss_decay = lambda_echo_decay * torch.mean((H_time * decay_mask)**2)
        
        # Штраф за отрицательные значения у H (эхо не может инвертировать фазу в таком виде)
        loss_positivity = lambda_echo_positivity * torch.mean(F.relu(-H_time)**2)
        
        total_loss = loss_sparse + loss_decay + loss_positivity
        
        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(
            Loss=f"{total_loss.item():.4f}", 
            Spar=f"{loss_sparse.item():.4f}", 
            Decay=f"{loss_decay.item():.5f}"
        )
        
        # Вызов callback для визуализации
        is_last_epoch = (epoch == epochs - 1)
        if callback_epoch_interval > 0 and ((epoch + 1) % callback_epoch_interval == 0 or is_last_epoch):
            latest_results = _invoke_callback(deconv_model)

    # Гарантированный финальный вызов, если цикл не вызвал его на последней итерации
    if latest_results is None:
        latest_results = _invoke_callback(deconv_model)

    return latest_results