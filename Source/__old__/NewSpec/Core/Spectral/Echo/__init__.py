from typing import Tuple, Optional
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from NewSpec.Core.Functions import SpecFunc
from NewSpec.Core.Units import UREG


class AnchoredConv2D_2DKernel(nn.Module):
    def __init__(self, data: torch.Tensor, zero_index: int):
        super().__init__()
        assert data.dim() >= 2, "Окно должно иметь минимум 2 оси (F, W)"
        assert 0 <= zero_index < data.size(-1), "zero_index выходит за границы"
    
        if not isinstance(data, nn.Parameter):
            self.data = nn.Parameter(data)
        else:
            self.data = data
            
        self.zero_index = zero_index

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        assert signal.dim() >= 2, "Сигнал должен иметь минимум (F, T)"
        
        F_sig, T = signal.shape[-2], signal.shape[-1]
        F_ker, W = self.data.shape[-2], self.data.shape[-1]
        assert F_sig == F_ker, f"Оси F не совпадают: сигнал {F_sig}, ядро {F_ker}"
        
        N = T + W - 1
        
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        
        Z = self.zero_index
        return out[..., Z : Z + T]


def deconvolveEcho(
    spec: SpecFunc,
    echo_duration_ms: float = 100.0,
    num_iterations: int = 200,
    learning_rate: float = 1e-3,
    lambda_sparsity: float = 1e-4,
    lambda_smooth_echo: float = 1e-2,
    gaussian_sigma_ratio: float = 0.3,
    enforce_energy_conservation: bool = True,
    device: Optional[torch.device] = None
) -> Tuple[SpecFunc, SpecFunc]:
    """
    Выполняет слепую деконволюцию для Z-нормированной спектрограммы.
    Использует гауссову инициализацию ядра эха.

    Модель: S_observed ≈ V_clean * E_echo (свёртка по времени)

    Args:
        spec: Исходная Z-нормированная спектрограмма.
        echo_duration_ms: Длительность окна эха в миллисекундах.
        num_iterations: Количество шагов оптимизации.
        learning_rate: Скорость обучения.
        lambda_sparsity: Коэффициент L1-штрафа для чистого сигнала.
        lambda_smooth_echo: Коэффициент гладкости эха (по времени).
        gaussian_sigma_ratio: Относительная ширина гаусса (sigma = duration * ratio).
        enforce_energy_conservation: Нормировать ли эхо на каждой итерации.
        device: Устройство для вычислений (cpu/cuda).

    Returns:
        Кортеж из двух SpecFunc: (очищенная_спектрограмма, профиль_эха).
    """
    
    # 1. Извлечение данных и подготовка параметров
    S_tensor, S_unit = spec.values
    freq_axis, _ = spec.freq
    time_axis, time_unit = spec.time
    
    if device is None:
        device = S_tensor.device
        
    S_tensor = S_tensor.to(device)
    freq_axis = freq_axis.to(device)
    time_axis = time_axis.to(device)
    
    dt_val, _ = spec.dt
    sr = 1.0 / dt_val
    
    num_freqs, num_times = S_tensor.shape[-2:]

    # Размер окна эха в сэмплах (округляем до нечетного для симметрии)
    echo_width = int(round((echo_duration_ms / 1000.0) * sr))
    if echo_width % 2 == 0:
        echo_width += 1
    if echo_width < 3:
        echo_width = 3
    
    anchor_idx = echo_width // 2

    # 2. Инициализация обучаемых параметров
    
    # Исходная гипотеза: чистый сигнал = наблюдаемый сигнал
    V_tensor = nn.Parameter(S_tensor.clone().detach())
    
    # Гауссово ядро для инициализации эха
    E_tensor_init = torch.zeros(num_freqs, echo_width, device=device, dtype=S_tensor.dtype)
    
    # Временная ось ядра (относительно центра)
    t_axis = torch.arange(echo_width, device=device) - anchor_idx
    t_seconds = t_axis * dt_val
    
    # 1D гаусс по времени
    sigma_time = (echo_duration_ms / 1000.0) * gaussian_sigma_ratio
    gaussian_1d = torch.exp(-0.5 * (t_seconds / sigma_time) ** 2)
    
    # Нормируем и применяем ко всем частотам
    gaussian_norm = gaussian_1d / (gaussian_1d.sum() + 1e-8)
    for f in range(num_freqs):
        E_tensor_init[f, :] = gaussian_norm
    
    echo_conv_layer = AnchoredConv2D_2DKernel(data=E_tensor_init, zero_index=anchor_idx).to(device)
    E_tensor_param = echo_conv_layer.data

    # 3. Настройка оптимизатора и функции потерь
    optimizer = torch.optim.Adam([V_tensor, E_tensor_param], lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    def smoothness_loss_1d(x: torch.Tensor) -> torch.Tensor:
        """L2 гладкость по временной оси"""
        diff = x[:, 1:] - x[:, :-1]
        return torch.mean(diff ** 2)
    
    print(f"Запуск деконволюции (гауссово ядро, sigma={sigma_time*1000:.1f}ms)")
    print(f"  Размер эха: {echo_width} сэмплов, итераций: {num_iterations}")
    
    # 4. Цикл оптимизации
    progress_bar = tqdm(range(num_iterations), desc="Деконволюция эха")
    for i in progress_bar:
        optimizer.zero_grad()

        # Прямой проход: S_hat = V * E
        S_hat = echo_conv_layer(V_tensor)

        # Расчет потерь
        loss_reconstruction = mse_loss(S_hat, S_tensor)
        loss_sparse = lambda_sparsity * torch.mean(torch.abs(V_tensor))
        loss_smooth_echo = lambda_smooth_echo * smoothness_loss_1d(E_tensor_param)
        loss_negative = torch.mean(torch.relu(-E_tensor_param)) * 0.1
        
        total_loss = loss_reconstruction + loss_sparse + loss_smooth_echo + loss_negative

        # Обратное распространение
        total_loss.backward()
        
        # Optional: gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_([V_tensor, E_tensor_param], max_norm=1.0)
        
        optimizer.step()
        
        # Принудительная нормировка эха (сохранение энергии)
        if enforce_energy_conservation:
            with torch.no_grad():
                for f in range(num_freqs):
                    sum_f = E_tensor_param[f, :].sum()
                    if sum_f > 0:
                        E_tensor_param[f, :] /= (sum_f + 1e-8)
        
        # Обновляем информацию в прогресс-баре
        if (i + 1) % 10 == 0 or i == num_iterations - 1:
            progress_bar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                rec=f"{loss_reconstruction.item():.4f}",
                sp=f"{loss_sparse.item():.4f}"
            )

    print(f"Финальная ошибка реконструкции: {loss_reconstruction.item():.6f}")

    # 5. Формирование результирующих объектов SpecFunc
    V_clean = V_tensor.detach().cpu()
    E_echo = E_tensor_param.detach().cpu()

    # Создаем новую временную ось для эха, центрированную в 0
    echo_time_axis = (torch.arange(echo_width) - anchor_idx) * dt_val

    clean_spec = SpecFunc(
        matrix=(V_clean, S_unit),
        freq=freq_axis.cpu(),
        time=time_axis.cpu()
    )

    echo_profile = SpecFunc(
        matrix=(E_echo, UREG.dimensionless),
        freq=freq_axis.cpu(),
        time=echo_time_axis
    )

    return clean_spec, echo_profile