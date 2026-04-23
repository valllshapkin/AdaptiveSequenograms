import torch

# Подставьте ваши актуальные пути импортов
from NewSpec.Core.Functions import TimeFunc, SpecFunc
from NewSpec.Core.Units import UREG

def noiseZNormByFreq(spec: SpecFunc, noise_percentile: float = 10.0) -> SpecFunc:
    """
    Адаптивная Z-нормировка по частотным полосам на основе профиля шума (версия для PyTorch).
    
    1. Интегрирует спектрограмму по частоте для получения энергии по времени.
    2. Находит временные кадры, энергия которых ниже заданного процентиля (считаем их шумом).
    3. Вычисляет среднее и СКО (сигму) для каждой частоты только на основе этих "шумовых" кадров.
    4. Применяет Z-нормировку ко всей спектрограмме вдоль оси времени для каждой частоты.
    
    Args:
        spec: Исходная спектрограмма.
        noise_percentile: Процентиль (от 0 до 100), определяющий границу "шумовых" кадров.
                          Например, 10.0 означает 10% самых тихих кадров.
    Returns:
        Спектрограмма после Z-нормировки (безразмерная величина).
    """
    
    # 1. Сворачиваем по оси частот для получения профиля энергии во времени
    time_func = spec.integrateOverFreq()
    energy_profile, _ = time_func.values

    # 2. Находим порог для самых тихих кадров
    # torch.quantile ожидает значение от 0 до 1
    q = torch.tensor(noise_percentile / 100.0, device=energy_profile.device, dtype=energy_profile.dtype)
    threshold = torch.quantile(energy_profile, q)
    
    # Создаем булеву маску для шумовых кадров
    noise_mask = energy_profile <= threshold

    # Защита от случая, если маска пуста (например, при percentile=0)
    if not torch.any(noise_mask):
        noise_mask[torch.argmin(energy_profile)] = True

    # Функция, которую мы применим к матрице спектрограммы
    def z_norm_func(mat: torch.Tensor) -> torch.Tensor:
        # 3. Выделяем кадры, относящиеся к шуму.
        # Форма матрицы '... freq time', маска по оси time.
        noise_mat = mat[..., :, noise_mask]

        # 4. Считаем среднее и СКО вдоль оси времени (dim=-1) для каждой частоты.
        # keepdim=True сохраняет размерность для broadcasting (форма будет '... freq 1')
        noise_mean = torch.mean(noise_mat, dim=-1, keepdim=True)
        noise_std = torch.std(noise_mat, dim=-1, keepdim=True)

        # Защита от деления на ноль
        safe_std = torch.where(noise_std < 1e-9, torch.tensor(1e-9, device=mat.device, dtype=mat.dtype), noise_std)

        # 5. Применяем Z-нормировку ко всей исходной матрице.
        # Broadcasting: (... freq time) - (... freq 1) / (... freq 1) -> OK
        return (mat - noise_mean) / safe_std

    # Возвращаем новый объект спектрограммы с безразмерной величиной
    return spec.cloneApply(func=z_norm_func, new_unit=UREG.dimensionless)


import torchaudio.transforms as T
from typing import Optional

def resampleTimeFunc(time_func: TimeFunc, new_sr: int) -> TimeFunc:
    """
    Ресемплирует временной сигнал (TimeFunc) в новую частоту дискретизации.
    """
    signal_a, signal_u = time_func.values
    
    # Восстанавливаем старую SR из шага по времени dt
    dt_val, _ = time_func.dt
    orig_sr = int(round(1.0 / dt_val))

    if orig_sr == new_sr:
        return time_func # Ресемплинг не требуется

    # Resampler ожидает форму (..., time)
    resampler = T.Resample(orig_sr, new_sr, dtype=signal_a.dtype).to(signal_a.device)
    resampled_a = resampler(signal_a)

    # Создаем новую временную ось
    num_samples = resampled_a.shape[-1]
    new_time_axis = torch.arange(num_samples, dtype=torch.float64, device=signal_a.device) / new_sr
    
    return TimeFunc(
        values=(resampled_a, signal_u),
        axis=new_time_axis
    )

def integrateAndClipEnvelope(
    spec: SpecFunc, 
    threshold: float = 5.0, 
    resample_sr: Optional[int] = None,
    make_zero: bool = True
) -> TimeFunc:
    """
    Интегрирует спектрограмму, нормализует её, выполняет опциональный ресемплинг 
    и отсекает шум по порогу сигм (версия для PyTorch).

    Args:
        spec: Входная спектрограмма.
        threshold: Коэффициент сигма для клиппинга (по умолчанию 5).
        resample_sr: Целевая частота дискретизации (SR). Если None - ресемплинг не проводится.
        make_zero: Если True, сдвигает отсеченные значения так, чтобы порог стал нулем.
    """
    # 1. Интегрируем по частотам
    time_func = spec.integrateOverFreq()
    
    # 2. Подготовка параметров нормализации
    freq_axis, freq_unit = spec.freq
    num_freq_bins = freq_axis.shape[-1]
    freq_range = freq_axis[-1] - freq_axis[0]
    if freq_range <= 0:
        freq_range = torch.tensor(1e-9, device=freq_axis.device, dtype=freq_axis.dtype)

    val_a, val_u = time_func.values
    
    # Масштабируем
    norm_factor = torch.sqrt(torch.tensor(num_freq_bins, device=val_a.device, dtype=val_a.dtype)) * freq_range
    scaled_values = val_a / norm_factor
    new_unit = unit_devide(val_u, freq_unit)

    # Создаем временный объект TimeFunc
    envelope = TimeFunc(
        values=(scaled_values, new_unit),
        axis=time_func.axis[0]
    )

    # 3. Ресемплинг (выполняется ДО расчета сигмы и клиппинга)
    if resample_sr is not None:
        envelope = resampleTimeFunc(envelope, resample_sr)

    # 4. Клиппинг на основе статистики уже (возможно) ресемплированного сигнала
    final_values, final_unit = envelope.values
    
    sigma = torch.std(final_values)
    limit = threshold * sigma
    
    # Применяем клиппинг: всё что ниже 'limit' становится равным 'limit'
    clipped_values = torch.clamp(final_values, min=limit)
    
    if make_zero:
        clipped_values = clipped_values - limit
    
    # Возвращаем финальный TimeFunc
    return TimeFunc(
        values=(clipped_values, final_unit),
        axis=envelope.axis[0]
    )