from pathlib import Path
from typing import Optional
import torch
from torch import Tensor
import numpy as np

from NewSpec.Core.Units import UREG, PintUnit
from NewSpec.Core.Functions import TimeFunc

def loadRecord(path: Path, unit: PintUnit = UREG.FS, device: torch.device | str | None = None) -> 'TimeFunc':
    """Загружает аудиофайл как TimeFunc (mono, float64)."""
    import soundfile as sf  # type: ignore

    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    # Читаем файл
    data, samplerate = sf.read(str(path), dtype='float64', always_2d=True)

    # Приводим к mono (смешиваем каналы или берем один)
    if data.ndim == 2:
        if data.shape[1] > 1:
            data = np.mean(data, axis=1)        # stereo -> mono
        else:
            data = data.flatten()               # (N, 1) -> (N,)

    # Создаем тензоры сразу на нужном устройстве
    data_t = torch.tensor(data, dtype=torch.float64, device=device)
    n_samples = len(data_t)
    time_t = torch.arange(n_samples, dtype=torch.float64, device=device) / samplerate

    return TimeFunc(
        values=(data_t, unit),
        axis=time_t
    )


def trimRecord(signal: 'TimeFunc', 
               t_start: float = 0.0, 
               t_end: Optional[float] = None, 
               device: torch.device | str | None = None) -> 'TimeFunc':
    """
    Обрезает запись по времени (в секундах). Поддерживает батчи.
    """
    if t_start < 0:
        raise ValueError("t_start не может быть отрицательным")

    time_array, time_unit = signal.time
    value_array, value_unit = signal.values

    # Если t_end не указан — берём конец сигнала
    if t_end is None:
        t_end = float(time_array[-1])

    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) должен быть больше t_start ({t_start})")

    # Находим индексы для обрезки (на том же устройстве, где и time_array)
    mask = (time_array >= t_start) & (time_array <= t_end)
    
    if not torch.any(mask):
        raise ValueError(f"Интервал [{t_start}, {t_end}] не пересекается с сигналом")

    # Обрезаем массивы (используем [..., mask], чтобы не сломать батчи перед осью времени)
    new_time = time_array[mask]
    new_values = value_array[..., mask]

    # Если запрошен перенос на другое устройство
    if device is not None:
        new_time = new_time.to(device)
        new_values = new_values.to(device)

    # Создаём новый объект TimeFunc
    return TimeFunc(
        values=(new_values, value_unit),
        axis=new_time
    )


def resampleRecord(signal: 'TimeFunc', 
                   new_sr: int, 
                   device: torch.device | str | None = None) -> 'TimeFunc':
    """
    Изменяет частоту дискретизации сигнала (resampling).
    """
    from scipy.signal import resample_poly  # type: ignore
    from math import gcd

    if new_sr <= 0:
        raise ValueError("new_sr должен быть положительным целым числом")

    time_array, time_unit = signal.time
    value_array, value_unit = signal.values
    orig_sr = signal.sr
    
    # Определяем целевое устройство: запрошенное, либо текущее устройство значений
    target_device = device if device is not None else value_array.device

    if orig_sr == new_sr:
        # Нет необходимости в ресэмплинге
        return TimeFunc(
            values=(value_array.clone().to(target_device), value_unit), 
            axis=time_array.clone().to(target_device)
        )

    g = gcd(orig_sr, new_sr)
    up = new_sr // g
    down = orig_sr // g

    # scipy работает только с numpy, поэтому временно переносим тензор в CPU
    values_np = value_array.detach().cpu().numpy()

    # ВАЖНО: В новой архитектуре ось времени всегда последняя (axis=-1).
    # Это позволяет корректно применять scipy.resample_poly к батчам (B, T)
    new_values_np = resample_poly(values_np, up=up, down=down, axis=-1)

    # Возвращаем данные обратно в PyTorch на нужное устройство
    new_values = torch.tensor(new_values_np, dtype=value_array.dtype, device=target_device)

    # Создаём новый временной массив
    new_length = new_values.shape[-1]
    new_dt = 1.0 / new_sr
    new_time = torch.arange(new_length, dtype=time_array.dtype, device=target_device) * new_dt

    return TimeFunc(
        values=(new_values, value_unit),
        axis=new_time
    )

def correctDC(f: TimeFunc) -> TimeFunc:
    v, V = f.values
    t, _ = f.time

    corrected_values = v - torch.median(v, dim=-1).values
    
    return TimeFunc(
        values=(corrected_values, V),
        axis=t.clone()
    )
