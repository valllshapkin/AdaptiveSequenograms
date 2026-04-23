from typing import Tuple, Union
import warnings

import torch
from torch import Tensor
import torch.nn.functional as F

# Подставьте ваши актуальные пути импортов
from NewSpec.Core.Functions import TimeFunc, SpecFunc
from NewSpec.Core.ConvWindow import Window, WindowNorm, WindowNormMismatchError
from NewSpec.Core.Units import UREG, unit_devide, unit_sqrt, unit_mul


def makeComplexSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    signal_a, signal_unit = signal.values
    device = signal_a.device
    sr = signal.sr

    win        = window.get_array(sr, device=device)
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop        = max(1, int(win_length * (1 - overlap)))

    if fft_length < win_length:
        warnings.warn(
            f"fft_length={fft_length} меньше win_length={win_length}. "
            f"Фрейм будет обрезан — это потеря данных."
        )
    if fft_length > win_length:
        warnings.warn(
            f"fft_length={fft_length} больше win_length={win_length}. "
            f"Это увеличивает разрешение по частоте, но не приносит реальные данные."
        )
    print(f"fft_length={fft_length} win_length={win_length} hop={hop}")

    # STFT с center=True автоматически добавляет паддинг для краев
    stft = torch.stft(
        signal_a,
        n_fft=fft_length,
        hop_length=hop,
        win_length=win_length,
        window=win,
        center=True,             # ВАЖНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ
        return_complex=True
    )

    scale = torch.sqrt(torch.tensor(2.0 / sr, dtype=torch.float64, device=device))
    num_frames = stft.shape[-1]
    
    freq_axis = torch.fft.rfftfreq(fft_length, d=1.0 / sr).to(device=device)
    time_axis = (torch.arange(num_frames, dtype=torch.float64, device=device) * hop) / sr

    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    return SpecFunc(
        matrix=(stft * scale, spec_unit),
        freq=freq_axis,
        time=time_axis,
    )


def inverseComplexSpec(spec: SpecFunc, window: Window, overlap: float = 0.5) -> TimeFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)
    
    complex_matrix, spec_unit = spec.values
    time_axis, _ = spec.time
    freq_axis, _ = spec.freq
    device = complex_matrix.device

    # 1. Восстанавливаем параметры FFT
    fft_length = (len(freq_axis) - 1) * 2
    
    # 2. Восстанавливаем SR (Частота дискретизации)
    sr_f_val, _ = spec.df 
    sr = int(round(sr_f_val * fft_length))
    
    # 3. Готовим окно и шаг
    win        = window.get_array(sr, device=device)
    win_length = len(win)
    hop        = max(1, int(win_length * (1 - overlap)))

    scale = torch.sqrt(torch.tensor(2.0 / sr, dtype=torch.float64, device=device))
    unscaled = complex_matrix / scale

    # 4. Обратный STFT с center=True для правильного восстановления
    signal_a = torch.istft(
        unscaled,
        n_fft=fft_length,
        hop_length=hop,
        win_length=win_length,
        window=win,
        center=True              # ВАЖНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ
    )
    
    new_time_axis = torch.arange(signal_a.shape[-1], dtype=torch.float64, device=device) / sr
    signal_unit = unit_mul(spec_unit, unit_sqrt(UREG.Hz))

    return TimeFunc(
        values=(signal_a, signal_unit),
        axis=new_time_axis,
    )


def makeSpec(signal: TimeFunc, window: Window, overlap: float = 0.5, bins: int = 300) -> SpecFunc:
    if window.norm != WindowNorm.ENERGY:
        raise WindowNormMismatchError(WindowNorm.ENERGY, window.norm)

    signal_a, signal_unit = signal.values
    device = signal_a.device
    sr = signal.sr

    win        = window.get_array(sr, device=device)
    fft_length = (bins - 1) * 2
    win_length = len(win)
    hop        = max(1, int(win_length * (1 - overlap)))

    stft = torch.stft(
        signal_a,
        n_fft=fft_length,
        hop_length=hop,
        win_length=win_length,
        window=win,
        center=True,             # ВАЖНОЕ ИСПРАВЛЕНИЕ ЗДЕСЬ
        return_complex=True
    )

    amplitude_matrix = torch.abs(stft)
    
    scale = torch.sqrt(torch.tensor(2.0 / sr, dtype=torch.float64, device=device))
    num_frames = amplitude_matrix.shape[-1]
    
    freq_axis = torch.fft.rfftfreq(fft_length, d=1.0 / sr).to(device=device)
    time_axis = (torch.arange(num_frames, dtype=torch.float64, device=device) * hop) / sr

    spec_unit = unit_devide(signal_unit, unit_sqrt(UREG.Hz))

    return SpecFunc(
        matrix=(amplitude_matrix * scale, spec_unit),
        freq=freq_axis,
        time=time_axis,
    )


REF_PA_ASD = 2e-5  
REF_FS_ASD = 1.0   
REF_V_ASD  = 1.0   

def makeLogDB(spec: SpecFunc, add_one: bool = False) -> SpecFunc:
    _, old_unit = spec.values

    if old_unit.is_compatible_with(UREG.Pa / (UREG.Hz ** 0.5)):
        ref_val = REF_PA_ASD
    elif old_unit.is_compatible_with(UREG.FS / (UREG.Hz ** 0.5)):
        ref_val = REF_FS_ASD
    elif old_unit.is_compatible_with(UREG.V / (UREG.Hz ** 0.5)):
        ref_val = REF_V_ASD
    else:
        ref_val = 1.0

    def to_db(arr: Tensor) -> Tensor:
        amplitude = torch.abs(arr)
        ratio = amplitude / ref_val
        
        if add_one:
            return 20 * torch.log10(ratio + 1)
        else:
            scaled_ratio = torch.clamp(ratio, min=1e-9)
            return 20 * torch.log10(scaled_ratio)

    return spec.cloneApply(func=to_db, new_unit=UREG.dB)

def interpolate(self: SpecFunc, new_freq: torch.Tensor, new_time: torch.Tensor) -> 'SpecFunc':
    """
    Интерполирует спектрограмму (билинейно) для новых осей частоты и времени.
    Если новые оси выходят за пределы старых, значения заполняются нулями.
    Если новые оси меньше старых, спектрограмма обрезается.
    """
    mat_a, mat_u = self.values
    old_freq, _ = self.freq
    old_time, _ = self.time
    
    device = mat_a.device
    new_freq = new_freq.to(device)
    new_time = new_time.to(device)

    # Функция для нормализации физических координат в диапазон [-1, 1],
    # который требуется для torch.nn.functional.grid_sample
    def normalize_coords(new_ax, old_ax):
        min_val, max_val = old_ax[0], old_ax[-1]
        if min_val == max_val:
            return torch.zeros_like(new_ax)
        # Линейное отображение: min_val -> -1, max_val -> 1
        return 2.0 * (new_ax - min_val) / (max_val - min_val) - 1.0

    # Получаем сетку нормализованных координат
    grid_x = normalize_coords(new_time, old_time)  # Ось X (ширина) - время
    grid_y = normalize_coords(new_freq, old_freq)  # Ось Y (высота) - частота

    # Создаем 2D сетку координат. indexing='ij' означает (freq, time) -> (Y, X)
    mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    
    # grid_sample ожидает сетку формы [N, H_out, W_out, 2] с парами (x, y)
    grid = torch.stack([mesh_x, mesh_y], dim=-1).unsqueeze(0) 
    
    # Обработка произвольных батчей (...)
    # grid_sample работает с 4D тензорами [N, C, H, W]
    original_shape = mat_a.shape
    batch_shape = original_shape[:-2]
    
    if len(batch_shape) > 0:
        flat_N = int(torch.prod(torch.tensor(batch_shape)))
    else:
        flat_N = 1
        
    # Решейп в [Батч, Каналы, Высота(freq), Ширина(time)]
    mat_reshaped = mat_a.reshape(flat_N, 1, old_freq.shape[0], old_time.shape[0])
    
    # Разворачиваем сетку на весь размер батча
    grid_dtype = mat_reshaped.real.dtype if mat_reshaped.is_complex() else mat_reshaped.dtype
    grid = grid.expand(flat_N, -1, -1, -1).to(grid_dtype)

    # Выполняем интерполяцию. 
    # align_corners=True важно: края пикселей (-1 и 1) соответствуют точным значениям f_min и f_max
    if mat_reshaped.is_complex():
        # Комплексные тензоры интерполируем по частям
        real_part = F.grid_sample(mat_reshaped.real, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        imag_part = F.grid_sample(mat_reshaped.imag, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        out_reshaped = torch.complex(real_part, imag_part)
    else:
        out_reshaped = F.grid_sample(mat_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # Возвращаем матрицу к оригинальной размерности с новыми размерами осей
    new_shape = (*batch_shape, new_freq.shape[0], new_time.shape[0])
    out_mat = out_reshaped.reshape(new_shape)

    return SpecFunc(
        matrix=(out_mat, mat_u),
        freq=new_freq,
        time=new_time
    )

def interpolateToShape(spec: 'SpecFunc', target_shape: Tuple[int, int]) -> 'SpecFunc':
    """
    Интерполирует спектрограмму до заданного числа частот и времени.

    Параметры:
        spec: исходный объект SpecFunc
        target_shape: (new_nfreq, new_ntime) – желаемое количество отсчётов
                      по частоте и времени соответственно.

    Возвращает:
        новый объект SpecFunc с интерполированной матрицей и новыми осями.
    """
    # Получаем старые оси
    old_freq, _ = spec.freq   # предполагается, что spec.freq возвращает (freq_vals, label?)
    old_time, _ = spec.time

    # Строим новые оси как равномерную сетку между min и max старых осей
    new_freq = torch.linspace(old_freq[0], old_freq[-1], target_shape[0], device=old_freq.device)
    new_time = torch.linspace(old_time[0], old_time[-1], target_shape[1], device=old_time.device)

    # Вызываем исходный метод интерполяции
    return interpolate(spec, new_freq, new_time)

def interpolateByFactors(spec: 'SpecFunc', factors: Tuple[float, float]) -> 'SpecFunc':
    """
    Масштабирует спектрограмму, изменяя количество отсчётов по частоте и времени
    в соответствии с заданными коэффициентами.

    Параметры:
        spec: исходный объект SpecFunc
        factors: (freq_factor, time_factor) — во сколько раз изменить размер.
                 Например, (2, 2) увеличит обе оси вдвое,
                 (0.5, 0.5) — уменьшит вдвое.

    Возвращает:
        новый объект SpecFunc с интерполированной матрицей и новыми осями.
    """
    old_freq, _ = spec.freq
    old_time, _ = spec.time

    # Новое количество отсчётов (округляем до целого)
    new_nfreq = max(1, int(round(old_freq.shape[0] * factors[0])))
    new_ntime = max(1, int(round(old_time.shape[0] * factors[1])))

    # Строим равномерные оси в тех же физических пределах
    new_freq = torch.linspace(old_freq[0], old_freq[-1], new_nfreq, device=old_freq.device)
    new_time = torch.linspace(old_time[0], old_time[-1], new_ntime, device=old_time.device)

    # Вызываем исходный метод интерполяции
    return interpolate(spec, new_freq, new_time)


def blurSpec(
    spec: 'SpecFunc',
    sigma: Union[float, Tuple[float, float]] = 1.0
) -> 'SpecFunc':
    """
    Применяет Гауссов блюр к спектрограмме, используя быструю свёртку через FFT.
    Этот метод значительно быстрее, чем прямая свёртка, особенно для больших матриц.
    Сохраняет исходные размеры матрицы и физические единицы.

    :param spec: Исходная спектрограмма (SpecFunc).
    :param sigma: Стандартное отклонение Гауссианы. Одно число или tuple (freq_sig, time_sig).
                  Размер ядра определяется автоматически на основе sigma.
    """
    
    # 1. Приводим sigma к формату кортежа (freq, time)
    if isinstance(sigma, (float, int)):
        sigma = (float(sigma), float(sigma))

    # Размер ядра выбираем как 3 стандартных отклонения в каждую сторону,
    # округляем до ближайшего нечетного числа. Это стандартная практика.
    kernel_size_f = int(round(sigma[0] * 3)) * 2 + 1
    kernel_size_t = int(round(sigma[1] * 3)) * 2 + 1
    kernel_size = (kernel_size_f, kernel_size_t)

    # Извлекаем текущие единицы измерения, так как блюр их не меняет
    _, current_unit = spec.values

    def apply_gaussian_blur_fft(mat: torch.Tensor) -> torch.Tensor:
        device = mat.device
        dtype = mat.real.dtype if mat.is_complex() else mat.dtype
        img_h, img_w = mat.shape[-2], mat.shape[-1]
        
        # 2. Создаем маленькое ядро Гаусса
        def get_1d_kernel(k: int, s: float) -> torch.Tensor:
            limit = (k - 1) / 2.0
            x = torch.linspace(-limit, limit, steps=k, device=device, dtype=dtype)
            gauss = torch.exp(-0.5 * (x / s).pow(2))
            return gauss / gauss.sum()

        kernel_y = get_1d_kernel(kernel_size[0], sigma[0])
        kernel_x = get_1d_kernel(kernel_size[1], sigma[1])
        kernel_2d_small = (kernel_y.unsqueeze(-1) * kernel_x.unsqueeze(0))
        
        # 3. Создаем ядро размером с изображение и помещаем маленькое ядро в угол.
        # Это необходимо для FFT-свёртки.
        padded_kernel = torch.zeros(img_h, img_w, device=device, dtype=dtype)
        k_h, k_w = kernel_2d_small.shape
        padded_kernel[:k_h, :k_w] = kernel_2d_small
        
        # 4. Сдвигаем ядро так, чтобы его центр (пик Гауссианы) оказался в точке (0, 0).
        # Это устраняет фазовый сдвиг, который возник бы после IFFT.
        padded_kernel = torch.roll(padded_kernel, shifts=(-k_h // 2, -k_w // 2), dims=(-2, -1))
        
        # 5. Выполняем свёртку в частотной области
        # FFT функции применяются к последним двум осям, батчи обрабатываются автоматически.
        fft_mat = torch.fft.fft2(mat, dim=(-2, -1))
        fft_kernel = torch.fft.fft2(padded_kernel, dim=(-2, -1))
        
        fft_result = fft_mat * fft_kernel
        
        # 6. Обратное преобразование Фурье
        ifft_result = torch.fft.ifft2(fft_result, dim=(-2, -1))

        # 7. Возвращаем результат нужного типа
        # Для реальных входных данных мнимая часть после IFFT будет очень мала (ошибки округления).
        # Для комплексных - результат и должен быть комплексным.
        if not mat.is_complex():
            return ifft_result.real
        else:
            return ifft_result

    # Используем ваш встроенный метод cloneApply для применения операции
    return spec.cloneApply(func=apply_gaussian_blur_fft, new_unit=current_unit)