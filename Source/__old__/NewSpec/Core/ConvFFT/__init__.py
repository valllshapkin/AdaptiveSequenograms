from matplotlib import pyplot as plt
import torch
import torch.fft
from typing import Union

from NewSpec.Core.Functions import SpecFunc, TimeFunc
from NewSpec.Core.Spectral.Some import matchPearsonFT
from NewSpec.Core.Units import UREG, unit_mul

def _get_zero_index(time_axis: torch.Tensor) -> int:
    """Находит индекс якоря: точку на оси времени, максимально близкую к 0."""
    return int(torch.argmin(torch.abs(time_axis)))

def conv_T_T(sig: TimeFunc, ker: TimeFunc) -> TimeFunc:
    """1D Сигнал * 1D Окно -> 1D Сигнал"""
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    # FFT
    Sig_f = torch.fft.rfft(sig_v, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v, n=N, dim=-1)
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    # Якорь берем из ядра
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    # Физическое интегрирование (умножаем на dt ядра)
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return TimeFunc(values=(out, new_unit), axis=sig.time[0])

def conv_S_T(sig: SpecFunc, ker: TimeFunc) -> SpecFunc:
    """2D Спектрограмма * 1D Окно -> 2D Спектрограмма (Фильтрация спектрограммы во времени)"""
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    # Добавляем фиктивную ось частот для ядра: (..., T) -> (..., 1, T)
    ker_v_expanded = ker_v.unsqueeze(-2)
    
    Sig_f = torch.fft.rfft(sig_v, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v_expanded, n=N, dim=-1)
    
    # Broadcasting: (..., F, N_freq) * (..., 1, N_freq)
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=sig.freq[0], time=sig.time[0])

def conv_T_S(sig: TimeFunc, ker: SpecFunc) -> SpecFunc:
    """1D Сигнал * 2D Окно -> 2D Спектрограмма (Генерация: Активации * Шаблоны)"""
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    # Добавляем фиктивную ось частот для сигнала: (..., T) -> (..., 1, T)
    sig_v_expanded = sig_v.unsqueeze(-2)
    
    Sig_f = torch.fft.rfft(sig_v_expanded, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v, n=N, dim=-1)
    
    # Broadcasting: (..., 1, N_freq) * (..., F, N_freq)
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    # Ось времени берем от сигнала (так как он задает длину), а частоты - от ядра (шаблона)
    return SpecFunc(matrix=(out, new_unit), freq=ker.freq[0], time=sig.time[0])

def conv_S_S(sig: SpecFunc, ker: SpecFunc) -> SpecFunc:
    """
    2D Спектрограмма * 2D Окно -> 2D Спектрограмма
    Внимание: это 1D-свертка по оси времени для каждой частоты независимо!
    (Модель реверберации: поэлементное умножение по частоте, свертка по времени).
    """
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    # Проверка, что частотные оси совпадают по размеру
    assert sig_v.shape[-2] == ker_v.shape[-2], "Размерности частот сигнала и ядра должны совпадать"
    
    T_sig = sig_v.shape[-1]
    T_ker = ker_v.shape[-1]
    N = T_sig + T_ker - 1
    
    Sig_f = torch.fft.rfft(sig_v, n=N, dim=-1)
    Ker_f = torch.fft.rfft(ker_v, n=N, dim=-1)
    
    # Элементное умножение по F и N_freq
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    Z = _get_zero_index(ker.time[0])
    out = out[..., Z : Z + T_sig]
    
    dt_val, dt_unit = ker.dt
    out = out * dt_val
    new_unit = unit_mul(sig_u, ker_u, dt_unit)
    
    return SpecFunc(matrix=(out, new_unit), freq=sig.freq[0], time=sig.time[0])

# =====================================================================
# Удобный диспетчер (Overloading)
# =====================================================================
def convolveTime(a: Union[TimeFunc, SpecFunc], b: Union[TimeFunc, SpecFunc]) -> Union[TimeFunc, SpecFunc]:
    """
    Универсальная функция свёртки. a - сигнал, b - ядро(фильтр).
    Определяет тип операции автоматически.
    """
    if isinstance(a, TimeFunc) and isinstance(b, TimeFunc):
        return conv_T_T(a, b)
    elif isinstance(a, SpecFunc) and isinstance(b, TimeFunc):
        return conv_S_T(a, b)
    elif isinstance(a, TimeFunc) and isinstance(b, SpecFunc):
        return conv_T_S(a, b)
    elif isinstance(a, SpecFunc) and isinstance(b, SpecFunc):
        return conv_S_S(a, b)
    else:
        raise TypeError(f"Неподдерживаемые типы для свёртки: {type(a)} и {type(b)}")
    
def convolveFT(sig: SpecFunc, ker: SpecFunc) -> SpecFunc:
    """
    ИСТИННАЯ 2D СВЁРТКА: 2D Спектрограмма * 2D Окно -> 2D Спектрограмма.
    Свертка происходит одновременно по осям Времени и Частоты (через 2D FFT).
    """
    sig_v, sig_u = sig.values
    ker_v, ker_u = ker.values
    
    # Размеры по частоте (F) и времени (T)
    F_sig, T_sig = sig_v.shape[-2], sig_v.shape[-1]
    F_ker, T_ker = ker_v.shape[-2], ker_v.shape[-1]
    
    # Размеры для 2D FFT (по теореме о свертке)
    N_f = F_sig + F_ker - 1
    N_t = T_sig + T_ker - 1
    
    # Применяем 2D вещественное БПФ
    # s=(N_f, N_t) указывает размерности, dim=(-2, -1) указывает оси
    Sig_fft = torch.fft.rfft2(sig_v, s=(N_f, N_t), dim=(-2, -1))
    Ker_fft = torch.fft.rfft2(ker_v, s=(N_f, N_t), dim=(-2, -1))
    
    # Умножаем в частотной (Фурье) области и возвращаем в пространственно-временную
    out = torch.fft.irfft2(Sig_fft * Ker_fft, s=(N_f, N_t), dim=(-2, -1))
    
    # Находим физические якоря (индексы нулей) в ядре по ОБЕИМ осям
    Z_f = _get_zero_index(ker.freq[0])
    Z_t = _get_zero_index(ker.time[0])
    
    # Отрезаем рамки: размер выходного сигнала равен размеру входного сигнала
    out = out[..., Z_f : Z_f + F_sig, Z_t : Z_t + T_sig]
    
    # Физическое интегрирование: двойной интеграл df dt
    df_val, df_unit = ker.df
    dt_val, dt_unit = ker.dt
    
    out = out * (df_val * dt_val)
    new_unit = unit_mul(sig_u, ker_u, df_unit, dt_unit)
    
    # Возвращаем новую спектрограмму, сохраняя оси исходного сигнала
    return SpecFunc(matrix=(out, new_unit), freq=sig.freq[0], time=sig.time[0])


if __name__ == "__main__":
    # --- ОБЩИЕ ПАРАМЕТРЫ ---
    TIME_STEPS = 1000
    FREQ_BINS = 64
    
    t_axis = torch.linspace(-5.0, 5.0, TIME_STEPS)
    f_axis = torch.linspace(0.0, 100.0, FREQ_BINS)
    
    unit = UREG.dimensionless
    extent_S = [t_axis[0].item(), t_axis[-1].item(), f_axis[0].item(), f_axis[-1].item()]

    # =================================================================
    # 1. ТЕСТ: T * T (Аналитическая свертка Лоренциан)
    # =================================================================
    print("\n--- 1. ТЕСТ (T * T): Свертка Лоренциан ---")
    
    y_vals = (1.0 / (t_axis**2 + 1.0)).unsqueeze(0)
    sig_T = TimeFunc(values=(y_vals, unit), axis=t_axis)
    ker_T = TimeFunc(values=(y_vals, unit), axis=t_axis)
    
    res_T_T = convolveTime(sig_T, ker_T)
    
    y_num = res_T_T.values[0][0]
    y_ana = (2 * torch.pi) / (t_axis**2 + 4.0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_axis.numpy(), y_ana.numpy(), 'k--', lw=3, label="Аналитика $2\pi / (t^2+4)$")
    plt.plot(t_axis.numpy(), y_num.numpy(), 'r-', alpha=0.7, lw=2, label="Численно T * T")
    plt.plot(t_axis.numpy(), y_vals[0].numpy(), 'b-', alpha=0.3, label="Исходный сигнал")
    plt.title("1. T * T (1D Свертка)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()


    # =================================================================
    # 2. ТЕСТ: S * T (Фильтрация спектрограммы во времени)
    # =================================================================
    print("--- 2. ТЕСТ (S * T): Сглаживание шумной спектрограммы ---")
    
    # Сигнал S: Диагональная линия (чирп) + сильный шум
    F_grid, T_grid = torch.meshgrid(f_axis, t_axis, indexing='ij')
    clean_chirp = torch.exp(-((F_grid - (T_grid + 5)*10)**2) / 20.0)
    noise = torch.rand_like(clean_chirp) * 0.5
    S_noisy_vals = (clean_chirp + noise).unsqueeze(0)
    
    # Ядро T: Гауссово окно сглаживания (ширина ~0.2 сек)
    T_gauss_vals = torch.exp(-(t_axis**2) / (2 * 0.2**2)).unsqueeze(0)
    
    sig_S = SpecFunc(matrix=(S_noisy_vals, unit), freq=f_axis, time=t_axis)
    ker_T_gauss = TimeFunc(values=(T_gauss_vals, unit), axis=t_axis)
    
    res_S_T = convolveTime(sig_S, ker_T_gauss)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(S_noisy_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='viridis')
    axs[0].set_title("Сигнал S (Шумный чирп)")
    axs[1].plot(t_axis.numpy(), T_gauss_vals[0].numpy(), 'g')
    axs[1].set_title("Ядро T (Гауссиана)")
    axs[1].set_xlim(-1, 1); axs[1].grid(True)
    axs[2].imshow(res_S_T.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='viridis')
    axs[2].set_title("Результат S * T (Сглажено)")
    plt.tight_layout()


    # =================================================================
    # 3. ТЕСТ: T * S (Генеративная модель)
    # =================================================================
    print("--- 3. ТЕСТ (T * S): Размещение шаблонов по спайкам ---")
    
    # Сигнал T: Три спайка в разное время (-3, 0, 2)
    T_spikes_vals = torch.zeros_like(t_axis)
    T_spikes_vals[torch.argmin(torch.abs(t_axis - (-3.0)))] = 1.0
    T_spikes_vals[torch.argmin(torch.abs(t_axis - 0.0))] = 0.5   # Тихий спайк
    T_spikes_vals[torch.argmin(torch.abs(t_axis - 2.0))] = 1.2   # Громкий спайк
    T_spikes_vals = T_spikes_vals.unsqueeze(0)
    
    # Шаблон S: Капля/Радио-импульс на частоте 60 Гц
    S_blob_vals = (torch.exp(-((F_grid - 60)**2) / 100.0) * torch.exp(-(T_grid**2) / 0.1)).unsqueeze(0)
    
    sig_T_spikes = TimeFunc(values=(T_spikes_vals, unit), axis=t_axis)
    ker_S_blob = SpecFunc(matrix=(S_blob_vals, unit), freq=f_axis, time=t_axis)
    
    res_T_S = convolveTime(sig_T_spikes, ker_S_blob)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(t_axis.numpy(), T_spikes_vals[0].numpy(), 'r')
    axs[0].set_title("Активации T (Спайки)")
    axs[0].grid(True)
    axs[1].imshow(S_blob_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='magma')
    axs[1].set_title("Ядро S (Шаблон)")
    axs[2].imshow(res_T_S.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='magma')
    axs[2].set_title("Результат T * S")
    plt.tight_layout()


    # =================================================================
    # 4. ТЕСТ: S * S (Реверберация среды)
    # =================================================================
    print("--- 4. ТЕСТ (S * S): Физичное эхо (частотно-зависимое) ---")
    
    # Сигнал S: Тот же чирп, но чистый
    S_clean_vals = clean_chirp.unsqueeze(0)
    
    # Эхо S: Экспоненциальное затухание. 
    # Физика: Высокие частоты (F) затухают быстрее (t * F/20). Эхо только для t >= 0.
    echo_mask = (T_grid >= 0).float()
    decay_rates = 1.0 + (F_grid / 20.0) 
    S_echo_vals = (torch.exp(-T_grid * decay_rates) * echo_mask).unsqueeze(0)
    
    sig_S_clean = SpecFunc(matrix=(S_clean_vals, unit), freq=f_axis, time=t_axis)
    ker_S_echo = SpecFunc(matrix=(S_echo_vals, unit), freq=f_axis, time=t_axis)
    
    res_S_S = convolveTime(sig_S_clean, ker_S_echo)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(S_clean_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='plasma')
    axs[0].set_title("Сигнал S (Оригинал)")
    axs[1].imshow(S_echo_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='plasma')
    axs[1].set_title("Профиль эха S (ВЧ затухают быстрее)")
    axs[1].set_xlim(-1, 3) # Приблизим окно эха
    axs[2].imshow(res_S_S.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='plasma')
    axs[2].set_title("Результат S * S (Шлейф эха)")
    plt.tight_layout()

    # =================================================================
    # 5. ТЕСТ: convolveFT (Истинная 2D свёртка S * S)
    # =================================================================
    print("\n--- 5. ТЕСТ (convolveFT): 2D Размытие спектрограммы ---")
    
    # 1. Сигнал S: Три точки-импульса в координатах (F, T)
    S_points = torch.zeros(1, FREQ_BINS, TIME_STEPS)
    # Ставим точки в случайных местах
    S_points[0, 15, 200] = 1.0  # Низкая частота, начало
    S_points[0, 45, 500] = 0.8  # Высокая частота, середина
    S_points[0, 30, 800] = 1.2  # Средняя частота, конец
    
    sig_S_2d = SpecFunc(matrix=(S_points, unit), freq=f_axis, time=t_axis)
    
    # 2. Ядро S: 2D Гауссово пятно
    # Создаем ОТДЕЛЬНЫЕ маленькие оси для ядра, центрированные вокруг 0!
    f_ker_axis = torch.linspace(-30.0, 30.0, 31) # от -30 до 30 Гц
    t_ker_axis = torch.linspace(-1.0, 1.0, 51)   # от -1 до 1 сек
    
    F_ker_grid, T_ker_grid = torch.meshgrid(f_ker_axis, t_ker_axis, indexing='ij')
    # Формула 2D Гауссианы
    S_gauss_2d = torch.exp(- (F_ker_grid**2 / 100.0) - (T_ker_grid**2 / 0.05)).unsqueeze(0)
    
    ker_S_2d = SpecFunc(matrix=(S_gauss_2d, unit), freq=f_ker_axis, time=t_ker_axis)
    
    # 3. Выполняем 2D свёртку
    res_FT = convolveFT(sig_S_2d, ker_S_2d)
    
    # 4. Отрисовка
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    axs[0].imshow(S_points[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='inferno')
    axs[0].set_title("Сигнал S (Одиночные точки)")
    
    # Ядро показываем в его собственных осях
    ker_extent = [t_ker_axis[0].item(), t_ker_axis[-1].item(), f_ker_axis[0].item(), f_ker_axis[-1].item()]
    axs[1].imshow(S_gauss_2d[0].numpy(), aspect='auto', origin='lower', extent=ker_extent, cmap='inferno')
    axs[1].set_title("Ядро S (2D Гауссиана центрированная в 0, 0)")
    axs[1].axhline(0, color='w', alpha=0.5, linestyle='--')
    axs[1].axvline(0, color='w', alpha=0.5, linestyle='--')
    
    axs[2].imshow(res_FT.values[0][0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='inferno')
    axs[2].set_title("Результат convolveFT (2D Свертка)")
    
    plt.tight_layout()

    
    # =================================================================
    # 6. ТЕСТ: matchPearsonFT (Поиск шаблона NCC)
    # =================================================================
    print("\n--- 6. ТЕСТ (matchPearsonFT): Поиск шаблона Пирсона в шуме ---")
    
    # Создаем базовые оси для паттерна (центрированные)
    f_patt_axis = torch.linspace(-20.0, 20.0, 21)
    t_patt_axis = torch.linspace(-0.5, 0.5, 51)
    F_patt_grid, T_patt_grid = torch.meshgrid(f_patt_axis, t_patt_axis, indexing='ij')
    
    # Шаблон: Сложная фигура (Крест)
    pattern_vals = torch.exp(-(F_patt_grid**2)/5.0) * torch.exp(-(T_patt_grid**2)/0.5) + \
                   torch.exp(-(F_patt_grid**2)/200.0) * torch.exp(-(T_patt_grid**2)/0.005)
    pattern_vals = pattern_vals.unsqueeze(0)
    
    # Исходный сигнал: чистый белый ШУМ [0, 1]
    S_search_vals = torch.rand(1, FREQ_BINS, TIME_STEPS) / 1000
    
    # Вживляем паттерн в сигнал в случайное место (индексы 40 по частоте, 700 по времени)
    f_insert, t_insert = 40, 700
    F_p, T_p = pattern_vals.shape[-2], pattern_vals.shape[-1]
    
    # Добавляем паттерн (накладываем поверх шума)
    S_search_vals[0, f_insert : f_insert + F_p, t_insert : t_insert + T_p] += pattern_vals[0]
    
    # Создаем классы
    sig_search = SpecFunc(matrix=(S_search_vals, unit), freq=f_axis, time=t_axis)
    ker_pattern = SpecFunc(matrix=(pattern_vals, unit), freq=f_patt_axis, time=t_patt_axis)
    
    # ВЫПОЛНЯЕМ ПОИСК
    res_pearson = matchPearsonFT(sig_search, ker_pattern)
    map_vals = res_pearson.values[0][0].numpy()
    
    # Отрисовка
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    axs[0].imshow(S_search_vals[0].numpy(), aspect='auto', origin='lower', extent=extent_S, cmap='gray')
    axs[0].set_title("Сигнал S (Шум + Вживленный крест)")
    
    patt_extent = [t_patt_axis[0].item(), t_patt_axis[-1].item(), f_patt_axis[0].item(), f_patt_axis[-1].item()]
    axs[1].imshow(pattern_vals[0].numpy(), aspect='auto', origin='lower', extent=patt_extent, cmap='magma')
    axs[1].set_title("Искомый шаблон (Ядро)")
    
    im = axs[2].imshow(map_vals, aspect='auto', origin='lower', extent=extent_S, cmap='RdBu_r', vmin=-1, vmax=1)
    axs[2].set_title("Карта Пирсона R [-1, 1]")
    fig.colorbar(im, ax=axs[2])
    
    plt.tight_layout()
    plt.show()
    
    # Проверяем максимум Пирсона
    max_R = map_vals.max()
    print(f"Максимальный коэффициент Пирсона: {max_R:.4f}")
    if max_R > 0.9:
        print("Патерн успешно найден!")