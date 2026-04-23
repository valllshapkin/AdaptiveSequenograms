from typing import List, Tuple

import torch
import torch.nn as nn

from NewSpec.Core.Units import UREG
from NewSpec.Core.Functions import SpecFunc, TimeFunc

def _get_zero_index(time_axis: torch.Tensor) -> int:
    """Находит индекс якоря: точку на оси времени, максимально близкую к 0."""
    return int(torch.argmin(torch.abs(time_axis)))

def matchPearsonTime(sig: SpecFunc, ker: SpecFunc) -> TimeFunc:
    """
    Поиск шаблона только по времени (1D профиль из 2D Кросс-Корреляции).
    Сводит задачу к 2D поиску (matchPearsonFT) с последующим извлечением 
    среза по оси времени при нулевом сдвиге по частоте.
    """
    # 1. Проверка идентичности осей частот
    sig_f_val, sig_f_u = sig.freq
    ker_f_val, ker_f_u = ker.freq
    
    if sig_f_u != ker_f_u:
        raise ValueError(f"Единицы измерения частот не совпадают: {sig_f_u} != {ker_f_u}")
    
    if sig_f_val.shape != ker_f_val.shape or not torch.allclose(sig_f_val, ker_f_val):
        raise ValueError("Оси частот сигнала и шаблона должны быть идентичны.")
        
    # 2. Вызываем базовый 2D поиск Пирсона (сигнал и ядро)
    corr_2d = matchPearsonFT(sig, ker)
    
    # 3. Находим индекс якоря по частоте (точка центровки)
    # При равных размерах частотной оси, этот индекс в выходном тензоре 
    # соответствует полному перекрытию частот (сдвиг = 0)
    Z_f = _get_zero_index(ker.freq[0])
    
    # 4. Вырезаем нужный срез по частоте
    corr_v, corr_u = corr_2d.values
    # corr_v имеет форму [..., freq, time]. Берем срез по предпоследнему измерению.
    out_time_v = corr_v[..., Z_f, :]
    
    # 5. Упаковываем в TimeFunc
    # corr_2d.time[0] — это временная ось (тензор)
    # corr_u — безразмерная величина (UREG.dimensionless) из matchPearsonFT
    return TimeFunc(
        values=(out_time_v, corr_u), 
        axis=corr_2d.time[0]
    )


def matchPearsonFT(sig: SpecFunc, ker: SpecFunc) -> SpecFunc:
    """
    Поиск шаблона (2D Кросс-Корреляция / Коэффициент Пирсона).
    Сигнал 2D * Шаблон 2D -> 2D Карта совпадений (от -1.0 до 1.0).
    """
    sig_v, _ = sig.values
    ker_v, _ = ker.values
    
    F_sig, T_sig = sig_v.shape[-2], sig_v.shape[-1]
    F_ker, T_ker = ker_v.shape[-2], ker_v.shape[-1]
    N_K = F_ker * T_ker  # Количество пикселей в шаблоне
    
    # 1. ЦЕНТРИРОВАНИЕ ШАБЛОНА (Вычитаем среднее)
    ker_mean = ker_v.mean(dim=(-2, -1), keepdim=True)
    ker_centered = ker_v - ker_mean
    
    # Считаем дисперсию (энергию) центрированного шаблона
    ker_var = torch.sum(ker_centered**2, dim=(-2, -1), keepdim=True)
    ker_sigma = torch.sqrt(ker_var)
    
    # 2. ОТЗЕРКАЛИВАНИЕ ДЛЯ КРОСС-КОРРЕЛЯЦИИ
    # Чтобы FFT-свёртка работала как корреляция, ядро нужно перевернуть!
    ker_flipped = torch.flip(ker_centered, dims=(-2, -1))
    
    # Создаем фиктивное отзеркаленное окно из единиц (для поиска локального среднего сигнала)
    ones_flipped = torch.ones_like(ker_flipped)
    
    # 3. FFT (Подготовка)
    N_f = F_sig + F_ker - 1
    N_t = T_sig + T_ker - 1
    
    Sig_f  = torch.fft.rfft2(sig_v, s=(N_f, N_t), dim=(-2, -1))
    Sig2_f = torch.fft.rfft2(sig_v**2, s=(N_f, N_t), dim=(-2, -1))
    
    Ker_flip_f = torch.fft.rfft2(ker_flipped, s=(N_f, N_t), dim=(-2, -1))
    Ones_flip_f = torch.fft.rfft2(ones_flipped, s=(N_f, N_t), dim=(-2, -1))
    
    # 4. ВЫЧИСЛЕНИЯ В ЧАСТОТНОЙ ОБЛАСТИ (Числитель и Знаменатель)
    
    # Числитель: Ковариация (Свертка сигнала с отзеркаленным центрированным ядром)
    Num = torch.fft.irfft2(Sig_f * Ker_flip_f, s=(N_f, N_t), dim=(-2, -1))
    
    # Суммы для локальной дисперсии сигнала
    Sum_S  = torch.fft.irfft2(Sig_f * Ones_flip_f, s=(N_f, N_t), dim=(-2, -1))
    Sum_S2 = torch.fft.irfft2(Sig2_f * Ones_flip_f, s=(N_f, N_t), dim=(-2, -1))
    
    # Локальная дисперсия: Var(S) = Sum(S^2) - (Sum(S))^2 / N_K
    Sig_var = Sum_S2 - (Sum_S**2) / N_K
    # Из-за погрешностей float могут быть микро-минусы (например -1e-7), зануляем их
    Sig_var = torch.clamp(Sig_var, min=0.0) 
    Sig_sigma = torch.sqrt(Sig_var)
    
    # 5. ИТОГОВЫЙ ПИРСОН
    epsilon = 1e-8
    R = Num / (Sig_sigma * ker_sigma + epsilon)
    R = torch.clamp(R, min=-1.0, max=1.0) # Защита границ [-1, 1]
    
    # 6. ПРАВИЛЬНЫЙ СРЕЗ (Отзеркаленные Якоря)
    # Так как ядро перевернули задом наперед, якорь сместился на противоположный конец
    Z_f = _get_zero_index(ker.freq[0])
    Z_t = _get_zero_index(ker.time[0])
    
    Z_f_corr = (F_ker - 1) - Z_f
    Z_t_corr = (T_ker - 1) - Z_t
    
    out = R[..., Z_f_corr : Z_f_corr + F_sig, Z_t_corr : Z_t_corr + T_sig]
    
    # 7. ФИЗИКА
    # Пирсон - безразмерная величина (UREG.dimensionless)
    return SpecFunc(matrix=(out, UREG.dimensionless), freq=sig.freq[0], time=sig.time[0])


class TimeFuncTopKLoss(nn.Module):
    """
    Функционал ошибки, который оценивает "заметность" найденных паттернов
    на временной оси. Чем выше и острее пики в TimeFunc, тем ниже (лучше) loss.
    """
    def __init__(self, top_k_peaks: int = 20):
        super().__init__()
        if top_k_peaks <= 0:
            raise ValueError("top_k_peaks должно быть положительным числом.")
        self.top_k = top_k_peaks

    def forward(self, time_timeline: TimeFunc) -> torch.Tensor:
        """
        Args:
            time_timeline (TimeFunc): Временная карта корреляций или откликов.
                                      Ожидается, что она хранит тензор с градиентами.

        Returns:
            torch.Tensor: Скалярное значение ошибки (0D тензор).
        """
        # Извлекаем сырой тензор (значения) и игнорируем размерность (PintUnit)
        values, _ = time_timeline.values
        
        # Если есть батчевое измерение, flatten() объединит их, 
        # что соответствует поведению из старого кода. 
        # Если нужно считать top_k для каждого элемента батча отдельно,
        # нужно использовать flatten(start_dim=1) и брать mean по батчу.
        flat_values = values.flatten()
        
        # Защита на случай, если длина сигнала меньше, чем запрашиваемый k
        actual_k = min(self.top_k, flat_values.numel())
        
        # Находим k самых высоких пиков
        top_peaks, _ = torch.topk(flat_values, k=actual_k)
        
        # Наша цель максимизировать корреляцию (пики), 
        # поэтому для минимизации градиентным спуском берем с минусом.
        loss = -torch.mean(top_peaks)
        
        return loss

# =======================================================
# Дополнительный вариант: Ошибка разреженности (Sparsity)
# =======================================================
class TimeFuncSparsityLoss(nn.Module):
    """
    Альтернативный Loss: штрафует за шум во временной функции.
    Заставляет модель делать карту откликов пустой везде, кроме мест идеального совпадения.
    (Минимизирует L1 норму сигнала).
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, time_timeline: TimeFunc) -> torch.Tensor:
        values, _ = time_timeline.values
        # L1 Loss (сумма модулей)
        loss = torch.mean(torch.abs(values))
        return loss * self.weight

def discover_best_pattern(
    spec: 'SpecFunc', 
    duration: float, 
    epochs: int = 100, 
    lr: float = 0.01,
    device: torch.device | str = "cpu",  # <--- Добавлен аргумент device
    top_k_peaks: int = 50,

) -> Tuple['TimeFunc', 'SpecFunc']:
    """
    Оптимизирует паттерн с помощью функции matchPearsonTime.
    """
    
    # 1. Извлекаем сетку и единицы измерения
    S_val, unit_v = spec.values
    time_arr, unit_t = spec.time
    freq_arr, unit_f = spec.freq
    
    dt = spec.dt[0]
    
    # --- ВАЖНО: Переносим исходный сигнал на выбранный device ---
    # Убеждаемся, что внутри лежат тензоры, и кидаем их на GPU/CPU
    
    S_val = S_val.to(device, dtype=torch.float32)
    time_arr = time_arr.to(device)
    freq_arr = freq_arr.to(device)
    
    # Пересобираем spec, чтобы он жил на нужном устройстве
    spec_device = SpecFunc(
        matrix=(S_val, unit_v),
        time=time_arr,
        freq=freq_arr
    )
    # -------------------------------------------------------------

    # Количество временных шагов для паттерна
    W_ker = max(1, int(round(duration / dt)))
    F_bins = len(freq_arr)
    
    # Создаем оси времени для паттерна сразу на нужном устройстве
    pattern_time_arr = torch.arange(W_ker, device=device, dtype=torch.float32) * dt - W_ker//4 * dt
    
    # 2. Создаем обучаемый тензор паттерна 
    # Указываем device прямо при инициализации случайного тензора!
    pattern_tensor = nn.Parameter(
        torch.eye(W_ker, dtype=torch.float32, device=device)[W_ker//4].unsqueeze(0).repeat(F_bins, 1) * 1.0
    )

    pattern_spec = SpecFunc(
        matrix=(pattern_tensor, unit_v), 
        time=pattern_time_arr,
        freq=freq_arr
    )

    # 3. Инициализируем оптимизатор и ваш кастомный Loss
    optimizer = torch.optim.Adam([pattern_tensor], lr=lr)
    loss_fn = TimeFuncTopKLoss(top_k_peaks=top_k_peaks) 
    # Если лосс содержит внутренние тензоры (тут вроде нет, но на всякий случай)
    loss_fn.to(device)
    
    print(f"Начинаем поиск паттерна через matchPearsonTime на устройстве: {device}...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 5. Вызываем вашу функцию 2D-поиска
        correlogram_timefunc = matchPearsonTime(spec_device, pattern_spec)
        
        # 6. Считаем Loss
        loss = loss_fn(correlogram_timefunc)
        
        # 7. Шаг оптимизации
        loss.backward()
        optimizer.step()
        
        if epoch % max(1, epochs // 10) == 0:
             print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
             
    print("matchPearsonTime закончил работу.")

    # 8. Финализируем результаты (отвязываем от графа PyTorch и переводим в numpy для вывода)
    with torch.no_grad():
        # final_pattern_np = pattern_tensor.detach().cpu().numpy()
        # pattern_time_np = pattern_time_arr.cpu().numpy()
        # freq_arr_np = freq_arr.cpu().numpy()
        
        # # Формируем итоговый объект паттерна в numpy
        # final_pattern_spec = SpecFunc(
        #     matrix=(final_pattern_np, unit_v), 
        #     time=pattern_time_np,
        #     freq=freq_arr_np
        # )
        
        # Делаем финальный проход, чтобы получить чистую итоговую корелограмму
        final_correlogram = matchPearsonTime(spec_device, pattern_spec) # Используем тензорный pattern_spec
        
    return final_correlogram, pattern_spec



def discover_multiple_patterns(
    spec: 'SpecFunc', 
    duration: float, 
    num_patterns: int = 3,           # <--- НОВОЕ: Количество искомых паттернов
    repulsion_weight: float = 0.5,   # <--- НОВОЕ: Сила отталкивания паттернов
    epochs: int = 100, 
    lr: float = 0.01,
    device: torch.device | str = "cpu",
    top_k_peaks: int = 50,
) -> Tuple[List['TimeFunc'], List['SpecFunc']]:
    """
    Ищет несколько независимых паттернов одновременно.
    Использует взаимную кросс-корреляцию паттернов для штрафа за их "одинаковость",
    причем штраф инвариантен к сдвигу паттернов по времени.
    """
    
    # 1. Извлекаем сетку и единицы измерения
    S_val, unit_v = spec.values
    time_arr, unit_t = spec.time
    freq_arr, unit_f = spec.freq
    dt = spec.dt[0]
    
    S_val = S_val.to(device, dtype=torch.float32)
    time_arr = time_arr.to(device)
    freq_arr = freq_arr.to(device)
    
    spec_device = SpecFunc(matrix=(S_val, unit_v), time=time_arr, freq=freq_arr)

    # Размеры паттерна
    W_ker = max(1, int(round(duration / dt)))
    F_bins = len(freq_arr)
    pattern_time_arr = torch.arange(W_ker, device=device, dtype=torch.float32) * dt - W_ker//4 * dt
    
    # 2. Инициализация тензора для НЕСКОЛЬКИХ паттернов
    # Форма: [num_patterns, F_bins, W_ker]
    base_pattern = torch.eye(W_ker, dtype=torch.float32, device=device)[W_ker//4].unsqueeze(0).repeat(F_bins, 1)
    
    # Копируем базовый паттерн N раз
    patterns_init = base_pattern.unsqueeze(0).repeat(num_patterns, 1, 1)
    
    # ВАЖНО: Добавляем случайный шум! 
    # Это "нарушит симметрию", чтобы паттерны с первой эпохи пошли искать РАЗНЫЕ фичи в сигнале.
    patterns_init = patterns_init + torch.randn_like(patterns_init) * 0.05
    
    patterns_tensor = nn.Parameter(patterns_init)

    # 3. Оптимизатор и базовый Loss для поиска в сигнале
    optimizer = torch.optim.Adam([patterns_tensor], lr=lr)
    loss_topk_fn = TimeFuncTopKLoss(top_k_peaks=top_k_peaks)
    
    print(f"Начинаем поиск {num_patterns} паттернов на устройстве: {device}...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_topk_loss = torch.tensor(0.0, device=device)
        pattern_specs = []
        
        # --- ШАГ А: Поиск паттернов в сигнале (TopK Loss) ---
        for i in range(num_patterns):
            # Собираем SpecFunc для конкретного паттерна
            pat_spec = SpecFunc(
                matrix=(patterns_tensor[i], unit_v), 
                time=pattern_time_arr,
                freq=freq_arr
            )
            pattern_specs.append(pat_spec)
            
            # Ищем i-й паттерн в главном сигнале
            corr_timefunc = matchPearsonTime(spec_device, pat_spec)
            
            # Суммируем ошибку (хотим, чтобы у каждого паттерна были яркие пики)
            total_topk_loss += loss_topk_fn(corr_timefunc)
            
        total_topk_loss = total_topk_loss / num_patterns
        
        # --- ШАГ Б: Отталкивание паттернов (Repulsion Loss) ---
        # Штрафуем за похожесть паттернов друг на друга (инвариантно к сдвигу по времени)
        repulsion_loss = torch.tensor(0.0, device=device)
        pairs_count = 0
        
        if num_patterns > 1 and repulsion_weight > 0.0:
            for i in range(num_patterns):
                for j in range(i + 1, num_patterns):
                    # Ищем один паттерн внутри другого! 
                    # matchPearsonTime вернет кросс-корреляцию при ВСЕХ временных сдвигах.
                    inter_corr = matchPearsonTime(pattern_specs[i], pattern_specs[j])
                    inter_vals, _ = inter_corr.values
                    
                    # Максимум по времени — это наивысшая похожесть при ИДЕАЛЬНОМ наложении
                    # (именно это дает инвариантность к трансляции по оси времени)
                    max_similarity = torch.max(inter_vals)
                    
                    # Штрафуем только за положительную похожесть.
                    # Квадрат делает градиенты плавнее.
                    repulsion_loss += torch.relu(max_similarity) ** 2 
                    pairs_count += 1
                    
            repulsion_loss = repulsion_loss / pairs_count

        # --- Итоговый Loss ---
        loss = total_topk_loss + repulsion_weight * repulsion_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % max(1, epochs // 10) == 0:
             print(f"Epoch {epoch}/{epochs} | TopK Loss: {total_topk_loss.item():.4f} | Repulsion: {repulsion_loss.item():.4f} | Total: {loss.item():.4f}")
             
    print("Поиск паттернов завершен.")

    # 4. Финализируем результаты и возвращаем списки
    final_correlograms = []
    final_pattern_specs = []
    
    with torch.no_grad():
        for i in range(num_patterns):
            pat_spec = SpecFunc(
                matrix=(patterns_tensor[i], unit_v), 
                time=pattern_time_arr,
                freq=freq_arr
            )
            final_pattern_specs.append(pat_spec)
            final_correlograms.append(matchPearsonTime(spec_device, pat_spec))
        
    return final_correlograms, final_pattern_specs