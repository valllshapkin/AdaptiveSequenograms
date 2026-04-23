from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

# =====================================================================
# 1. Свертка: 1D Сигнал * 1D Окно -> 1D Сигнал
# (Используется для Loss-функций: штрафы изоляции и т.д.)
# =====================================================================
class AnchoredConv1D_1DKernel(nn.Module):
    def __init__(self, data: Float[Tensor, "... W"], zero_index: int):
        super().__init__()
        assert data.dim() >= 1, "Окно должно иметь хотя бы 1 ось (W)"
        assert 0 <= zero_index < data.size(-1), "zero_index выходит за границы"

        if isinstance(data, nn.Parameter):
            self.data = data 
        else:
            self.register_buffer('data', data)

        self.register_buffer('data', data)
        self.zero_index = zero_index

    def forward(self, signal: Float[Tensor, "... T"]) -> Float[Tensor, "... T"]:
        T, W = signal.shape[-1], self.data.shape[-1]
        N = T + W - 1
        
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        
        Z = self.zero_index
        return out[..., Z : Z + T]


# =====================================================================
# 2. Свертка: 1D Сигнал * 2D Окно -> 2D Спектрограмма
# (Генеративная модель: H * W -> V)
# =====================================================================
class AnchoredConv1D_2DKernel(nn.Module):
    def __init__(self, data: Float[Tensor, "... F W"], zero_index: int):
        super().__init__()
        assert data.dim() >= 2, "Окно должно иметь минимум 2 оси (F, W)"
        assert 0 <= zero_index < data.size(-1), "zero_index выходит за границы"
        
        if isinstance(data, nn.Parameter):
            self.data = data 
        else:
            self.register_buffer('data', data)

        self.zero_index = zero_index

    def forward(self, signal: Float[Tensor, "... T"]) -> Float[Tensor, "... F T"]:
        T, W = signal.shape[-1], self.data.shape[-1]
        N = T + W - 1
        
        # Добавляем фиктивную ось F: (..., T) -> (..., 1, T)
        sig_expanded = signal.unsqueeze(-2)
        
        Sig_f = torch.fft.rfft(sig_expanded, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        
        # Broadcasting: (..., 1, N_freq) * (..., F, N_freq) -> (..., F, N_freq)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        
        Z = self.zero_index
        return out[..., Z : Z + T]


# =====================================================================
# 3. Свертка: 2D Сигнал * 2D Окно -> 2D Спектрограмма
# (Реверберация: V * E -> S_hat)
# =====================================================================
class AnchoredConv2D_2DKernel(nn.Module):
    def __init__(self, data: Float[Tensor, "... F W"], zero_index: int):
        super().__init__()
        assert data.dim() >= 2, "Окно должно иметь минимум 2 оси (F, W)"
        assert 0 <= zero_index < data.size(-1), "zero_index выходит за границы"
    
        if isinstance(data, nn.Parameter):
            self.data = data 
        else:
            self.register_buffer('data', data)
            
        self.zero_index = zero_index

    def forward(self, signal: Float[Tensor, "... F T"]) -> Float[Tensor, "... F T"]:
        assert signal.dim() >= 2, "Сигнал должен иметь минимум (F, T)"
        
        F_sig, T = signal.shape[-2], signal.shape[-1]
        F_ker, W = self.data.shape[-2], self.data.shape[-1]
        assert F_sig == F_ker, f"Оси F не совпадают: сигнал {F_sig}, ядро {F_ker}"
        
        N = T + W - 1
        
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        
        # Broadcasting: (..., F, N_freq) * (..., F, N_freq) -> (..., F, N_freq)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        
        Z = self.zero_index
        return out[..., Z : Z + T]


# --- ФИЗИЧЕСКИЙ ТЕСТ ПАЙПЛАЙНА ---
if __name__ == "__main__":
    F_BINS = 64
    TIME_STEPS = 1000
    ANYBATCH = (10, )

    TEMPLATES_COUNT = 4
    TEMPLATES_TIME = 21
    ECHO_L = 51

    print("--- 1. Эмиссия (Формирование чистой спектрограммы V) ---")
    
    # 1 ИЗМЕНЕНИЕ: H теперь обучаемый параметр! (Скрытые секвенограммы)
    # В задачах NMF обычно инициализируется малым случайным шумом
    H_tensor = torch.rand(*ANYBATCH, TEMPLATES_COUNT, TIME_STEPS) * 0.01
    H_tensor[..., 0, 100] = 1.0 # Оставим один явный пик
    H = nn.Parameter(H_tensor)

    # Шаблоны считаем фиксированными (известный словарь)
    templates_data = torch.rand(*ANYBATCH, TEMPLATES_COUNT, F_BINS, TEMPLATES_TIME)
    templates_layer = AnchoredConv1D_2DKernel(data=templates_data, zero_index=0)

    # Прямой проход: H * W
    V_components: Float[Tensor, '... template freq time'] = templates_layer(H)
    V: Float[Tensor, '... freq time'] = V_components.sum(dim=-3)
    print("V (суммированная):", V.shape)


    print("\n--- 2. Реверберация (Применение эха среды E) ---")
    
    # 2 ИЗМЕНЕНИЕ: Эхо теперь тоже обучаемый параметр!
    echo_tensor = torch.rand(*ANYBATCH, F_BINS, ECHO_L)
    echo_data = nn.Parameter(echo_tensor)
    
    echo_layer = AnchoredConv2D_2DKernel(data=echo_data, zero_index=0)
    
    # Прямой проход: V * E
    S_hat: Float[Tensor, '... freq time'] = echo_layer(V)
    print("S_hat (Искаженная спектрограмма):", S_hat.shape)


    print("\n--- 3. Проверка градиентов (Тест графа) ---")
    # Допустим, это реальная наблюдаемая спектрограмма
    S_target = torch.rand_like(S_hat)
    
    # Считаем ошибку (MSE для простоты теста)
    loss = torch.nn.functional.mse_loss(S_hat, S_target)
    loss.backward()

    print("Градиенты секвенограмм (H) получены:", H.grad is not None)
    print("Форма градиента H:", H.grad.shape)
    
    # Так как классы мы не меняли (там стоит register_buffer), градиенты 
    # для профиля эха накапливаются во внутреннем тензоре слоя.
    print("Градиенты профиля эха получены:", echo_layer.data.grad is not None)
    print("Форма градиента эха:", echo_layer.data.grad.shape)


# =====================================================================
# 4. Поиск Шаблона (Кросс-Корреляция): 2D Спектрограмма * 2D Паттерн
# Переписано в стиле AnchoredConv (через отзеркаливание ядра)
# =====================================================================
class AnchoredMatch2D(nn.Module):
    def __init__(self, data: Float[Tensor, "... F W"], zero_f: int, zero_t: int):
        super().__init__()
        assert data.dim() >= 2, "Паттерн должен иметь минимум 2 оси (F, W)"
        assert 0 <= zero_f < data.size(-2), "zero_f выходит за границы"
        assert 0 <= zero_t < data.size(-1), "zero_t выходит за границы"
        
        # МАГИЯ ЗДЕСЬ: Для кросс-корреляции через алгоритм свёртки, 
        # паттерн нужно физически отзеркалить по обеим осям.
        flipped_data = torch.flip(data, dims=(-2, -1))
        
        if isinstance(data, nn.Parameter):
            # Если вы решите обучать сам шаблон поиска
            self.data = nn.Parameter(flipped_data)
        else:
            self.register_buffer('data', flipped_data)
            
        # Раз паттерн отзеркален, координаты якоря тоже нужно отзеркалить!
        F_ker, W_ker = data.shape[-2], data.shape[-1]
        self.Z_f = F_ker - 1 - zero_f
        self.Z_t = W_ker - 1 - zero_t

    def forward(self, signal: Float[Tensor, "... F T"]) -> Float[Tensor, "... F T"]:
        assert signal.dim() >= 2, "Сигнал должен иметь минимум (F, T)"
        
        F_sig, T_sig = signal.shape[-2], signal.shape[-1]
        F_ker, W_ker = self.data.shape[-2], self.data.shape[-1]
        
        N_f = F_sig + F_ker - 1
        N_t = T_sig + W_ker - 1
        
        # Стандартное БПФ
        Sig_f = torch.fft.rfft2(signal, s=(N_f, N_t), dim=(-2, -1))
        Ker_f = torch.fft.rfft2(self.data, s=(N_f, N_t), dim=(-2, -1))
        
        # ОБЫЧНОЕ умножение (без conj!), так как ядро уже отзеркалено
        out = torch.fft.irfft2(Sig_f * Ker_f, s=(N_f, N_t), dim=(-2, -1))
        
        # ВАШ ИДЕАЛЬНЫЙ СРЕЗ (используем новые отзеркаленные якоря Z_f и Z_t)
        return out[..., self.Z_f : self.Z_f + F_sig, self.Z_t : self.Z_t + T_sig]