import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

# =====================================================================
# 4. Кросс-Корреляция: 2D Спектрограмма * 2D Паттерн -> 2D Карта откликов
# (Поиск шаблона со сдвигами по частоте и по времени)
# =====================================================================
class TemplateMatch2D(nn.Module):
    def __init__(self, pattern: Float[Tensor, "F_ker T_ker"], zero_f: int = 0, zero_t: int = 0):
        super().__init__()
        assert pattern.dim() == 2, "Паттерн должен быть 2D (F, T)"
        
        if isinstance(pattern, nn.Parameter):
            self.pattern = pattern
        else:
            self.register_buffer('pattern', pattern)
            
        self.zero_f = zero_f
        self.zero_t = zero_t

    def forward(self, signal: Float[Tensor, "... F_sig T_sig"]) -> Float[Tensor, "... F_out T_out"]:
        F_sig, T_sig = signal.shape[-2], signal.shape[-1]
        F_ker, T_ker = self.pattern.shape[-2], self.pattern.shape[-1]
        
        # 1. Вычисляем размеры для 2D Zero-padding
        N_f = F_sig + F_ker - 1
        N_t = T_sig + T_ker - 1
        
        # 2. 2D прямое БПФ с автоматическим паддингом через параметр s=(...)
        # rfft2 автоматически считает последнее измерение вещественным (time)
        Sig_f = torch.fft.rfft2(signal, s=(N_f, N_t), dim=(-2, -1))
        Ker_f = torch.fft.rfft2(self.pattern, s=(N_f, N_t), dim=(-2, -1))
        
        # 3. Умножение на комплексно-сопряженное (Кросс-корреляция, а не свертка!)
        # Это нужно, чтобы паттерн искался "как есть", без отзеркаливания
        Corr_f = Sig_f * torch.conj(Ker_f)
        
        # 4. Обратное 2D БПФ
        out = torch.fft.irfft2(Corr_f, s=(N_f, N_t), dim=(-2, -1))
        
        # 5. Обрезка (Anchoring). 
        # Если zero_f=0, zero_t=0, то результат показывает отклик, если 
        # левый верхний угол паттерна поместить в (f, t) спектрограммы.
        Z_f, Z_t = self.zero_f, self.zero_t
        
        # Вырезаем область, соответствующую размеру исходного сигнала
        # (или любую другую нужную вам область, например 'valid' режим)
        return out[..., Z_f : Z_f + F_sig, Z_t : Z_t + T_sig]


# --- ТЕСТ 2D ПОИСКА ---
if __name__ == "__main__":
    # Спектрограмма: 300 частот, 50000 шагов по времени
    H_sig, W_sig = 300, 50000 
    
    # Паттерн: например, звук птицы (50 частот, 200 шагов времени)
    H_ker, W_ker = 50, 200    
    
    spectrogram = torch.randn(1, H_sig, W_sig) # С фоновым шумом
    
    # Создаем искомый паттерн
    pattern = torch.zeros(H_ker, W_ker)
    pattern[10:40, 50:150] = 1.0 # Рисуем какой-то прямоугольник
    
    # Вставляем паттерн в спектрограмму в позицию F=100, T=25000
    target_f, target_t = 100, 25000
    spectrogram[0, target_f : target_f + H_ker, target_t : target_t + W_ker] += pattern * 5.0
    
    # Инициализируем слой поиска
    matcher = TemplateMatch2D(pattern)
    
    # Ищем
    print("Ищем паттерн... (это займет доли секунды на GPU и чуть дольше на CPU)")
    response_map = matcher(spectrogram)
    print("Форма карты откликов:", response_map.shape) # Будет (1, 300, 50000)
    
    # Находим координаты максимума
    max_idx = torch.argmax(response_map[0])
    found_f = max_idx // W_sig
    found_t = max_idx % W_sig
    
    print(f"Паттерн был вставлен в: F={target_f}, T={target_t}")
    print(f"Максимум найден в:      F={found_f}, T={found_t}")