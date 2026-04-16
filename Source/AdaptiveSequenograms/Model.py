import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

class AnchoredWindow(nn.Module):
    """
    Универсальная сущность для 1D-окон с компенсацией фазового сдвига (FFT свертка).
    """
    def __init__(self, data: Float[Tensor, "W"], zero_index: int):
        super().__init__()
        assert data.dim() == 1, "Окно должно быть 1D тензором"
        assert 0 <= zero_index < data.size(0), "zero_index выходит за границы окна"
        
        # Данные окна регистрируются как буфер (переносятся на GPU, сохраняются в state_dict)
        self.register_buffer('data', data)
        self.zero_index = zero_index

    def forward(self, signal: Float[Tensor, "... T"]) -> Float[Tensor, "... T"]:
        T = signal.shape[-1]
        W = self.data.shape[-1]
        N = T + W - 1
        
        # Быстрая свертка без батч-осей (broadcasting)
        Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
        Ker_f = torch.fft.rfft(self.data, n=N, dim=-1)
        out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
        
        # Компенсация сдвига
        Z = self.zero_index
        return out[..., Z : Z + T]


class PeakRegularizationLoss(nn.Module):
    """
    Loss-функция для формирования идеальных изолированных пиков.
    Не использует редукцию (возвращает тензор исходной формы).
    """
    def __init__(
        self, 
        window_size: int = 101, 
        window_weight: float = 1.0, 
        sharpness_weight: float = 1.0
    ):
        super().__init__()
        assert window_size % 2 == 1, "Размер окна должен быть нечетным!"
        
        self.window_weight = window_weight
        self.sharpness_weight = sharpness_weight

        # --- СКРЫТАЯ ИНИЦИАЛИЗАЦИЯ ОКНА ---
        # 1. Создаем массив из единиц
        kernel_data = torch.ones(window_size)
        
        # 2. Находим физический центр и зануляем его
        zero_index = window_size // 2
        kernel_data[zero_index] = 0.0 
        
        # 3. Инкапсулируем в нашу умную сущность.
        # Т.к. это nn.Module, PyTorch автоматически добавит его в список подмодулей.
        self.window = AnchoredWindow(data=kernel_data, zero_index=zero_index)

    def forward(self, x: Float[Tensor, "... T"]) -> Float[Tensor, "... T"]:
        # x содержит значения в диапазоне [0, 1]
        
        # 1. Штраф за соседей (умножаем точку на сумму ее соседей)
        # self.window сама делает FFT и выравнивает фазу
        neighbors_sum = self.window(x)
        window_loss = x * neighbors_sum

        # 2. Штраф за размытие (парабола, заставляющая стремиться к 0 или 1)
        sharpness_loss = x * (1.0 - x)

        # Суммируем штрафы без редукции по осям
        return (self.window_weight * window_loss) + (self.sharpness_weight * sharpness_loss)

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    # Максимально простой API: передали только размер
    loss_fn = PeakRegularizationLoss(window_size=11)
    
    # Автоматический перенос на GPU:
    # Буфер внутри `AnchoredWindow`, который находится внутри `PeakRegularizationLoss`, 
    # автоматически и рекурсивно перенесется на нужное устройство!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = loss_fn.to(device)
    
    # Сигнал произвольной формы (... T)
    x = torch.tensor([
        [0.0, 0.0, 1.0, 0.0, 0.0], # Идеал
        [0.0, 1.0, 1.0, 0.0, 0.0]  # Слипшиеся
    ], device=device)
    
    loss_out = loss_fn(x)
    print(loss_out)