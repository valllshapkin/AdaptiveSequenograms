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
        
        # Данные окна регистрируются как буфер
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


class PhysicsInformedSequenogramLoss(nn.Module):
    """
    Калиброванная функция потерь (Gauge Calibrated Loss) для извлечения секвенограмм.
    Объединяет структурные штрафы в изотропном Евклидовом пространстве без ручных весов.
    """
    def __init__(
        self, 
        window_size: int, 
        dt: float = 1.0, 
        expected_density: float = 0.01,
        eps: float = 1e-6
    ):
        """
        window_size: размер окна отталкивания в бинах (должен быть нечетным).
        dt: шаг дискретизации во времени (для инвариантности к частоте дискретизации).
        expected_density: ожидаемое количество пиков на единицу физического времени (rho_0).
        eps: малая константа для численной стабильности градиентов в нуле.
        """
        super().__init__()
        assert window_size % 2 == 1, "Размер окна должен быть нечетным!"
        
        self.dt = dt
        self.rho0 = expected_density
        self.eps = eps
        
        # Физический радиус окна в бинах (tau_w / dt)
        self.W_rad = window_size // 2

        # --- ИНИЦИАЛИЗАЦИЯ ОКНА (Zero Ground State) ---
        kernel_data = torch.ones(window_size)
        zero_index = self.W_rad
        kernel_data[zero_index] = 0.0  # Выколотый центр
        
        self.window = AnchoredWindow(data=kernel_data, zero_index=zero_index)

        # --- ВЫЧИСЛЕНИЕ КАЛИБРОВОЧНЫХ КОНСТАНТ (Unit Variational Restoring Force) ---
        # lambda_sharp = 1.0 (Уже откалибровано по умолчанию)
        
        # lambda_iso = 1 / (4 * rho0 * tau_w * tau_e)
        # где tau_w = W_rad * dt (радиус окна во времени)
        # где tau_e = dt (ожидаемая ширина пика во времени)
        self.lambda_iso = 1.0 / (4.0 * self.rho0 * (self.W_rad * self.dt) * self.dt)


    def forward(self, x: Float[Tensor, "... T"]) -> Float[Tensor, "..."]:
        """
        x: тензор прогнозов (секенограммы) со значениями в [0, 1].
        Возвращает скаляр для каждой секвенограммы (интеграл по времени T).
        """
        
        # --- 1. Функционал резкости (Sharpness / Binarization) ---
        # Интеграл: \int f(t)(1 - f(t)) dt  =>  dt * \sum f_i(1 - f_i)
        raw_sharp = x * (1.0 - x)
        L_sharp = torch.sum(raw_sharp, dim=-1) * self.dt
        
        # --- 2. Функционал изоляции (Isolation / Repulsion) ---
        # Интеграл: \int f(t) (w * f)(t) dt  =>  dt * \sum f_i (w * f)_i
        neighbors_sum = self.window(x)
        raw_iso = x * neighbors_sum
        L_iso_integral = torch.sum(raw_iso, dim=-1) * self.dt
        
        # Применяем калибровочную константу (Gauge Calibration)
        L_iso = self.lambda_iso * L_iso_integral

        # --- 3. Изотропная Евклидова Агрегация ---
        # Геометрия конуса: снимает давление градиента с компонентов, достигших нуля
        L_total = torch.sqrt(L_sharp**2 + L_iso**2 + self.eps)

        # Возвращаем тензор формы `...` (по одному значению на каждую секвенограмму)
        # Если нужен единый скаляр для backward(), пользователь может вызвать .mean()
        return L_total

# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Задаем физические параметры
    DT = 0.01          # Шаг 10 миллисекунд
    DENSITY = 2.0      # Ждем 2 пика в секунду
    WINDOW_BINS = 21   # Окно = 21 бин (радиус 10 бинов = 0.1 секунды)
    
    loss_fn = PhysicsInformedSequenogramLoss(
        window_size=WINDOW_BINS, 
        dt=DT, 
        expected_density=DENSITY
    ).to(device)
    
    # Создаем 3 разные секвенограммы длиной 100 бинов (1 секунда)
    # T = 100
    x = torch.zeros(3, 100, device=device, requires_grad=True)
    
    # Конфигурация 0: ИДЕАЛ (два изолированных единичных пика)
    with torch.no_grad():
        x[0, 20] = 1.0
        x[0, 80] = 1.0
        
    # Конфигурация 1: РАЗМЫТИЕ (значения вокруг 0.5)
    with torch.no_grad():
        x[1, 20:25] = 0.5
        x[1, 80] = 1.0
        
    # Конфигурация 2: СЛИПАНИЕ (два пика внутри радиуса окна 10 бинов)
    with torch.no_grad():
        x[2, 20] = 1.0
        x[2, 25] = 1.0  # Слишком близко к 20
    
    # Вычисляем потери
    losses = loss_fn(x)
    
    print("Форма выхода:", losses.shape)  # Ожидаем torch.Size([3])
    print(f"Loss 0 (Идеал)  : {losses[0].item():.6f}") # Ожидаем ~0 (только eps)
    print(f"Loss 1 (Размытие): {losses[1].item():.6f}")
    print(f"Loss 2 (Слипание): {losses[2].item():.6f}")
    
    # Проверка градиентов
    total_batch_loss = losses.mean()
    total_batch_loss.backward()
    
    print("\nГрадиент в идеальной точке x[0, 20]:", x.grad[0, 20].item())
    print("Градиент в слипшейся точке x[2, 20]:", x.grad[2, 20].item())