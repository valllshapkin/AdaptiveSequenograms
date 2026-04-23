import torch
from torch import Tensor
from jaxtyping import Float

# Подставьте ваши актуальные пути импортов
from NewSpec.Core.Units import UREG, unit_mul, unit_devide
from NewSpec.Core.Functions import TimeFunc, FreqFunc, SpecFunc
from NewSpec.Core.Spectral import makeComplexSpec, inverseComplexSpec
from NewSpec.Core.ConvWindow import Window, TEST_HANN_WINODW

# =====================================================================
# 1. Аналитические модели приборов (векторизованные под PyTorch)
# =====================================================================

class CalibrationModel:
    """Базовый класс для аналитической модели АЧХ прибора."""
    
    def __call__(self, freq_hz: Float[Tensor, 'freq']) -> Float[Tensor, 'freq']:
        """
        Возвращает калибровочный коэффициент (Pa / FS) для тензора частот.
        Вычисления производятся векторизованно на устройстве (CPU/GPU) входного тензора.
        """
        raise NotImplementedError

    def generate_calibration_curve(self, freq_axis: Float[Tensor, 'freq']) -> FreqFunc:
        """
        Генерирует кривую калибровки и упаковывает её в FreqFunc с единицами Pa / FS.
        """
        # Вызываем векторизованную модель (быстро работает прямо на GPU)
        coef_values = self(freq_axis)
        unit_pa_per_fs = unit_devide(UREG.Pa, UREG.FS)

        return FreqFunc(
            values=(coef_values, unit_pa_per_fs),
            axis=freq_axis,
        )

        
class FlatResponseModel(CalibrationModel):
    """
    Идеальный прибор с ровной АЧХ.
    1.0 FS всегда равен sensitivity_pa Паскалей на любой частоте.
    """
    def __init__(self, sensitivity_pa: float = 20.0):
        self.sensitivity_pa = sensitivity_pa

    def __call__(self, freq_hz: Float[Tensor, 'freq']) -> Float[Tensor, 'freq']:
        # Создаем тензор того же размера и на том же устройстве, что и freq_hz
        return torch.full_like(freq_hz, self.sensitivity_pa)


class PetterssonM500Model(CalibrationModel):
    """
    Примерная выдуманная модель для ультразвукового микрофона.
    Чувствительность падает на высоких частотах — коэффициент растёт выше 20 кГц.
    """
    def __init__(self, base_pa_per_fs: float = 15.0, rolloff_start_hz: float = 20_000.0,
                 rolloff_rate: float = 5.0 / 10_000.0):
        self.base_pa_per_fs   = base_pa_per_fs
        self.rolloff_start_hz = rolloff_start_hz
        self.rolloff_rate     = rolloff_rate

    def __call__(self, freq_hz: Float[Tensor, 'freq']) -> Float[Tensor, 'freq']:
        # Используем torch.clamp вместо max() для тензоров
        boost = torch.clamp(freq_hz - self.rolloff_start_hz, min=0.0) * self.rolloff_rate
        return self.base_pa_per_fs + boost


class ResonanceMicModel(CalibrationModel):
    """
    Модель микрофона с резонансом.
    На резонансной частоте микрофон выдаёт больший сигнал (Гауссиан вычитается из базы).
    """
    def __init__(self, base_pa: float = 25.0, resonance_hz: float = 40_000.0,
                 resonance_width_hz: float = 5_000.0, resonance_depth_pa: float = 15.0):
        self.base_pa            = base_pa
        self.resonance_hz       = resonance_hz
        self.resonance_width_hz = resonance_width_hz
        self.resonance_depth_pa = resonance_depth_pa

    def __call__(self, freq_hz: Float[Tensor, 'freq']) -> Float[Tensor, 'freq']:
        # Используем torch.exp для векторизованной экспоненты
        exponent   = -0.5 * ((freq_hz - self.resonance_hz) / self.resonance_width_hz) ** 2
        resonance  = self.resonance_depth_pa * torch.exp(exponent)
        return self.base_pa - resonance


# =====================================================================
# 2. Главная функция: TimeFunc(FS) -> TimeFunc(Pa)
# =====================================================================

def applyСalibration(signal_fs: TimeFunc, 
                     model: CalibrationModel, 
                     window: Window = TEST_HANN_WINODW, 
                     overlap: float = 0.5, 
                     bins: int = 300) -> TimeFunc:
    """
    Переводит сырой цифровой сигнал (FS) в физические Паскали (Pa),
    учитывая частотно-зависимую калибровку (АЧХ) прибора.
    """
    # 1. Проверяем единицы входного сигнала
    _, signal_u = signal_fs.values
    if not signal_u.is_compatible_with(UREG.FS):
        raise ValueError(f"Ожидался сигнал в единицах FS, получено: {signal_u}")

    # 2. Переводим в частотно-временную область (STFT)
    complex_spec = makeComplexSpec(signal_fs, window=window, overlap=overlap, bins=bins)

    # 3. Извлекаем оси и генерируем калибровочную кривую
    freq_axis, _ = complex_spec.freq
    calib_curve  = model.generate_calibration_curve(freq_axis)

    # 4. Получаем данные для перемножения
    spec_matrix, spec_u = complex_spec.values  # Форма: (... freq time)
    coef_vector, coef_u = calib_curve.values   # Форма: (freq)

    # 5. Применяем коэффициенты к спектру.
    # Так как spec_matrix имеет форму (... freq time), а coef_vector — (freq),
    # нам нужно добавить пустое измерение в конец вектора, чтобы получилось (freq, 1).
    # Тогда PyTorch автоматически растянет (broadcast) коэффициенты вдоль оси времени.
    coef_vector_broadcasted = coef_vector.unsqueeze(-1)
    
    calibrated_matrix = spec_matrix * coef_vector_broadcasted
    new_spec_unit = unit_mul(spec_u, coef_u)

    # 6. Собираем откалиброванную спектрограмму
    time_axis, _ = complex_spec.time

    calibrated_spec = SpecFunc(
        matrix=(calibrated_matrix, new_spec_unit),
        freq=freq_axis,
        time=time_axis,
    )

    # 7. Обратный STFT -> временной сигнал в Паскалях
    return inverseComplexSpec(calibrated_spec, window, overlap=overlap)