from pathlib import Path
from typing import Any, Dict, Callable, Union, Optional
from enum import Enum

import torch
from torch import Tensor
from numpy.typing import NDArray

ScriptDir = Path(__file__).parent

class WindowNorm(Enum):
    NONE   = "none"
    ENERGY = "energy"
    AREA   = "area"

class Window:
    """
    Класс для генерации оконных функций.
    """
    def __init__(self, 
            func: Callable[[int], Union[Tensor, NDArray[Any]]], 
            time: float, 
            norm: WindowNorm = WindowNorm.NONE
        ):
        self.func = func
        self.time = time
        self.norm = norm

    def get_array(self, sr: int, device: Optional[Union[torch.device, str]] = None) -> Tensor:
        """
        Возвращает тензор окна заданного размера (с учетом sr) на указанном устройстве.
        """
        size = int(self.time * sr)

        if not 16 < size < 2048:
            print(f"Странный размер окна: {size}")

        # Вызываем функцию из внешнего скрипта
        row = self.func(size)

        # Конвертируем в PyTorch тензор, если внешний скрипт вернул numpy
        if not isinstance(row, Tensor):
            row = torch.tensor(row, dtype=torch.float64, device=device)
        else:
            # Если это уже тензор, переносим на нужное устройство (если указано)
            row = row.to(device=device, dtype=torch.float64)

        # Применяем нормировку средствами PyTorch
        match self.norm:
            case WindowNorm.NONE:   
                return row
            case WindowNorm.ENERGY: 
                return row / torch.sqrt(torch.sum(row ** 2))
            case WindowNorm.AREA:   
                return row / torch.sum(row)


def load_file_window(file: Path, norm: WindowNorm = WindowNorm.ENERGY) -> Dict[str, Window]:
    """
    Динамически загружает оконную функцию из Python файла.
    """
    namespace: Dict[str, Any] = {}

    with file.open("r", encoding="utf-8") as f:
        exec(f.read(), namespace)

    name     = namespace.get("EXPORT_NAME")
    window   = namespace.get("EXPORT_WINDOW")
    duration = namespace.get("EXPORT_DURATION")

    if not name:     raise RuntimeError(f"EXPORT_NAME не задан в {file}")
    if not window:   raise RuntimeError(f"EXPORT_WINDOW не задан в {file}")
    if not duration: raise RuntimeError(f"EXPORT_DURATION не задан в {file}")

    return {name: Window(func=window, time=duration, norm=norm)}


class WindowNormMismatchError(Exception):
    """Исключение, сигнализирующее о несовпадении ожидаемой и фактической нормировки окна."""

    def __init__(self, expected: WindowNorm, actual: WindowNorm, message: Optional[str] = None) -> None:
        self.expected = expected
        self.actual = actual
        if message is None:
            message = f"Несовпадение нормировки окна: ожидалась '{expected.value}', получена '{actual.value}'."
        super().__init__(message)


# Инициализация тестовых окон (остается без изменений)
TEST_HANN_WINODW = load_file_window(ScriptDir / "__assets__" / "Hann.STFT.2.py")["Hann STFT 2ms"]
TEST_HANN_AREA   = load_file_window(ScriptDir / "__assets__" / "Hann.STFT.2.py", norm=WindowNorm.AREA)["Hann STFT 2ms"]
TEST_BIG_WINDOW  = load_file_window(ScriptDir / "__assets__" / "Hann.STFT.600.py")["Hann STFT 600ms"]