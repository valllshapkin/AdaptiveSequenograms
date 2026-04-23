from pathlib import Path
from typing import Iterator, Self, Callable, Union
from jaxtyping import Float, Shaped
from torch import Tensor
import torch
from NewSpec.Core.Units import PintUnit, Phisical, UREG, unit_mul
from NewSpec.Core.SaveIntegral import SaveIntegral

from jaxtyping import Float
from torch import Tensor


class Function: pass


class Function1D(Function, SaveIntegral[Phisical[Shaped[Tensor, '...']]]):
    _value_a: Shaped[Tensor, '... x']
    _value_u: PintUnit
    _axis__a: Shaped[Tensor, 'x']
    _axis__u: PintUnit

    def __init__(self, values: Phisical[Shaped[Tensor, '... x']], axis: Phisical[Float[Tensor, "axis"]]):
        self._value_a, self._value_u = values
        self._axis__a, self._axis__u = axis
       
    @property
    def _d_axis__a(self) -> Phisical[float]:
        if len(self._axis__a) < 2:
            raise ValueError("Cannot calculate step size for an AnyArray with less than 2 elements.")
        step = float((self._axis__a[-1] - self._axis__a[0]) / (len(self._axis__a) - 1))
        return step, self._axis__u

    def _SaveIntegral_Energy(self) -> Phisical[Shaped[Tensor, '...']]:
        dA, dU = self._d_axis__a
        return torch.sum(self._value_a**2, dim=-1) * dA, unit_mul(self._value_u, self._value_u, dU)

    def _SaveIntegral_Area(self) -> Phisical[Shaped[Tensor, '...']]:
        dA, dU = self._d_axis__a
        return torch.sum(self._value_a, dim=-1) * dA, unit_mul(self._value_u, dU), 

    def _FileSystem_Save(self, path: Path) -> None:
        raise NotImplementedError()
    
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        raise NotImplementedError()

    def __len__(self) -> int:
        if self._value_a.ndim <= 1:
            raise TypeError("Нет батчевого измерения.")
        return self._value_a.shape[0]

    def __getitem__(self, idx: Union[int, slice]) -> 'Function1D':
        if self._value_a.ndim <= 1:
            raise TypeError("Нет батчевого измерения.")
        return Function1D(
            values=(self._value_a[idx], self._value_u),
            axis=(self._axis__a, self._axis__u)
        )

    def __iter__(self) -> Iterator['Function1D']:
        for i in range(len(self)):
            yield self[i]

class TimeFunc(Function1D):
    _axis__u: PintUnit = UREG.second

    def __init__(self, values: Phisical[Shaped[Tensor, '... time']], axis: Float[Tensor, 'time']):
        # Проверяем последнее измерение (time)
        if values[0].shape[-1] != axis.shape[-1]: 
            raise ValueError(f"Последнее измерение values ({values[0].shape[-1]}) и длина axis ({axis.shape[-1]}) должны совпадать")
        self._value_a, self._value_u = values
        self._axis__a = axis


    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        return cls((func._value_a, func._value_u), func._axis__a)
    
    @property
    def time(self) -> Phisical[Float[Tensor, 'time']]:
        return self._axis__a, self._axis__u
    
    @time.setter
    def time(self, value: Float[Tensor, 'time']):
        if len(value) != len(self._axis__a):
            raise ValueError(f"New time AnyArray must have length {len(self._axis__a)}, got {len(value)}")
        self._axis__a = value

    @property
    def start(self) -> float:
        return float(self._axis__a[0])
    
    @property
    def end(self) -> float:
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Phisical[Shaped[Tensor, '... time']]:
        return self._value_a, self._value_u

    @values.setter
    def values(self, value: Phisical[Shaped[Tensor, '... time']]):
        if value[0].shape[-1] != self._axis__a.shape[-1]:
            raise ValueError(f"Новый массив значений должен иметь размер последнего измерения {self._axis__a.shape[-1]}, получено {value[0].shape[-1]}")
        self._value_a, self._value_u = value
        
    @property
    def dt(self) -> Phisical[float]:
        return self._d_axis__a

    @property
    def sr(self) -> int:
        dT, _ = self.dt
        return round(1.0 / dT)

    def _FileSystem_Save(self, path: Path):
        Function1D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path):
        return cls.from_Function1D(Function1D._FileSystem_Load(path))

    def __getitem__(self, idx: Union[int, slice]) -> 'TimeFunc':
        return TimeFunc.from_Function1D(Function1D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['TimeFunc']:
        for i in range(len(self)):
            yield self[i]



class FreqFunc(Function1D):
    _axis__u: PintUnit = UREG.hertz

    def __init__(self, values: Phisical[Shaped[Tensor, '... freq']], axis: Float[Tensor, 'freq']):
        if values[0].shape[-1] != axis.shape[-1]: 
            raise ValueError(f"Последнее измерение values ({values[0].shape[-1]}) и длина axis ({axis.shape[-1]}) должны совпадать")
        self._value_a, self._value_u = values
        self._axis__a = axis

    @classmethod
    def from_Function1D(cls, func: Function1D) -> Self:
        """Создает экземпляр FreqFunc из базового Function1D."""
        return cls((func._value_a, func._value_u), func._axis__a)
    
    @property
    def freq(self) -> Phisical[Float[Tensor, 'freq']]:
        return self._axis__a, self._axis__u

    @freq.setter
    def freq(self, value: Float[Tensor, 'freq']):
        """Устанавливает новый массив частот."""
        if len(value) != len(self._axis__a):
            raise ValueError(f"Новый массив частот должен иметь длину {len(self._axis__a)}, получено {len(value)}")
        self._axis__a = value

    @property
    def min_freq(self) -> float:
        """Возвращает минимальное значение частоты."""
        return float(self._axis__a[0])
    
    @property
    def max_freq(self) -> float:
        """Возвращает максимальное значение частоты."""
        return float(self._axis__a[-1])
    
    @property
    def values(self) -> Phisical[Shaped[Tensor, '... freq']]:
        """Возвращает значения: (тензор, единица измерения)."""
        return self._value_a, self._value_u

    @values.setter
    def values(self, value: Phisical[Shaped[Tensor, '... freq']]):
        if value[0].shape[-1] != self._axis__a.shape[-1]:
            raise ValueError(f"Новый массив значений должен иметь размер последнего измерения {self._axis__a.shape[-1]}, получено {value[0].shape[-1]}")
        self._value_a, self._value_u = value
        
    @property
    def df(self) -> Phisical[float]:
        """Возвращает шаг по частоте (delta frequency) и единицу измерения."""
        return self._d_axis__a

    def _FileSystem_Save(self, path: Path):
        """Сохраняет объект в файл."""
        Function1D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        """Загружает объект из файла."""
        return cls.from_Function1D(Function1D._FileSystem_Load(path))
    
    def __getitem__(self, idx: Union[int, slice]) -> 'FreqFunc':
        return FreqFunc.from_Function1D(Function1D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['FreqFunc']:
        for i in range(len(self)):
            yield self[i]

    

class Function2D(Function, SaveIntegral[Phisical[Shaped[Tensor, '...']]]):
    _matrx_a: Shaped[Tensor, '... x y']
    _matrx_u: PintUnit
    _first_a: Shaped[Tensor, 'x']
    _first_u: PintUnit
    _sec___a: Shaped[Tensor, 'y']
    _sec___u: PintUnit

    def __init__(self, 
                 matrix: Phisical[Shaped[Tensor, '... x y']], 
                 first: Phisical[Float[Tensor, "x"]], 
                 sec: Phisical[Float[Tensor, "y"]]):
        self._matrx_a, self._matrx_u = matrix
        self._first_a, self._first_u = first
        self._sec___a, self._sec___u = sec

    @property
    def _d_first_a(self) -> Phisical[float]:
        if len(self._first_a) < 2:
            raise ValueError("Cannot calculate step size for an array with less than 2 elements.")
        step = float((self._first_a[-1] - self._first_a[0]) / (len(self._first_a) - 1))
        return step, self._first_u

    @property
    def _d_sec___a(self) -> Phisical[float]:
        if len(self._sec___a) < 2:
            raise ValueError("Cannot calculate step size for an array with less than 2 elements.")
        step = float((self._sec___a[-1] - self._sec___a[0]) / (len(self._sec___a) - 1))
        return step, self._sec___u

    def _SaveIntegral_Area(self) -> Phisical[Shaped[Tensor, '...']]:
        f_da, f_du = self._d_first_a
        s_da, s_du = self._d_sec___a
        return torch.sum(self._matrx_a, dim=(-2, -1)) * f_da * s_da, unit_mul(self._matrx_u, f_du, s_du)
    
    def _SaveIntegral_Energy(self) -> Phisical[Shaped[Tensor, '...']]:
        f_da, f_du = self._d_first_a
        s_da, s_du = self._d_sec___a
        return torch.sum(self._matrx_a ** 2, dim=(-2, -1)) * f_da * s_da, unit_mul(self._matrx_u, self._matrx_u, f_du, s_du)

    def _FileSystem_Save(self, path: Path) -> None:
        raise NotImplementedError()
    
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        raise NotImplementedError()

    # --- ДОБАВЛЕНО ДЛЯ ИТЕРАЦИИ ---
    def __len__(self) -> int:
        if self._matrx_a.ndim <= 2:
            raise TypeError("Нет батчевого измерения.")
        return self._matrx_a.shape[0]

    def __getitem__(self, idx: Union[int, slice]) -> 'Function2D':
        if self._matrx_a.ndim <= 2:
            raise TypeError("Нет батчевого измерения.")
        return Function2D(
            matrix=(self._matrx_a[idx], self._matrx_u),
            first=(self._first_a, self._first_u),
            sec=(self._sec___a, self._sec___u)
        )

    def __iter__(self) -> Iterator['Function2D']:
        for i in range(len(self)):
            yield self[i]


class SpecFunc(Function2D):
    _first_u: PintUnit = UREG.hertz
    _sec___u: PintUnit = UREG.second

    def __init__(self, 
                 matrix: Phisical[Shaped[Tensor, '... freq time']], 
                 freq: Float[Tensor, 'freq'],
                 time: Float[Tensor, 'time']):
        if matrix[0].shape[-2] != freq.shape[-1]:
            raise ValueError(f"Размер матрицы по оси частот ({matrix[0].shape[-2]}) не совпадает с длиной оси freq ({freq.shape[-1]})")
        if matrix[0].shape[-1] != time.shape[-1]:
            raise ValueError(f"Размер матрицы по оси времени ({matrix[0].shape[-1]}) не совпадает с длиной оси time ({time.shape[-1]})")
        
        self._matrx_a, self._matrx_u = matrix
        self._first_a = freq  # dim=-2
        self._sec___a = time  # dim=-1

    @classmethod
    def from_Function2D(cls, func: Function2D) -> Self:
        return cls(
            matrix=(func._matrx_a, func._matrx_u), 
            freq=func._first_a, 
            time=func._sec___a
        )

    @property
    def freq(self) -> Phisical[Float[Tensor, 'freq']]:
        return self._first_a, self._first_u
    
    @freq.setter
    def freq(self, value: Float[Tensor, 'freq']):
        if value.shape[-1] != self._first_a.shape[-1]:
            raise ValueError(f"Новая ось частот должна иметь длину {self._first_a.shape[-1]}, получено {value.shape[-1]}")
        self._first_a = value

    @property
    def time(self) -> Phisical[Float[Tensor, 'time']]:
        return self._sec___a, self._sec___u
    
    @time.setter
    def time(self, value: Float[Tensor, 'time']): 
        if value.shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Новая ось времени должна иметь длину {self._sec___a.shape[-1]}, получено {value.shape[-1]}")
        self._sec___a = value

    @property
    def values(self) -> Phisical[Shaped[Tensor, '... freq time']]:
        return self._matrx_a, self._matrx_u
    
    @values.setter
    def values(self, value: Phisical[Shaped[Tensor, '... freq time']]):
        if value[0].shape[-2] != self._first_a.shape[-1]:
            raise ValueError(f"Размер матрицы по частоте (-2) должен быть {self._first_a.shape[-1]}")
        if value[0].shape[-1] != self._sec___a.shape[-1]:
            raise ValueError(f"Размер матрицы по времени (-1) должен быть {self._sec___a.shape[-1]}")
        self._matrx_a, self._matrx_u = value

    @property
    def df(self) -> Phisical[float]:
        return self._d_first_a

    @property
    def dt(self) -> Phisical[float]:
        return self._d_sec___a

    def _FileSystem_Save(self, path: Path):
        Function2D._FileSystem_Save(self, path)
        
    @classmethod
    def _FileSystem_Load(cls, path: Path) -> Self:
        return cls.from_Function2D(Function2D._FileSystem_Load(path))

    def integrateOverTime(self) -> FreqFunc:
        dt_value, dt_unit = self.dt
        integrated_values = torch.sum(self._matrx_a, dim=-1) * dt_value
        new_unit = unit_mul(self._matrx_u, dt_unit)
        return FreqFunc(
            values=(integrated_values, new_unit),
            axis=self._first_a
        )

    def integrateOverFreq(self) -> TimeFunc:
        df_value, df_unit = self.df
        integrated_values = torch.sum(self._matrx_a, dim=-2) * df_value
        new_unit = unit_mul(self._matrx_u, df_unit)
        return TimeFunc(
            values=(integrated_values, new_unit),
            axis=self._sec___a
        )
    
    def cloneApply(self, func: Callable[[Tensor], Tensor], new_unit: PintUnit) -> 'SpecFunc':
        return SpecFunc(
            matrix=(func(self._matrx_a), new_unit), 
            freq=self._first_a,
            time=self._sec___a
        )

    # --- ДОБАВЛЕНО ДЛЯ ИТЕРАЦИИ ---
    def __getitem__(self, idx: Union[int, slice]) -> 'SpecFunc':
        return SpecFunc.from_Function2D(Function2D.__getitem__(self, idx))
    
    def __iter__(self) -> Iterator['SpecFunc']:
        for i in range(len(self)):
            yield self[i]