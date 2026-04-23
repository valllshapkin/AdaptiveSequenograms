import pint
from functools import reduce
import operator

type PintUnit = pint.Unit
UREG = pint.UnitRegistry[float]()

def unit_mul(*units: PintUnit) -> PintUnit:
    return reduce(operator.mul, units)

def unit_devide(uintup: PintUnit, unitdown: PintUnit) -> PintUnit:
    res = uintup / unitdown
    if not isinstance(res, pint.Unit): raise RuntimeError()
    return res

def unit_sqrt(unit: PintUnit) -> PintUnit:
    res = unit**0.5
    if not isinstance(res, pint.Unit): raise RuntimeError()
    return res

# Определяем новую базовую размерность для уровня цифрового сигнала
UREG.define('[digital_level] = 1') 
# Определяем саму единицу измерения "Full Scale"
UREG.define('full_scale = [digital_level] = FS')
# Теперь UREG.FS - это полноценная единица измерения!


from typing import Tuple

type Phisical[T] = Tuple[T, PintUnit]


