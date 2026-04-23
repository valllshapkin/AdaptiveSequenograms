from jaxtyping import Float
from torch import Tensor

# нужна спектрограмма
# нам нужен огромный батч спектрограмм чтобы находить обощение игнорируя шумы

# вырезки спектрограмм достаточно большие чтобы содержать секвенсы
# они очищены и логарифмированы
spec: Float[Tensor, 'batch freq time']

# для них нужны массив звукозаписей
record: Float[Tensor, 'batch time']

