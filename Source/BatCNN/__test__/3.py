from pathlib import Path
import numpy as np

ScriptDir = Path(__file__).parent

# --- Импорты BatSpec ---
from BatSpec.Core.ConvWindow import TEST_HANN_WINODW, TEST_BLHA_WINODW, Window, WindowNorm
from BatSpec.Core.Record import loadRecord, correctDC
from BatSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Core.Spectral import interpolateByFactors, makeLogDB, DSPContext, makeSpec, makePiecewiseLog, makeRobustSpec
from BatSpec.Core.Spectral.Statistic import deadZoneFilter, medianDenoise, multiScaleGeometricMean, multiScaleLogProduct, noiseZNormByFreq, localNoiseZNorm, blurSpec
from BatSpec.Core.Spectral.Smooth import curveEnhanceSpec
from BatSpec.Core.Functions import Function2D # Добавили импорт базового 2D класса
from BatSpec.Core.Physical.Units import UREG  # Добавили импорт размерностей

from BatSpec.Visualize import update_spec2d, run_visualizer, update_function
from MultiArray.Core import ArrayContext, Framework, DeviceType

# --- Импорты MultiArray ---
import MultiArray as ma
from scipy.signal.windows import hann

@run_visualizer
def main():
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")

    record = correctDC(record)
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    update_function("record", record)

    ctx_numpy_cpu = ArrayContext(Framework.NUMPY, DeviceType.CPU, None)
    
    print("--- Основные вычисления ---")
    with DSPContext(ctx_numpy_cpu):
        spec = makeSpec(
            record,
            window=Window(
                lambda n: hann(n, sym=False), 0.001, WindowNorm.ENERGY
            ),
            overlap=0.8,
            bins=300,
        )

    update_spec2d("Spec", makeLogDB(spec, add_one=False))

    zspec = noiseZNormByFreq(spec)

    from BatCNN.Inference import BatCNNInference

    # Инициализируем 1 раз
    inference = BatCNNInference()

    # Получаем твою спектрограмму
    log_spec = makeLogDB(zspec, add_one=True)

    # Прогоняем
    recon_spec, latent_funcs = inference.process(log_spec)

    # Визуализируем
    update_spec2d("Reconstructed_CNN", recon_spec)

    # Выкидываем на графики латентные каналы
    for i, tf in enumerate(latent_funcs):
        update_function(f"Latent_Channel_{i}", tf)


    # =====================================================================
    # ПОСТРОЕНИЕ МАТРИЦЫ САМОПОДОБИЯ (Self-Similarity Matrix)
    # =====================================================================
    print("--- Вычисление Матрицы Самоподобия (SSM) ---")
    
    # 1. Извлекаем сырые NumPy массивы из всех 32 каналов. 
    # Формируем матрицу размерности (32, Time)
    latent_matrix_np = np.array([ma.to_numpy(f.values[0]) for f in latent_funcs])
    
    # 2. Нормализация (L2 Norm). Чтобы косинусное сходство работало корректно,
    # длина вектора состояния в каждый момент времени должна быть равна 1.
    epsilon = 1e-9
    norms = np.linalg.norm(latent_matrix_np, axis=0, keepdims=True) # Норма по каналам (ось 0)
    latent_normalized = latent_matrix_np / (norms + epsilon)

    # 3. Вычисление Косинусного сходства
    # Перемножаем транспонированную матрицу (Time, 32) на обычную (32, Time)
    # Результат: матрица (Time, Time) со значениями от 0.0 до 1.0 (т.к. у нас softplus, минусов нет)
    ssm_np = np.dot(latent_normalized.T, latent_normalized)
    
    # Чтобы матрица выглядела контрастнее, можно возвести её в степень (усилить высокие сходства)
    ssm_np = ssm_np ** 3 

    # 4. Упаковка обратно в объект архитектуры BatSpec
    ctx_orig = latent_funcs[0].context
    ssm_ma = ma.convert_to(ssm_np, ctx_orig)
    
    time_axis_ma, time_unit = latent_funcs[0].time
    
    # Создаем Function2D, где обе оси - это Время
    ssm_func = Function2D(
        matrix=(ssm_ma, UREG.dimensionless), # Значения сходства безразмерны
        first=(time_axis_ma, time_unit),     # Ось Y (Время)
        sec=(time_axis_ma, time_unit)        # Ось X (Время)
    )

    # Отправляем в визуализатор
    update_spec2d("Self-Similarity Matrix", ssm_func)