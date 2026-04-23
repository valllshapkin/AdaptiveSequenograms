from pathlib import Path

from NewSpec.Core.Functions import SpecFunc, TimeFunc
from NewSpec.Core.Spectral.Echo import deconvolveEcho
ScriptDir = Path(__file__).parent
from NewSpec.Core.ConvWindow import TEST_HANN_WINODW
from NewSpec.Core.Record import loadRecord, correctDC
from NewSpec.Core.Record.Calibration import FlatResponseModel, applyСalibration
from NewSpec.Core.SaveIntegral import SaveIntegral
from NewSpec.Core.Spectral import makeSpec, makeLogDB, interpolateByFactors, blurSpec
from NewSpec.Core.Spectral.Statistic import noiseZNormByFreq
from NewSpec.Core.Spectral.Saprse.second import discover_best_pattern_with_callback
from NewSpec.Core.Spectral.Echo.Blind import deconvolve_echo_wiener_style
from BatSpec.QtApp.Visualize import update_function, update_spec2d, run_visualizer

@run_visualizer
def main():
        
    record = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
    record = correctDC(record)
    print(SaveIntegral.Energy(record))

    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    print(SaveIntegral.Energy(record))
    update_function("record", record)

    spec = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    print(spec.values[0].shape, spec.values[1])

    zspec = noiseZNormByFreq(spec, noise_percentile=10)
    update_spec2d("zspec", makeLogDB(zspec, add_one=True))

    # zblur = blurSpec(zspec, sigma=(4, 4))
    # update_spec2d("zblur", makeLogDB(zblur, add_one=True))

    # low = interpolateByFactors(zblur, (0.4, 0.4))
    # update_spec2d("low", makeLogDB(low, add_one=True))


    # Предполагается, что у вас есть функции:
    # def update_spec2d(name: str, spec: SpecFunc): ...
    # def update_function(name: str, func: TimeFunc): ...
    # def makeLogDB(spec, add_one): ...
    # low: SpecFunc = ...

    def my_visualizer_callback(a: SpecFunc, b: SpecFunc):
        update_spec2d("a", makeLogDB(a))
        update_spec2d("b", makeLogDB(b))

    deconvolve_echo_wiener_style(
        interpolateByFactors(spec, (0.5, 0.5)),
        echo_duration_limit_ms=50,
        epochs=1000,
        learning_rate=0.01,
        lambda_sparsity=0.05,
        lambda_echo_decay=10,
        lambda_echo_positivity=0.5,
        device="cuda",
        callback=my_visualizer_callback,
        callback_interval_percent=1
    )


    # # Запуск основной функции
    # discover_best_pattern_with_callback(
    #     spec=low,
    #     window_duration_ms=1500.0, # 150 сэмплов при dt=10ms
    #     anchor_freq_percent=70.0,  # Ожидаем его вверху
    #     anchor_time_percent=50.0,  # Ожидаем его в центре окна
    #     epochs=150,
    #     learning_rate=0.05,
    #     top_k_peaks=15,
    #     lambda_tv=0.001,
    #     lambda_center=0.2, # Увеличим вес, чтобы якоря работали надежнее
    #     callback=my_visualizer_callback,
    #     callback_interval_percent=20.0

    # )

    print("Обучение завершено.")


    # update_spec2d("low", makeLogDB(low))
    # update_spec2d("A", makeLogDB(A))
