from pathlib import Path
ScriptDir = Path(__file__).parent
from BatSpec.Work.Record import loadRecord, correctDC, localRMS, resampleRecord, makeLog
from BatSpec.Work.Record.Calibration import FlatResponseModel, applyСalibration
from BatSpec.Work.SaveIntegral import SaveIntegralEnergy
from BatSpec.Work.Spectral import localZNorm, makeSpec, makeLogDB, correlationTransform, stackSpecsMax
from BatSpec.Work.Spectral.Statistic import noiseZNormByFreq, integrateAndClipEnvelope
from BatSpec.Work.ConvWindow import TEST_HANN_WINODW, TEST_HANN_AREA, TEST_BIG_WINDOW
from BatSpec.Work.Function import Function2D, TimeFunc, SpecFunc
import numpy as np
from scipy import signal

def get_low_freq_spectrogram_hps(f: TimeFunc, window_sec=3.0, max_freq=50.0, num_harmonics=3) -> SpecFunc:
    dt = f.dt[1]
    arr = f.values[1]
    fs = 1.0 / dt
    nperseg = int(window_sec * fs)
    noverlap = int(nperseg * 0.99)
    nfft = 4096 
    
    f_full, t, Sxx_full = signal.spectrogram(
        arr, fs=fs, window='hann', nperseg=nperseg, 
        noverlap=noverlap, nfft=nfft, detrend='constant' 
    )
    
    # --- НАЧАЛО БЛОКА HPS ---
    # Схлопываем гармоники (перемножаем спектр с его сжатыми копиями)
    Sxx_hps = np.copy(Sxx_full)
    for h in range(2, num_harmonics + 1):
        # Сжимаем спектр в h раз
        downsampled = Sxx_full[::h, :]
        # Умножаем (или можно складывать, если Sxx_full перевести в логарифм)
        Sxx_hps[:len(downsampled), :] *= downsampled
    # --- КОНЕЦ БЛОКА HPS ---

    # Обрезаем лишние частоты
    valid_idx = f_full <= max_freq
    f_trim = f_full[valid_idx]
    Sxx = Sxx_hps[valid_idx, :]
    
    return SpecFunc(Sxx.T, t, f_trim, f.values[0])

def get_sliding_autocorrelogram(f: TimeFunc, window_sec=1.0, max_freq=10.0, min_freq=2.0) -> SpecFunc:
    dt = f.dt[1]
    arr = f.values[1]
    fs = 1.0 / dt
    
    win_len = int(window_sec * fs)
    hop_len = int(win_len * 0.01) # Шаг 5% (перекрытие 95%)
    
    # Ограничения по задержкам (чтобы не считать лишнего)
    min_lag = int(fs / max_freq)
    max_lag = int(fs / min_freq)
    
    times = []
    acf_matrix = []
    
    for i in range(0, len(arr) - win_len, hop_len):
        windowed_signal = arr[i : i + win_len]
        # Вычитаем среднее
        windowed_signal = windowed_signal - np.mean(windowed_signal)
        
        # Считаем автокорреляцию
        acf = np.correlate(windowed_signal, windowed_signal, mode='full')
        # Берем только правую половину (положительные задержки)
        acf = acf[len(acf)//2:]
        
        # Вырезаем интересующий диапазон задержек (от min_freq до max_freq)
        acf_matrix.append(acf[min_lag:max_lag])
        times.append(f.time[1][i + win_len//2])
        
    acf_matrix = np.array(acf_matrix)
    
    # Переводим лаги обратно в частоты (Гц) для совместимости оси Y
    lags = np.arange(min_lag, max_lag)
    freqs = fs / lags
    
    # Поскольку lags идут по возрастанию (частоты по убыванию), переворачиваем массив
    acf_matrix = acf_matrix[:, ::-1]
    freqs = freqs[::-1]
    
    # Нормализация, чтобы не было отрицательных значений (если нужны логи)
    acf_matrix[acf_matrix < 0] = 1e-10 
    
    return SpecFunc(acf_matrix, np.array(times), freqs, f.values[0])

record_row = loadRecord(ScriptDir / "MYODAS_20230624_004924.wav")
record_row = correctDC(record_row)
# rms = localRMS(record_row, TEST_HANN_AREA)
# rms = makeLog(rms)
# rms = resampleRecord(rms, 5000)
# rms = correctDC(rms)
# auto = get_low_freq_spectrogram(rms, window_sec=0.5, max_freq=50)
# auto = makeLogDB(SpecFunc.from_Function2D(auto))

record = applyСalibration(record_row, FlatResponseModel(sensitivity_pa=20))

# print(f"""
# SaveIntegralEnergy(record_row): {SaveIntegralEnergy(record_row)}
# SaveIntegralEnergy(record): {SaveIntegralEnergy(record)}
# """)

# print(f"""
# record.values[0]: {record.values[0]}
# record.time[0]: {record.time[0]}
# """)

spec = makeSpec(record, TEST_HANN_WINODW, overlap=0.9, bins=300)
# print(f'''
# spec.values[0]: {spec.values[0]}
# spec.time[0]: {spec.time[0]}
# spec.freq[0]: {spec.freq[0]}
# ''')

# print(f"""
# SaveIntegralEnergy(spec): {SaveIntegralEnergy(spec)}
# SaveIntegralEnergy(record): {SaveIntegralEnergy(record)}
# """)

# SPSL = makeLogDB(spec)
# spec = spec.cloneApply(lambda arr: arr**2, spec.values[0])
zspec = noiseZNormByFreq(spec, noise_percentile=10)
lg = makeLogDB(zspec, add_one=True)
peaks = integrateAndClipEnvelope(zspec, threshold=0.1, resample=1000, make_zero=True)
# peaks = makeLog(peaks, add_one=True)
# low_freq = get_sliding_autocorrelogram(peaks, window_sec=1, max_freq=40, min_freq=1)

from BatSpec.QtApp.Visualize import update_function, update_spec2d, run_visualizer
update_function("record_row", record_row)
# update_function("record", record)
update_function("peaks", peaks)
# update_spec2d("low_freq", makeLogDB(low_freq))
update_spec2d("lg", lg)
# update_spec2d("SPSL", SPSL)


from AnimalSpecies.BatPack import get_TYPE1, get_TYPE_002_001, get_TYPE_002_002, get_TYPE_002_003, get_TYPE_002_004

from AdaptiveSequenograms.Tools import AnchoredMatch2D
from BatSpec.Work.Function import SpecFunc, TimeFunc
import torch
import numpy as np

def makeMatch(spec: SpecFunc, wingen, name: str):
    call = wingen(spec.dt[1], spec.freq[1])
    update_spec2d(f"{name}_call", call)
    call_finnder = AnchoredMatch2D(torch.Tensor(call.values[1].T), len(call.freq[1])//2, len(call.time[1])//2)
    new_spec = call_finnder(torch.Tensor(spec.values[1].T))

    new = spec.cloneApply(lambda arr: new_spec.numpy().T, spec.values[0] * call.values[0])
    update_spec2d(f"{name}_new", new)

    seq = TimeFunc(new.values[1][:, len(call.freq[1])//2], new.time[1], new.values[0])
    update_function(f"{name}_seq", seq)

    return new, seq

makeMatch(lg, get_TYPE1, name="1")

A = stackSpecsMax(*(makeMatch(lg, f, name=f"2_{i+1}")[0] for i, f in enumerate([get_TYPE_002_001, get_TYPE_002_002, get_TYPE_002_003, get_TYPE_002_004])))
B = TimeFunc(A.values[1][:, len(A.freq[1])//2], A.time[1], A.values[0])
update_spec2d("A", A)
update_function("B", B)
run_visualizer()
