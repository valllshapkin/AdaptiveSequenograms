from NewSpec.Core.ConvFFT import convolveTime, convolveFT
from NewSpec.Core.Spectral.Some import discover_best_pattern, discover_multiple_patterns

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
    record = applyСalibration(record, FlatResponseModel(sensitivity_pa=20))
    spec = makeSpec(record, TEST_HANN_WINODW, overlap=0.8, bins=300)
    zspec = noiseZNormByFreq(spec, noise_percentile=10)
    zblur = blurSpec(zspec, sigma=(3, 3))
    zlow = interpolateByFactors(zblur, (0.4, 0.4))
    update_spec2d("zlow", makeLogDB(zlow, add_one=True))
    # seq, patt = discover_best_pattern(zlow, 0.060, epochs=400, lr=0.01, device="cuda", top_k_peaks=10)

    Lseq, Lpatt = discover_multiple_patterns(zlow, 0.030, 3, repulsion_weight=0.4, epochs=500, lr=0.01, device="cuda", top_k_peaks=10)
    for i, seq in enumerate(Lseq): update_function(f"seq{i}", seq)
    for i, patt in enumerate(Lpatt): update_spec2d(f"patt{i}", makeLogDB(patt, add_one=True))


    # clean_spectrogram = remove_pattern_from_spec(
    #     spec=makeLogDB(zlow, add_one=True),
    #     pattern=patt, 
    # )
    # update_spec2d("clean_spectrogram", clean_spectrogram)

