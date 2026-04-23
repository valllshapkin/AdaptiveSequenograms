from scipy.signal.windows import hann

EXPORT_NAME = "Hann STFT 2ms"
EXPORT_WINDOW = lambda n: hann(n, sym=False)    
EXPORT_DURATION = 0.002

