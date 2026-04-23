from scipy.signal.windows import hann

EXPORT_NAME = "Hann STFT 600ms"
EXPORT_WINDOW = lambda n: hann(n, sym=True)    
EXPORT_DURATION = 0.1




