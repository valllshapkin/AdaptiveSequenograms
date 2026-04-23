from scipy.signal.windows import hann

EXPORT_NAME = "Hann SYM 2ms"
EXPORT_WINDOW = lambda n: hann(n, sym=True)    
EXPORT_DURATION = 0.002




