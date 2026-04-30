import torch
import psutil

class HardwareConfig:
    """Умный анализатор среды. Адаптирует гиперпараметры под доступное железо."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Константы
        self.CHUNK_SIZE = 1024
        self.FREQ_BINS = 256
        self.TARGET_SR = 256000
        
        # Динамические параметры
        self.BATCH_SIZE = 4
        self.GPU_POOL_SIZE = 4
        self.CPU_QUEUE_SIZE = 2
        
        self._analyze_hardware()

    def _analyze_hardware(self):
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if self.device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            name = torch.cuda.get_device_name(0)
            print(f"[*] Обнаружена GPU: {name} (VRAM: {vram_gb:.1f} GB)")
            print(f"[*] Системная RAM: {ram_gb:.1f} GB")
            
            if vram_gb >= 20:     
                self.BATCH_SIZE = 128
                self.GPU_POOL_SIZE = 40
            elif vram_gb >= 14:   
                self.BATCH_SIZE = 64
                self.GPU_POOL_SIZE = 25
            elif vram_gb >= 8:    
                self.BATCH_SIZE = 32
                self.GPU_POOL_SIZE = 15
            elif vram_gb >= 4:
                self.BATCH_SIZE = 16
                self.GPU_POOL_SIZE = 8
            else: # Ультра-бюджетные карты типа MX150 (2GB VRAM)
                self.BATCH_SIZE = 4
                self.GPU_POOL_SIZE = 4
                
        else:
            print(f"[*] Обнаружен CPU. Обучение будет медленным. (RAM: {ram_gb:.1f} GB)")
            self.BATCH_SIZE = 4
            self.GPU_POOL_SIZE = 2
            
        if ram_gb >= 32:
            self.CPU_QUEUE_SIZE = 15
        elif ram_gb >= 16:
            self.CPU_QUEUE_SIZE = 8
        else:
            self.CPU_QUEUE_SIZE = 3
            
        print(f"[*] Настроено: Batch={self.BATCH_SIZE}, "
              f"GPU_Pool={self.GPU_POOL_SIZE}, CPU_Queue={self.CPU_QUEUE_SIZE}")