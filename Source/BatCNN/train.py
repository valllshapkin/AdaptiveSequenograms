import torch
import time
import argparse
from pathlib import Path

from BatCNN.Config import HardwareConfig
from BatCNN.Model import SlidingPerceptronAutoencoder, calculate_loss
from BatCNN.DataLoader import DualProcessDataLoader
import os
os.chdir(Path(__file__).parent)

class LiveVisualizer:
    # ... (код визуализатора без изменений)
    def __init__(self):
        try:
            import pyqtgraph as pg
            from PySide6 import QtWidgets
            self.has_gui = True
        except ImportError:
            self.has_gui = False
            return

        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
            
        self.win = pg.GraphicsLayoutWidget(title="BatCNN Real-Time Monitor")
        self.win.resize(1200, 400)

        self.p1 = self.win.addPlot(title="Original Spectrogram")
        self.img_orig = pg.ImageItem()
        self.p1.addItem(self.img_orig)

        self.p2 = self.win.addPlot(title="Latent Space (Tokens vs Time)")
        self.img_latent = pg.ImageItem()
        self.p2.addItem(self.img_latent)

        self.p3 = self.win.addPlot(title="Reconstructed")
        self.img_recon = pg.ImageItem()
        self.p3.addItem(self.img_recon)

        colormap = pg.colormap.get('viridis')
        self.img_orig.setColorMap(colormap)
        self.img_latent.setColorMap(pg.colormap.get('plasma'))
        self.img_recon.setColorMap(colormap)

        self.win.show()

    def update(self, orig_tensor, recon_tensor, latent_tensor):
        if not self.has_gui: return
        orig_np = orig_tensor[0, 0].detach().cpu().to(torch.float32).numpy().T
        recon_np = recon_tensor[0, 0].detach().cpu().to(torch.float32).numpy().T
        latent_np = latent_tensor[0].squeeze(1).detach().cpu().to(torch.float32).numpy().T

        self.img_orig.setImage(orig_np, autoLevels=True)
        self.img_recon.setImage(recon_np, autoLevels=True)
        self.img_latent.setImage(latent_np, autoLevels=True)
        self.app.processEvents()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BatCNN Training")
    parser.add_argument('--vis', action='store_true', help="Включить реалтайм визуализацию")
    args = parser.parse_args()

    config = HardwareConfig()
    data_dir = Path(__file__).parent / "Resources"
    wav_paths = list(data_dir.rglob("*.wav"))
    
    if not wav_paths:
        print("[!] ОШИБКА: WAV файлы не найдены!")
        exit(1)

    dataloader = DualProcessDataLoader(wav_paths, config)
    model = SlidingPerceptronAutoencoder().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device='cuda' if config.device == 'cuda' else 'cpu')
    
    CHECKPOINT_FILE = Path("batcnn_weights.pt")
    TEMP_CHECKPOINT = Path("batcnn_weights.tmp")
    start_epoch = 0

    if CHECKPOINT_FILE.exists():
        try:
            print(f"[*] Найден чекпоинт. Загрузка...")
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[*] Успешно. Продолжаем с эпохи {start_epoch+1}.")
        except Exception as e:
            print(f"[!] Ошибка загрузки чекпоинта. Начинаем с нуля.")

    visualizer = LiveVisualizer() if args.vis else None

    EPOCHS = 200
    STEPS_PER_EPOCH = 50
    ACCUMULATION_STEPS = 8 
    
    model.train()
    print("\n================ START TRAINING ================")
    
    try:
        for epoch in range(start_epoch, EPOCHS):
            total_recon_loss = 0
            total_sparsity_loss = 0 # <--- Добавлено для логирования
            start_time = time.time()
            
            optimizer.zero_grad()
            
            for step in range(STEPS_PER_EPOCH):
                batch = dataloader.get_batch()
                
                with torch.autocast(device_type='cuda' if config.device == 'cuda' else 'cpu', dtype=torch.float16):
                    reconstructed, latent = model(batch, return_latent=True)
                    # Loss теперь возвращает общую ошибку и её компоненты
                    total_loss, loss_components = calculate_loss(reconstructed, batch, latent, sparsity_weight=0.01)
                    
                    total_loss = total_loss / ACCUMULATION_STEPS
                
                scaler.scale(total_loss).backward()
                
                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == STEPS_PER_EPOCH:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                with torch.no_grad():
                    total_recon_loss += loss_components['recon'].item() * ACCUMULATION_STEPS
                    total_sparsity_loss += loss_components['sparsity'].item() * ACCUMULATION_STEPS
                
                if visualizer and step % 5 == 0:
                    visualizer.update(batch, reconstructed, latent)
                
            epoch_time = time.time() - start_time
            avg_recon = total_recon_loss / STEPS_PER_EPOCH
            avg_sparsity = total_sparsity_loss / STEPS_PER_EPOCH
            
            # Обновленный лог для вывода всех компонент
            print(f"Эпоха [{epoch+1:03d}/{EPOCHS}] | "
                  f"Recon Loss: {avg_recon:.3f} | "
                  f"Sparsity: {avg_sparsity:.3f} | "
                  f"Время: {epoch_time:.2f} сек.")
                  
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, TEMP_CHECKPOINT)
            TEMP_CHECKPOINT.replace(CHECKPOINT_FILE)

    except KeyboardInterrupt:
        print("\n[!] Обучение прервано. Сохранено.")