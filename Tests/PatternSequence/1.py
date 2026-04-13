import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ
# ==========================================
def generate_patterns(freq_bins=64):
    """Создаем 5 разных 2D паттернов. Макс ширина = 40 фреймов."""
    patterns = []
    
    # 1. Горизонтальная линия (Тон)
    p1 = np.zeros((freq_bins, 40))
    p1[20:25, :] = 1.0
    patterns.append(p1)
    
    # 2. Вертикальная линия (Клик/Транзиент)
    p2 = np.zeros((freq_bins, 10))
    p2[:, 3:7] = 1.0
    patterns.append(p2)
    
    # 3. Диагональ вверх (Chirp up)
    p3 = np.zeros((freq_bins, 30))
    for i in range(30):
        p3[10 + i, i] = 1.0
    patterns.append(p3)
    
    # 4. Диагональ вниз (Chirp down)
    p4 = np.zeros((freq_bins, 30))
    for i in range(30):
        p4[50 - i, i] = 1.0
    patterns.append(p4)
    
    # 5. Квадратный "блок" шума
    p5 = np.zeros((freq_bins, 20))
    p5[40:50, 5:15] = 1.0
    patterns.append(p5)
    
    return patterns

def generate_spectrogram(time_steps=20000, freq_bins=64, num_events=120):
    patterns = generate_patterns(freq_bins)
    # Шум (фон)
    spectrogram = np.random.normal(0.05, 0.02, (freq_bins, time_steps))
    
    ground_truth = {i: [] for i in range(len(patterns))}
    
    for _ in range(num_events):
        p_idx = np.random.randint(0, len(patterns))
        p = patterns[p_idx]
        p_len = p.shape[1]
        
        t_start = np.random.randint(0, time_steps - p_len)
        amplitude = np.random.uniform(0.8, 1.2) # Делаем паттерны ярче
        
        spectrogram[:, t_start:t_start+p_len] += p * amplitude
        ground_truth[p_idx].append(t_start)
        
    return spectrogram, ground_truth, patterns

# ==========================================
# 2. ИСПРАВЛЕННАЯ НЕЙРОСЕТЬ (Wide Receptive Field)
# ==========================================
class SpectrogramDataset(Dataset):
    def __init__(self, spec, window_size=512): # Увеличили окно для обучения
        self.spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        self.window_size = window_size
        self.time_steps = spec.shape[1]
        
    def __len__(self):
        return 1000
        
    def __getitem__(self, idx):
        start = np.random.randint(0, self.time_steps - self.window_size)
        return self.spec[:, :, start:start+self.window_size]

class SparseFCAE(nn.Module):
    def __init__(self, num_patterns_n=8):
        super(SparseFCAE, self).__init__()
        
        # Рецептивное поле по времени (kernel=9, stride=1, padding=4)
        # За 5 слоев RF составит: 1 + 5*(9-1) = 41 фрейм по времени!
        # Сжатие по частоте: 64 -> 32 -> 16 -> 8 -> 4 -> 1
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, num_patterns_n, kernel_size=(4, 9), stride=(1, 1), padding=(0, 4)), # 4 -> 1, freq=1
            nn.ReLU() # ReLU чтобы пики были только положительные
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_patterns_n, 64, kernel_size=(4, 9), stride=(1, 1), padding=(0, 4)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 1, kernel_size=(4, 9), stride=(2, 1), padding=(1, 4)),
            nn.Sigmoid() 
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

# ==========================================
# 3. ИНТЕРФЕЙС STREAMLIT
# ==========================================
st.set_page_config(layout="wide")
st.title("Секвенсер Паттернов: Исправленная архитектура (2D)")

if 'spec' not in st.session_state:
    st.session_state.spec = None
if 'model' not in st.session_state:
    st.session_state.model = None

with st.sidebar:
    st.header("Настройки")
    # L1 сильно уменьшен, чтобы избежать коллапса мод!
    l1_lambda = st.slider("L1 Sparsity", 0.0001, 0.0050, 0.0010, step=0.0001, format="%.4f")
    epochs = st.slider("Эпохи обучения", 1, 15, 5)
    n_latent = st.slider("Количество паттернов сети (N)", 5, 12, 8)
    
    if st.button("1. Сгенерировать данные"):
        with st.spinner("Генерация спектрограммы..."):
            spec, gt, _ = generate_spectrogram(time_steps=20000, freq_bins=64, num_events=200)
            spec = np.clip(spec, 0, 1) # Ограничиваем от 0 до 1
            st.session_state.spec = spec
            st.session_state.gt = gt
        st.success("Готово!")

if st.session_state.spec is not None:
    st.subheader("Исходная спектрограмма (кусок 0-2000)")
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.imshow(st.session_state.spec[:, :2000], aspect='auto', origin='lower', cmap='magma')
    st.pyplot(fig)

with st.sidebar:
    if st.button("2. Обучить нейросеть"):
        if st.session_state.spec is None:
            st.error("Сначала сгенерируй данные!")
        else:
            st.session_state.model = SparseFCAE(num_patterns_n=n_latent)
            dataset = SpectrogramDataset(st.session_state.spec, window_size=512)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            
            optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.001)
            mse_loss = nn.MSELoss()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            st.session_state.model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                total_l1 = 0
                for i, batch in enumerate(dataloader):
                    optimizer.zero_grad()
                    
                    recon, latent = st.session_state.model(batch)
                    
                    loss_recon = mse_loss(recon, batch)
                    loss_l1 = l1_lambda * torch.mean(torch.abs(latent))
                    
                    loss = loss_recon + loss_l1
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss_recon.item()
                    total_l1 += loss_l1.item()
                    
                    if i % 10 == 0:
                        progress = (epoch * len(dataloader) + i) / (epochs * len(dataloader))
                        progress_bar.progress(progress)
                
                status_text.text(f"Эпоха {epoch+1} | MSE: {total_loss/len(dataloader):.4f} | L1: {total_l1/len(dataloader):.4f}")
            
            progress_bar.empty()
            st.success("Модель обучена!")

if st.session_state.model is not None:
    st.session_state.model.eval()
    with torch.no_grad():
        full_tensor = torch.tensor(st.session_state.spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        recon, latent = st.session_state.model(full_tensor)
        
        recon_np = recon.squeeze().numpy()
        latent_np = latent.squeeze().numpy() 

    st.subheader("3. Анализ: Внутренние представления и Пики")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Латентное пространство (Активации во времени)")
        fig, axes = plt.subplots(n_latent, 1, figsize=(15, 2 * n_latent), sharex=True)
        if n_latent == 1: axes = [axes]
        
        for i in range(n_latent):
            act = latent_np[i, :2000]
            axes[i].plot(act, color='blue')
            # Ищем пики. Порог 0.05 отсекает мелкий шум.
            peaks, _ = find_peaks(act, height=0.05, distance=15)
            axes[i].plot(peaks, act[peaks], "x", color='red', markersize=8)
            axes[i].set_ylabel(f"Ch {i}")
            axes[i].grid(True)
        st.pyplot(fig)

    with col2:
        st.write("Выученные 2D Шаблоны (Импульсный отклик декодера)")
        st.markdown("Мы подаем в декодер одиночный спайк для каждого канала. Если сеть поняла паттерны, она нарисует их здесь!")
        
        # Создаем пустой тензор с одиночным импульсом посередине (ширина 100)
        impulse_tensor = torch.zeros((n_latent, n_latent, 1, 100))
        for i in range(n_latent):
            impulse_tensor[i, i, 0, 50] = 1.0 # Спайк в центре
            
        with torch.no_grad():
            patterns_reconstructed = st.session_state.model.decoder(impulse_tensor).squeeze().numpy()
            
        fig_pat, axes_pat = plt.subplots(n_latent, 1, figsize=(3, 2 * n_latent))
        if n_latent == 1: axes_pat = [axes_pat]
        for i in range(n_latent):
            # Показываем кусок шириной 50, где паттерн должен отрисоваться
            axes_pat[i].imshow(patterns_reconstructed[i, :, 25:75], aspect='auto', origin='lower', cmap='magma')
            axes_pat[i].set_ylabel(f"Ch {i}")
            axes_pat[i].set_xticks([])
            axes_pat[i].set_yticks([])
        st.pyplot(fig_pat)

    st.subheader("Восстановленная спектрограмма (Реконструкция 0-2000)")
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.imshow(recon_np[:, :2000], aspect='auto', origin='lower', cmap='magma')
    st.pyplot(fig)