import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ (Чистый NumPy)
# ==========================================
def generate_patterns(freq_bins=64):
    patterns = []
    # 1. Горизонтальная линия (Длина 40)
    p1 = np.zeros((freq_bins, 40)); p1[20:25, :] = 1.0; patterns.append(p1)
    # 2. Вертикальная линия (Длина 10)
    p2 = np.zeros((freq_bins, 10)); p2[:, 3:7] = 1.0; patterns.append(p2)
    # 3. Диагональ вверх (Длина 30)
    p3 = np.zeros((freq_bins, 30))
    for i in range(30): p3[10 + i, i] = 1.0
    patterns.append(p3)
    # 4. Диагональ вниз (Длина 30)
    p4 = np.zeros((freq_bins, 30))
    for i in range(30): p4[50 - i, i] = 1.0
    patterns.append(p4)
    # 5. Квадрат (Длина 20)
    p5 = np.zeros((freq_bins, 20)); p5[40:50, 5:15] = 1.0; patterns.append(p5)
    return patterns

def generate_spectrogram(time_steps=5000, freq_bins=64, num_events=100):
    patterns = generate_patterns(freq_bins)
    # Позитивный шум (NMF требует строго неотрицательных значений)
    spectrogram = np.random.uniform(0.0, 0.1, (freq_bins, time_steps))
    
    for _ in range(num_events):
        p_idx = np.random.randint(0, len(patterns))
        p = patterns[p_idx]
        t_start = np.random.randint(0, time_steps - p.shape[1])
        amplitude = np.random.uniform(1.0, 2.0)
        spectrogram[:, t_start:t_start+p.shape[1]] += p * amplitude
        
    return spectrogram

print("Генерация данных...")
V = generate_spectrogram(time_steps=5000, num_events=150)
F, T = V.shape
print(f"Размер спектрограммы: F={F}, T={T}")

# ==========================================
# 2. АЛГОРИТМ CONV-NMF
# ==========================================
K = 9        # Количество компонентов (каналов)
L = 45       # Длина шаблона (должна быть >= длины самого длинного паттерна)
epochs = 300  # Количество итераций
sparsity = 0.7 # L1 штраф на активации для выделения пиков

# Инициализация случайными числами
W = np.random.rand(F, K, L) # Шаблоны (Частота, Канал, Время шаблона)
H = np.random.rand(K, T)    # Активации (Канал, Время сигнала)

print(f"\nЗапуск Conv-NMF (Компонентов: {K}, Ширина окна: {L})...")
start_time = time.time()

for epoch in range(epochs):
    # 1. Считаем Реконструкцию V_hat
    V_hat = np.zeros((F, T))
    for l in range(L):
        # Сдвиг H вправо на l
        H_shifted = np.zeros_like(H)
        if l == 0: H_shifted = H
        else:      H_shifted[:, l:] = H[:, :-l]
        V_hat += W[:, :, l] @ H_shifted
        
    # 2. Обновление активаций H (Multiplicative Update)
    num_H = np.zeros_like(H)
    den_H = np.zeros_like(H)
    for l in range(L):
        # Сдвиг V и V_hat влево на l
        V_shifted = np.zeros_like(V)
        V_hat_shifted = np.zeros_like(V_hat)
        if l == 0:
            V_shifted = V
            V_hat_shifted = V_hat
        else:
            V_shifted[:, :-l] = V[:, l:]
            V_hat_shifted[:, :-l] = V_hat[:, l:]
            
        Wt = W[:, :, l].T # (K, F)
        num_H += Wt @ V_shifted
        den_H += Wt @ V_hat_shifted
        
    # Применяем обновление + Sparsity
    H = H * (num_H / (den_H + sparsity + 1e-9))
    
    # 3. Обновление шаблонов W
    for l in range(L):
        H_shifted = np.zeros_like(H)
        if l == 0: H_shifted = H
        else:      H_shifted[:, l:] = H[:, :-l]
        
        Ht = H_shifted.T # (T, K)
        W[:, :, l] = W[:, :, l] * ((V @ Ht) / (V_hat @ Ht + 1e-9))
        
    # 4. Нормализация (чтобы W не росло бесконечно, а H не падало к нулю)
    for k in range(K):
        max_val = np.max(W[:, k, :])
        if max_val > 0:
            W[:, k, :] /= max_val
            H[k, :] *= max_val
            
    if (epoch + 1) % 10 == 0:
        error = np.linalg.norm(V - V_hat)
        print(f"Итерация {epoch+1:2d}/{epochs} | Ошибка реконструкции: {error:.2f}")

print(f"Готово за {time.time() - start_time:.2f} секунд!")

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ И ПОИСК СЕКВЕНСОВ
# ==========================================
# 1. Оригинал vs Восстановление
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
ax1.imshow(V, aspect='auto', origin='lower', cmap='magma')
ax1.set_title("Оригинальная Спектрограмма")
ax2.imshow(V_hat, aspect='auto', origin='lower', cmap='magma')
ax2.set_title("Восстановление (Conv-NMF)")
plt.tight_layout()
plt.show()

# 2. Выученные шаблоны
print("\n--- ШАБЛОНЫ CONV-NMF (Окно L=45) ---")
fig_t, axes_t = plt.subplots(1, K, figsize=(20, 3))
for k in range(K):
    axes_t[k].imshow(W[:, k, :], aspect='auto', origin='lower', cmap='magma')
    axes_t[k].set_title(f"Компонент {k}")
    axes_t[k].axis('off')
plt.show()

# 3. Графики активаций (Секвенсер)
print("\n--- АКТИВАЦИИ ВО ВРЕМЕНИ ---")
fig_a, axes_a = plt.subplots(K, 1, figsize=(15, 12), sharex=True)

for k in range(K):
    act = H[k, :]
    axes_a[k].plot(act, color='blue', alpha=0.7)
    
    max_val = np.max(act)
    if max_val > 0.1: # Отсекаем пустые каналы
        peaks, _ = find_peaks(act, height=max_val*0.3, distance=10)
        axes_a[k].plot(peaks, act[peaks], "x", color='red', markersize=8)
    
    axes_a[k].set_ylabel(f"Ch {k}")
    axes_a[k].grid(True, alpha=0.3)

axes_a[-1].set_xlabel("Время (фреймы)")
plt.tight_layout()
plt.show()