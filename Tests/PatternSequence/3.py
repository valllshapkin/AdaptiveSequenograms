import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ И GROUND TRUTH
# ==========================================
def generate_patterns(freq_bins=64):
    patterns = []
    p1 = np.zeros((freq_bins, 40)); p1[20:25, :] = 1.0; patterns.append(p1)
    p2 = np.zeros((freq_bins, 10)); p2[:, 3:7] = 1.0; patterns.append(p2)
    p3 = np.zeros((freq_bins, 30))
    for i in range(30): p3[10 + i, i] = 1.0
    patterns.append(p3)
    p4 = np.zeros((freq_bins, 30))
    for i in range(30): p4[50 - i, i] = 1.0
    patterns.append(p4)
    p5 = np.zeros((freq_bins, 20)); p5[40:50, 5:15] = 1.0; patterns.append(p5)
    return patterns

def generate_spectrogram(time_steps=5000, freq_bins=64, num_events=150):
    patterns = generate_patterns(freq_bins)
    spectrogram = np.random.uniform(0.0, 0.1, (freq_bins, time_steps))
    
    # Массив для хранения правильных ответов (Ground Truth)
    # Форма: (Кол-во_паттернов, Время)
    gt_signals = np.zeros((len(patterns), time_steps))
    
    for _ in range(num_events):
        p_idx = np.random.randint(0, len(patterns))
        p = patterns[p_idx]
        t_start = np.random.randint(0, time_steps - p.shape[1])
        amplitude = np.random.uniform(1.0, 2.0)
        
        spectrogram[:, t_start:t_start+p.shape[1]] += p * amplitude
        # Отмечаем единичкой момент старта паттерна
        gt_signals[p_idx, t_start] = 1.0 
        
    return spectrogram, gt_signals

print("Генерация данных...")
V, gt_signals = generate_spectrogram(time_steps=5000, num_events=150)
F, T = V.shape

# ==========================================
# 2. CONV-NMF (С УБИЙСТВОМ ЛИШНИХ КАНАЛОВ)
# ==========================================
K = 9          # Ищем 9 компонент (при реальных 5)
L = 45         # Ширина окна
epochs = 500   # Больше итераций для идеальной сходимости

# Штрафы (Sparsity). Именно они заставят лишние 4 канала стать нулями.
sparsity_H = 5.5 # Сильный штраф на использование каналов во времени
sparsity_W = 0.5 # Легкий штраф на "грязь" в самих картинках шаблонов

W = np.random.rand(F, K, L)
H = np.random.rand(K, T)

print(f"\nЗапуск Conv-NMF: {epochs} итераций...")
start_time = time.time()

for epoch in range(epochs):
    # Реконструкция
    V_hat = np.zeros((F, T))
    for l in range(L):
        H_shifted = H if l == 0 else np.concatenate([np.zeros((K, l)), H[:, :-l]], axis=1)
        V_hat += W[:, :, l] @ H_shifted
        
    # Обновление H
    num_H = np.zeros_like(H)
    den_H = np.zeros_like(H)
    for l in range(L):
        V_shifted = V if l == 0 else np.concatenate([V[:, l:], np.zeros((F, l))], axis=1)
        V_hat_shifted = V_hat if l == 0 else np.concatenate([V_hat[:, l:], np.zeros((F, l))], axis=1)
            
        Wt = W[:, :, l].T
        num_H += Wt @ V_shifted
        den_H += Wt @ V_hat_shifted
        
    # sparsity_H давит на знаменатель. Если канал слабый, он стремится к 0.
    H = H * (num_H / (den_H + sparsity_H + 1e-9))
    
    # Обновление W
    for l in range(L):
        H_shifted = H if l == 0 else np.concatenate([np.zeros((K, l)), H[:, :-l]], axis=1)
        Ht = H_shifted.T
        W[:, :, l] = W[:, :, l] * ((V @ Ht) / (V_hat @ Ht + sparsity_W + 1e-9))
        
    # Нормализация
    for k in range(K):
        max_val = np.max(W[:, k, :])
        if max_val > 0:
            W[:, k, :] /= max_val
            H[k, :] *= max_val
            
    if (epoch + 1) % 20 == 0:
        print(f"Итерация {epoch+1:3d} | Ошибка: {np.linalg.norm(V - V_hat):.2f}")

print(f"Обучение завершено за {time.time() - start_time:.2f} сек!")

# ==========================================
# 3. АНАЛИЗ И СОПОСТАВЛЕНИЕ С GROUND TRUTH
# ==========================================
# Авто-сопоставление: ищем, какой реальный паттерн (из 5) скрывается за каждым выученным каналом (из 9)
matched_gt_for_k = {}
for k in range(K):
    # Считаем корреляцию канала H[k] с каждым из 5 правильных ответов
    correlations = [np.sum(H[k] * gt_signals[p]) for p in range(5)]
    best_p = np.argmax(correlations)
    
    # Если канал пустой (убит sparsity), игнорируем
    if np.max(H[k]) < 0.1:
        matched_gt_for_k[k] = -1 # Канал мертв
    else:
        matched_gt_for_k[k] = best_p

# Отрисовка Шаблонов
print("\n--- ВЫУЧЕННЫЕ ШАБЛОНЫ ---")
fig_t, axes_t = plt.subplots(1, K, figsize=(20, 3))
for k in range(K):
    axes_t[k].imshow(W[:, k, :], aspect='auto', origin='lower', cmap='magma')
    gt_id = matched_gt_for_k[k]
    title = f"Ch {k} (Pattern {gt_id})" if gt_id != -1 else f"Ch {k} (EMPTY)"
    axes_t[k].set_title(title)
    axes_t[k].axis('off')
plt.show()

# Отрисовка Активаций с Идеальными ответами
print("\n--- АКТИВАЦИИ И СРАВНЕНИЕ С ИДЕАЛОМ ---")
fig_a, axes_a = plt.subplots(K, 1, figsize=(15, 2 * K), sharex=True)

for k in range(K):
    act = H[k, :]
    axes_a[k].plot(act, color='blue', alpha=0.6, label='Conv-NMF Activation')
    
    max_val = np.max(act)
    if matched_gt_for_k[k] != -1:
        # Находим и рисуем пики NMF
        peaks, _ = find_peaks(act, height=max_val*0.2, distance=10)
        axes_a[k].plot(peaks, act[peaks], "x", color='red', markersize=8, label='Detected Peaks')
        
        # РИСУЕМ ПРАВИЛЬНЫЕ ОТВЕТЫ (Зеленые линии)
        gt_p = matched_gt_for_k[k]
        true_peaks = np.where(gt_signals[gt_p] == 1.0)[0]
        # Рисуем зеленые точки на высоте максимума для наглядности
        axes_a[k].plot(true_peaks, [max_val] * len(true_peaks), "o", color='green', markersize=5, label='Ground Truth')
        
        axes_a[k].set_ylabel(f"Ch {k}\n(GT {gt_p})", rotation=0, labelpad=30, va='center')
    else:
        axes_a[k].set_ylabel(f"Ch {k}\n(Dead)", rotation=0, labelpad=30, va='center')
        
    if k == 0:
        axes_a[k].legend(loc='upper right')
    axes_a[k].grid(True, alpha=0.3)

axes_a[-1].set_xlabel("Время (фреймы)")
plt.tight_layout()
plt.show()