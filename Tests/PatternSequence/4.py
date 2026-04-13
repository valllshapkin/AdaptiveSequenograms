import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

# ==========================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ И GROUND TRUTH
# ==========================================
def generate_patterns(freq_bins=64):
    # (Код генерации паттернов остался без изменений)
    patterns = []
    p1 = np.zeros((freq_bins, 40)); p1[20:25, :] = 1.0; patterns.append(p1)
    p2 = np.zeros((freq_bins, 10)); p2[:, 3:7] = 1.0; patterns.append(p2)
    p3 = np.zeros((freq_bins, 30)); np.fill_diagonal(p3[10:, :], 1.0); patterns.append(p3)
    p4 = np.zeros((freq_bins, 30)); np.fill_diagonal(np.fliplr(p4[20:50, :]), 1.0); patterns.append(p4)
    p5 = np.zeros((freq_bins, 20)); p5[40:50, 5:15] = 1.0; patterns.append(p5)
    return patterns

def generate_spectrogram(time_steps=5000, freq_bins=64, num_events=150):
    patterns = generate_patterns(freq_bins)
    spectrogram = np.random.uniform(0.0, 0.1, (freq_bins, time_steps))
    gt_signals = np.zeros((len(patterns), time_steps))
    
    for _ in range(num_events):
        p_idx = np.random.randint(0, len(patterns))
        p = patterns[p_idx]
        t_start = np.random.randint(0, time_steps - p.shape[1])
        amplitude = np.random.uniform(1.0, 2.0)
        
        spectrogram[:, t_start:t_start+p.shape[1]] += p * amplitude
        gt_signals[p_idx, t_start] = 1.0 
        
    return spectrogram, gt_signals

print("Генерация данных...")
V_orig, gt_signals = generate_spectrogram(time_steps=5000)
F, T_orig = V_orig.shape

# ==========================================
# 2. CONV-NMF С ГРАНИЧНЫМ ПАДДИНГОМ
# ==========================================
K = 9        # Количество компонентов (с запасом)
L = 45       # Длина шаблона
epochs = 200 # Количество итераций
sparsity_H = 1.5 # Штраф на активации
sparsity_W = 0.1 # Штраф на шаблоны

# --- ГЛАВНОЕ ИЗМЕНЕНИЕ: ДОБАВЛЯЕМ НУЛИ (ZERO-PADDING) ---
# Добавляем справа L-1 нулевых столбцов, чтобы избежать краевых эффектов
padding_amount = L - 1
V = np.pad(V_orig, ((0, 0), (0, padding_amount)), 'constant')
_, T_padded = V.shape
print(f"Размер оригинала: {T_orig}, Размер с паддингом: {T_padded}")

# H остается оригинального размера, т.к. активации соответствуют началу паттерна
W = np.random.rand(F, K, L)
H = np.random.rand(K, T_orig)

print(f"\nЗапуск Conv-NMF с паддингом...")
start_time = time.time()

for epoch in range(epochs):
    # Реконструкция V_hat (размером с паддингом)
    V_hat = np.zeros((F, T_padded))
    for l in range(L):
        H_shifted = np.zeros((K, T_padded))
        H_shifted[:, l:l+T_orig] = H
        V_hat += W[:, :, l] @ H_shifted
    
    # Обновление H
    num_H = np.zeros_like(H)
    den_H = np.zeros_like(H)
    for l in range(L):
        # Корреляция W с V и V_hat
        Wt = W[:, :, l].T
        V_correlated = Wt @ V
        V_hat_correlated = Wt @ V_hat
        
        # Сдвигаем результат корреляции, чтобы выровнять по времени
        num_H += np.roll(V_correlated, -l, axis=1)[:, :T_orig]
        den_H += np.roll(V_hat_correlated, -l, axis=1)[:, :T_orig]

    H = H * (num_H / (den_H + sparsity_H + 1e-9))
    
    # Обновление W
    for l in range(L):
        H_shifted = np.zeros((K, T_padded))
        H_shifted[:, l:l+T_orig] = H
        Ht = H_shifted.T
        W[:, :, l] = W[:, :, l] * ((V @ Ht) / (V_hat @ Ht + sparsity_W + 1e-9))
    
    # Нормализация
    for k in range(K):
        norm_factor = np.max(W[:, k, :])
        if norm_factor > 0:
            W[:, k, :] /= norm_factor
            H[k, :] *= norm_factor
    
    if (epoch + 1) % 40 == 0:
        error = np.linalg.norm(V - V_hat)
        print(f"Итерация {epoch+1:3d} | Ошибка: {error:.2f}")

print(f"Обучение завершено за {time.time() - start_time:.2f} сек!")

# ==========================================
# 3. АНАЛИЗ И ВИЗУАЛИЗАЦИЯ (РЕЗУЛЬТАТЫ БЕЗ ПАДДИНГА)
# ==========================================
# Обрезаем паддинг с V_hat для корректного отображения
V_hat_orig = V_hat[:, :T_orig]

# (Остальной код анализа и визуализации остался без изменений)
matched_gt_for_k = {}
for k in range(K):
    correlations = [np.sum(H[k] * gt_signals[p]) for p in range(5)]
    best_p = np.argmax(correlations)
    matched_gt_for_k[k] = best_p if np.max(H[k]) > 0.1 else -1

# Графики...
# 1. Сравнение оригинала и восстановления
fig_rec, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
ax1.imshow(V_orig, aspect='auto', origin='lower', cmap='magma'); ax1.set_title("Оригинал")
ax2.imshow(V_hat_orig, aspect='auto', origin='lower', cmap='magma'); ax2.set_title("Восстановление")
plt.show()

# 2. Шаблоны
print("\n--- ВЫУЧЕННЫЕ ШАБЛОНЫ ---")
fig_t, axes_t = plt.subplots(1, K, figsize=(20, 3))
for k in range(K):
    axes_t[k].imshow(W[:, k, :], aspect='auto', origin='lower', cmap='magma')
    gt_id = matched_gt_for_k[k]
    title = f"Ch {k} (P {gt_id})" if gt_id != -1 else f"Ch {k} (EMPTY)"
    axes_t[k].set_title(title)
    axes_t[k].axis('off')
plt.show()

# 3. Активации
print("\n--- АКТИВАЦИИ И СРАВНЕНИЕ С ИДЕАЛОМ ---")
fig_a, axes_a = plt.subplots(K, 1, figsize=(15, 2 * K), sharex=True)
for k in range(K):
    act = H[k, :]
    axes_a[k].plot(act, color='blue', alpha=0.6, label='NMF Activation')
    max_val = np.max(act)
    if matched_gt_for_k[k] != -1:
        peaks, _ = find_peaks(act, height=max_val*0.2, distance=10)
        axes_a[k].plot(peaks, act[peaks], "x", color='red', markersize=8, label='Detected')
        gt_p = matched_gt_for_k[k]
        true_peaks = np.where(gt_signals[gt_p] == 1.0)[0]
        axes_a[k].plot(true_peaks, [max_val] * len(true_peaks), "o", color='green', markersize=5, label='Ground Truth')
        axes_a[k].set_ylabel(f"Ch {k}\n(GT {gt_p})", rotation=0, labelpad=30, va='center')
    else:
        axes_a[k].set_ylabel(f"Ch {k}\n(Dead)", rotation=0, labelpad=30, va='center')
    if k == 0: axes_a[k].legend(loc='upper right')
    axes_a[k].grid(True, alpha=0.3)
axes_a[-1].set_xlabel("Время (фреймы)")
plt.show()