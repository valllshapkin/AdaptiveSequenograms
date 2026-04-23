import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Импортируем сгенерированные данные
from AdaptiveSequenograms.TestData import W_tensor, S_tensor, H_tensor, E_tensor, dt_step, TEMPLATE_FRAMES
from AdaptiveSequenograms.Tools import AnchoredConv1D_1DKernel, AnchoredConv1D_2DKernel, AnchoredConv2D_2DKernel

# =========================================================
# 1. ПОДГОТОВКА МОДЕЛИ
# =========================================================
K_TEMPLATES, F_BINS, TIME_STEPS = W_tensor.shape[0], W_tensor.shape[1], S_tensor.shape[-1]
ECHO_FRAMES = E_tensor.shape[-1]

print("Инициализация слепой деконволюции (ConvNMF)...")

NOISE_H = 0.05      # амплитуда шума для H (в долях от [0,1])
NOISE_E = 0.02      # амплитуда шума для E (относительная)

torch.manual_seed(42)

H_noisy = H_tensor.clone()
H_noisy += torch.randn_like(H_noisy) * NOISE_H
H_noisy = H_noisy.clamp(0.0, 1.0)
H_param = nn.Parameter(H_noisy)

E_noisy = E_tensor.clone()
E_noisy += torch.randn_like(E_noisy) * NOISE_E * E_noisy.abs().mean()
E_noisy = E_noisy.clamp(min=1e-8)
E_noisy /= E_noisy.sum(dim=-1, keepdim=True)  # нормировка как в цикле
E_layer = AnchoredConv2D_2DKernel(data=nn.Parameter(E_noisy), zero_index=0)

conv_W = AnchoredConv1D_2DKernel(data=W_tensor, zero_index=TEMPLATE_FRAMES // 2)

# =========================================================
# 2. ПОДГОТОВКА СТРУКТУРНЫХ ШТРАФОВ
# =========================================================
TAU_W_MS = 10.0
tau_w_frames = int((TAU_W_MS / 1000.0) / dt_step)

w_iso_data = torch.ones(1, tau_w_frames * 2 + 1)
w_iso_data[0, tau_w_frames] = 0.0
w_iso_layer = AnchoredConv1D_1DKernel(data=w_iso_data, zero_index=tau_w_frames)

# =========================================================
# 3. НАСТРОЙКА ОПТИМИЗАТОРА
# =========================================================
# ГЛАВНОЕ ИСПРАВЛЕНИЕ: Увеличиваем eps в Adam для стабильности вблизи нуля!
optimizer = torch.optim.SGD([H_param, E_layer.data], lr=0.01, momentum=0.9)
BETA_REG = 1.0 
EPOCHS = 15

print("Начало оптимизации...")
for epoch in range(EPOCHS):
    with torch.no_grad():
        H_param.clamp_(0.0, 1.0)
        E_layer.data.clamp_(min=1e-8)
        E_layer.data /= E_layer.data.sum(dim=-1, keepdim=True)

    optimizer.zero_grad()
    
    V_pred = conv_W(H_param).sum(dim=0)
    S_hat = E_layer(V_pred)
    
    eps_noise = 1e-3
    S_safe = S_tensor + eps_noise
    S_hat_safe = S_hat + eps_noise
    L_rec = torch.sum(S_safe * torch.log(S_safe / S_hat_safe) - S_safe + S_hat_safe) / F_BINS
    
    L_sharp = torch.sum(H_param * (1.0 - H_param))
    density = w_iso_layer(H_param)
    L_iso = 0.5 * torch.sum(H_param * density)
    L_reg = torch.sqrt(L_sharp**2 + L_iso**2 + 1e-4)
    
    loss = L_rec + BETA_REG * L_reg
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_([H_param, E_layer.data], max_norm=1.0)
    optimizer.step()

    if epoch % 1 == 0 or epoch == EPOCHS - 1:
        theta_deg = np.degrees(torch.atan2(L_iso, L_sharp).item())
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
              f"KL/F: {L_rec.item():.4f} | "
              f"Sharp: {L_sharp.item():.4f} | Iso: {L_iso.item():.4f} | "
              f"Угол θ: {theta_deg:.1f}°")

print("Обучение завершено!\n")

# =========================================================
# 4. ВИЗУАЛИЗАЦИЯ
# =========================================================
print("Рендеринг результатов...")
with torch.no_grad():
    V_final = conv_W(H_param).sum(dim=0)
    S_final = E_layer(V_final)
    H_final_np = H_param.cpu().numpy()
    E_final_np = E_layer.data.cpu().numpy()

T_START, T_END = 35_000, 45_000
t_axis = np.arange(T_START, T_END) * dt_step

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle("Адаптивное извлечение секвенограмм (Стабильная Оптимизация)", fontsize=16)

# ИСПРАВЛЕНИЕ ВИЗУАЛИЗАЦИИ: Фиксируем цветовую шкалу, чтобы видеть правду
vmax_echo = E_tensor.numpy().max() * 1.2
vmax_spec = S_tensor[:, T_START:T_END].numpy().max() * 1.2

axes[0, 0].set_title("1A. Истинная наблюдаемая спектрограмма (S)")
axes[0, 0].imshow(S_tensor[:, T_START:T_END].numpy(), aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec)

axes[1, 0].set_title("2A. Истинные Секвенограммы (H_true)")
axes[1, 0].plot(t_axis, H_tensor[0, T_START:T_END].numpy(), color='red', label='H1')
axes[1, 0].plot(t_axis, H_tensor[1, T_START:T_END].numpy(), color='blue', alpha=0.7, label='H2')
axes[1, 0].set_ylim(0, 1.1)

axes[2, 0].set_title("3A. Истинная чистая спектрограмма (V_true)")
axes[2, 0].imshow(conv_W(H_tensor).sum(0)[:, T_START:T_END].numpy(), aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec)

axes[3, 0].set_title("4A. Истинный профиль Эха (E_true)")
axes[3, 0].imshow(E_tensor.numpy(), aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax_echo)

axes[0, 1].set_title("1B. Реконструированная спектрограмма (S_hat)")
axes[0, 1].imshow(S_final[:, T_START:T_END].numpy(), aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec)

axes[1, 1].set_title("2B. Извлеченные Секвенограммы (H_pred)")
axes[1, 1].plot(t_axis, H_final_np[0, T_START:T_END], color='red', label='H1_pred')
axes[1, 1].plot(t_axis, H_final_np[1, T_START:T_END], color='blue', alpha=0.7, label='H2_pred')
axes[1, 1].set_ylim(0, 1.1)

axes[2, 1].set_title("3B. Очищенная спектрограмма (V_pred)")
axes[2, 1].imshow(V_final[:, T_START:T_END].numpy(), aspect='auto', origin='lower', cmap='magma', vmin=0, vmax=vmax_spec)

axes[3, 1].set_title("4B. Изученный профиль Эха (E_pred)")
axes[3, 1].imshow(E_final_np, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=vmax_echo)

for ax in axes.flatten(): ax.label_outer()
plt.tight_layout()
plt.show()