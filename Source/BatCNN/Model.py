import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function


# ==========================================
# 1. МАГИЯ ГРАДИЕНТОВ (Только для обратного прохода)
# ==========================================
class ApplyWindowToGrad(Function):
    @staticmethod
    def forward(ctx, weight, window):
        ctx.save_for_backward(window)
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad_output):
        window, = ctx.saved_tensors
        grad_weight = grad_output * window
        return grad_weight, None


# ==========================================
# 2. АРХИТЕКТУРА
# ==========================================
class WindowedPainterLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        window = 0.5 + 0.5 * torch.hann_window(kernel_size)
        self.register_buffer("window", window.view(1, 1, kernel_size))
        
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        windowed_weight = ApplyWindowToGrad.apply(self.weight, self.window)
        return F.conv_transpose1d(
            x, 
            weight=windowed_weight, 
            bias=None, 
            stride=self.stride, 
            padding=self.padding,
            output_padding=self.output_padding
        )

class SlidingPerceptronAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER ---
        self.enc_layer1 = nn.Conv1d(in_channels=256, out_channels=192, kernel_size=502, stride=10, padding=251)
        self.enc_norm1 = nn.InstanceNorm1d(192)
        self.enc_act1 = nn.GELU()
        
        self.enc_layer2 = nn.Conv1d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.enc_norm2 = nn.InstanceNorm1d(64)
        self.enc_act2 = nn.GELU()
        
        # Латентный слой 32 канала
        self.enc_bottleneck = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1)
        
        # --- DECODER ---
        self.dec_expand = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=1)
        self.dec_norm1 = nn.InstanceNorm1d(64)
        self.dec_act1 = nn.GELU()
        
        self.dec_layer2 = nn.ConvTranspose1d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.dec_norm2 = nn.InstanceNorm1d(192)
        self.dec_act2 = nn.GELU()
        
        self.dec_layer3 = WindowedPainterLayer(
            in_channels=192, out_channels=256, 
            kernel_size=502, stride=10, padding=251, output_padding=4
        )

    def forward(self, x, return_latent=False):
        x = x.squeeze(1) 
        
        e1 = self.enc_act1(self.enc_norm1(self.enc_layer1(x)))
        e2 = self.enc_act2(self.enc_norm2(self.enc_layer2(e1)))
        
        latent_pre_activation = self.enc_bottleneck(e2)
        latent = F.softplus(latent_pre_activation)
        
        d1 = self.dec_act1(self.dec_norm1(self.dec_expand(latent)))
        d2 = self.dec_act2(self.dec_norm2(self.dec_layer2(d1)))
        
        reconstructed = self.dec_layer3(d2)
        reconstructed = F.softplus(reconstructed, beta=1.0)
        reconstructed = reconstructed.unsqueeze(1)
        
        if return_latent:
            latent_for_vis = latent.unsqueeze(2) 
            return reconstructed, latent_for_vis
        return reconstructed


def calculate_loss(pred, target, latent, sparsity_weight=1e-5):
    valid_pred = pred[:, :, :, 251:-251]
    valid_target = target[:, :, :, 251:-251]
    
    # 1. Мягкая энергетическая маска (внимание на громкие звуки)
    # Нормализуем таргет внутри батча в диапазон ~[0, 1] для весов
    max_vals = valid_target.amax(dim=(2, 3), keepdim=True) + 1e-6
    # Фоновому шуму оставляем вес 0.1, полезным сигналам вес стремится к 1.0
    weight_mask = 0.1 + 0.9 * (valid_target / max_vals)
    
    # 2. Пиксельная ошибка: Huber Loss (Гладкий L1)
    # Не сходит с ума от резких скачков (как MSE), но градиент стабилен возле нуля
    pixel_loss = F.huber_loss(valid_pred, valid_target, delta=2.0, reduction='none')
    weighted_pixel_loss = torch.mean(pixel_loss * weight_mask)
    
    # 3. Структурная ошибка (Edge / Gradient Loss) - Замена SSIM
    # Учит форму (наклоны, края) спектрограмм, заставляя избегать "размытия"
    
    # Разница по оси времени
    diff_pred_t = valid_pred[:, :, :, 1:] - valid_pred[:, :, :, :-1]
    diff_target_t = valid_target[:, :, :, 1:] - valid_target[:, :, :, :-1]
    loss_grad_t = F.l1_loss(diff_pred_t, diff_target_t)
    
    # Разница по оси частот
    diff_pred_f = valid_pred[:, :, 1:, :] - valid_pred[:, :, :-1, :]
    diff_target_f = valid_target[:, :, 1:, :] - valid_target[:, :, :-1, :]
    loss_grad_f = F.l1_loss(diff_pred_f, diff_target_f)
    
    structural_loss = loss_grad_t + loss_grad_f
    
    # 4. Штраф латентного пространства
    sparsity_loss = torch.mean(latent)
    
    # Итоговая ошибка (0.5 для баланса структурной ошибки)
    recon_loss = weighted_pixel_loss + 0.5 * structural_loss
    total_loss = recon_loss + sparsity_weight * sparsity_loss
    
    return total_loss, {'recon': recon_loss, 'sparsity': sparsity_loss}