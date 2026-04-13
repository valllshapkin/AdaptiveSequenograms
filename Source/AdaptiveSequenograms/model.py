import torch
import torch.nn as nn
import torch.nn.functional as F

def fft_conv1d(signal: torch.Tensor, kernel: torch.Tensor, centered: bool = False):
    """Fast 1D convolution via FFT without circular wrap-around."""
    C, T_sig = signal.shape
    _, K_len = kernel.shape
    
    N = T_sig + K_len - 1
    
    Sig_f = torch.fft.rfft(signal, n=N, dim=-1)
    Ker_f = torch.fft.rfft(kernel, n=N, dim=-1)
    
    out = torch.fft.irfft(Sig_f * Ker_f, n=N, dim=-1)
    
    if centered:
        start = K_len // 2
        return out[:, start : start + T_sig]
    else:
        return out[:, :T_sig]

class BlindEchoDeconv(nn.Module):
    def __init__(self, templates, bounds, T, F_bins, E_len):
        super().__init__()
        self.templates = templates
        self.bounds = bounds
        self.K = len(templates)
        self.F_bins = F_bins
        self.T = T
        
        # Learnable parameters
        self.H = nn.Parameter(torch.rand(self.K, T) * 0.01)
        initial_echo = torch.exp(-0.05 * torch.arange(E_len).float()).repeat(F_bins, 1)
        self.E = nn.Parameter(initial_echo)
        
        # Обучаемый фоновый шум (вектор по оси частот), абсорбирует константный шум
        self.bg = nn.Parameter(torch.ones(F_bins, 1) * 0.1)
        
    def forward(self):
        V = torch.zeros(self.F_bins, self.T, device=self.H.device)
        
        for k in range(self.K):
            f_start, f_end = self.bounds[k]
            h_k = self.H[k].unsqueeze(0)
            w_k = self.templates[k]
            
            h_expanded = h_k.expand(w_k.shape[0], -1)
            conv_res = F.relu(fft_conv1d(h_expanded, w_k, centered=True))
            V[f_start:f_end, :] = V[f_start:f_end, :] + conv_res
            
        S_hat = F.relu(fft_conv1d(V, self.E, centered=False))
        
        # Добавляем выученный фон
        return S_hat + self.bg
