import torch
import torch.optim as optim

def kl_divergence(y_true, y_pred, eps=1e-8):
    y_pred = y_pred + eps
    return torch.sum(y_true * torch.log((y_true + eps) / y_pred) - y_true + y_pred)

def train_model(model, S_data, epochs=150, lr=0.01, lambda_l1=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        S_hat = model()
        
        loss_kl = kl_divergence(S_data, S_hat)
        loss_l1 = lambda_l1 * torch.sum(model.H)
        loss = loss_kl + loss_l1
        
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
        
        optimizer.step()
        
        # Ограничения проекции (Non-negativity)
        with torch.no_grad():
            model.H.clamp_(min=1e-8)
            model.E.clamp_(min=1e-8)
            model.bg.clamp_(min=1e-8) # Фон тоже строго положительный
            
            # Нормализация эха (максимум каждой частоты = 1.0) 
            # Это заставляет модель растить пики H вместо бесконечного роста эха E.
            max_E = model.E.max(dim=1, keepdim=True)[0]
            model.E.div_(max_E + 1e-8)
            
        if epoch % 30 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Total Loss: {loss.item():.2f}")
            
    return model
