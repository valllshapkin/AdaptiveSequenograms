import torch
import matplotlib.pyplot as plt

# Параметры
x = torch.tensor([1.0], requires_grad=True)
x_history = [x.item()]

# ВАРИАНТ 1: SGD с моментом (не адаптивный)
optimizer = torch.optim.SGD([x], lr=0.1, momentum=0.5)

# ВАРИАНТ 2: SGD без моментов (самый стабильный)
# optimizer = torch.optim.SGD([x], lr=0.01)

# ВАРИАНТ 3: RAdam с огромным epsilon (подавляет знаменатель)
# optimizer = torch.optim.RAdam([x], lr=0.01, eps=1e-4)

# ВАРИАНТ 4: Ваш старый Adam, но с градиентной клиппингой
# optimizer = torch.optim.Adam([x], lr=0.01)
# grad_clip = 0.1

print(f"Start: {x.item()}")

for step in range(100):
    optimizer.zero_grad()
    loss = torch.abs(x)
    loss.backward()
    
    # ВАРИАНТ 4: Обрезаем градиент перед шагом
    # torch.nn.utils.clip_grad_norm_([x], max_norm=0.01)
    
    optimizer.step()
    
    # ВАРИАНТ 5: Искусственно притягиваем к нулю (кастыль)
    # with torch.no_grad():
    #     x.data = x.data * 0.999  # небольшая затухающая добавка
    
    x_history.append(x.item())
    
    if step % 100 == 0:
        print(f"Step {step:4d}: x = {x.item():.8f}")

# График
plt.figure(figsize=(10, 6))
plt.plot(x_history, linewidth=2)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.title(f"Оптимизатор: {optimizer.__class__.__name__}")
plt.xlabel("Шаг")
plt.ylabel("x")
plt.ylim(-0.1, 0.1)  # Масштабируем чтобы видеть колебания
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nФинал: x = {x.item():.10f}")