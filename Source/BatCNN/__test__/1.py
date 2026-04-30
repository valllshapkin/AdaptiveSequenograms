import torch
import torch.nn.functional as F
from torch.autograd import Function

# ── Микро-функция для модификации градиента весов ──────────────────────────────

class ApplyWindowToGrad(Function):
    @staticmethod
    def forward(ctx, weight, window):
        ctx.save_for_backward(window)
        # Просто возвращаем тензор, сохраняя его в графе вычислений
        return weight.view_as(weight)

    @staticmethod
    def backward(ctx, grad_output):
        window, = ctx.saved_tensors
        # grad_output — это стандартный градиент по весам, который посчитал PyTorch.
        # Умножаем его на наше окно (края будут обновляться медленнее).
        grad_weight = grad_output * window
        # Для window градиент не нужен, возвращаем None
        return grad_weight, None


# ── Наша обёртка (теперь это обычная функция) ──────────────────────────────────

def windowed_conv_transpose1d(x, weight, window, stride=1, padding=0, output_padding=0):
    # 1. Оборачиваем веса в наш модификатор градиента
    windowed_weight = ApplyWindowToGrad.apply(weight, window)
    
    # 2. Вызываем стандартную функцию PyTorch (всю сложную математику PyTorch сделает сам!)
    return F.conv_transpose1d(
        x, windowed_weight, bias=None,
        stride=stride, padding=padding,
        output_padding=output_padding
    )


# ── тесты ──────────────────────────────────────────────────────────────────────

def make_window(kernel_size):
    w = 0.5 + 0.5 * torch.hann_window(kernel_size)   # [0.5, ..., 1.0, ..., 0.5]
    return w.view(1, 1, kernel_size)


def test_output_shape():
    B, C_in, C_out, T = 2, 3, 4, 16
    kernel_size, stride = 8, 4
    x      = torch.randn(B, C_in, T)
    weight = torch.randn(C_in, C_out, kernel_size)
    window = make_window(kernel_size)
    # Обрати внимание: убрали .apply()
    out = windowed_conv_transpose1d(x, weight, window, stride, 0, 0)
    expected_T = (T - 1) * stride + kernel_size
    assert out.shape == (B, C_out, expected_T), f"shape mismatch: {out.shape}"
    print("test_output_shape passed:", out.shape)


def test_forward_equals_standard():
    """Прямой проход должен совпадать с обычным conv_transpose1d."""
    B, C_in, C_out, T = 2, 3, 4, 16
    kernel_size, stride = 8, 4
    x      = torch.randn(B, C_in, T)
    weight = torch.randn(C_in, C_out, kernel_size)
    window = make_window(kernel_size)
    out_ours = windowed_conv_transpose1d(x, weight, window, stride, 0, 0)
    out_std  = F.conv_transpose1d(x, weight, bias=None, stride=stride)
    assert torch.allclose(out_ours, out_std), "forward mismatch"
    print("test_forward_equals_standard passed")


def test_grad_weight_scaled_by_window():
    """grad_weight на краях должен быть ~0.5 от grad_weight в центре."""
    B, C_in, C_out, T = 1, 1, 1, 16
    kernel_size, stride = 8, 4
    x      = torch.randn(B, C_in, T, requires_grad=False)
    weight = torch.randn(C_in, C_out, kernel_size, requires_grad=True)
    window = make_window(kernel_size)

    # наш градиент
    out = windowed_conv_transpose1d(x, weight, window, stride, 0, 0)
    out.sum().backward()
    grad_ours = weight.grad.clone()

    # стандартный градиент (без window)
    weight.grad = None
    out_std = F.conv_transpose1d(x, weight, bias=None, stride=stride)
    out_std.sum().backward()
    grad_std = weight.grad.clone()

    w = window.squeeze()
    ratio = grad_ours.squeeze() / grad_std.squeeze()
    assert torch.allclose(ratio, w, atol=1e-5), f"window scaling wrong:\n{ratio}\nvs\n{w}"
    print("test_grad_weight_scaled_by_window passed")
    print("  window:    ", w.tolist())
    print("  grad ratio:", ratio.tolist())


def test_grad_x_correct():
    """gradcheck для grad_x — сравниваем с численным градиентом."""
    B, C_in, C_out, T = 1, 2, 2, 8
    kernel_size, stride = 4, 2
    x      = torch.randn(B, C_in, T, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(C_in, C_out, kernel_size, dtype=torch.float64, requires_grad=False)
    window = make_window(kernel_size).double()

    ok = torch.autograd.gradcheck(
        lambda inp: windowed_conv_transpose1d(inp, weight, window, stride, 0, 0),
        (x,), eps=1e-4, atol=1e-3
    )
    assert ok
    print("test_grad_x_correct passed")


if __name__ == "__main__":
    test_output_shape()
    test_forward_equals_standard()
    test_grad_weight_scaled_by_window()
    test_grad_x_correct()