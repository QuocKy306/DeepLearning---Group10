"""MLP tùy biến số tầng ẩn và số nơ-ron mỗi tầng."""
import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid}


class MLP(nn.Module):
    """x (2) -> chuẩn hóa z-score -> [Linear -> activation] x len(hidden) -> Linear (1).

    * Chuẩn hóa đầu vào nằm trong mô hình (buffer in_mean, in_std), được tính trên tập train
      của từng lần chạy -> checkpoint tự chứa, không rò rỉ thống kê của validation/test.
    * Đầu ra tuyến tính (không sigmoid) vì đây là bài toán hồi quy.
    * Trọng số khởi tạo theo mặc định của nn.Linear (Kaiming-uniform, a = sqrt(5)).
    """

    def __init__(self, in_dim: int = 2, hidden=(32, 32), out_dim: int = 1, activation: str = "tanh"):
        super().__init__()
        layers, prev = [], in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), ACTIVATIONS[activation]()]
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)
        self.activation = activation
        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))

    def set_input_stats(self, x_train: torch.Tensor):
        """Lưu mean/std của tập train để chuẩn hóa mọi đầu vào về sau."""
        self.in_mean.copy_(x_train.mean(0))
        self.in_std.copy_(x_train.std(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.in_mean) / self.in_std)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
