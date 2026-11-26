"""Cấu hình chuẩn (baseline) – mọi thực nghiệm chỉ thay đổi đúng một yếu tố so với cấu hình này."""
from dataclasses import dataclass, asdict
import numpy as np

# ---- Bài toán (theo đề) ----
MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5], [-0.5, 2.0]])
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)

# ---- Dữ liệu ----
DATA_SEED = 2026          # seed sinh dữ liệu (cố định cho mọi cấu hình)
N_TRAIN_POOL = 2000       # A tối đa = 2000; nhóm A lấy N mẫu đầu (tập lồng nhau)
N_VAL = 400               # validation độc lập: early stopping, checkpoint, chọn mô hình
N_TEST = 100              # test độc lập 100 mẫu (theo đề) – chỉ dùng đánh giá cuối

# ---- Thực nghiệm ----
A_VALUES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
B_VALUES = list(range(1, 11)) + [16, 32, 64]          # 1..10 bắt buộc, mở rộng 16/32/64
B_SMALL_N = 200                                        # bổ sung: ít dữ liệu, huấn luyện đủ số bước
B_SMALL_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
A_STEPS_VALUES = [200, 400, 800, 1200]                 # bổ sung: cố định số bước cập nhật như A = 2000
REF_STEPS_N = 2000                                     # số bước tham chiếu = 300 epoch x ceil(2000/64)
C_VALUES = [1, 2, 3, 4, 6, 8]                          # 1..4 bắt buộc, mở rộng 6/8
C_WIDTH = 32
SEEDS = [11, 22, 33]


@dataclass
class TrainConfig:
    hidden: tuple = (32, 32)       # số nơ-ron mỗi tầng ẩn -> độ rộng & độ sâu
    activation: str = "tanh"       # hàm kích hoạt tầng ẩn
    optimizer: str = "adam"
    lr: float = 1e-2               # learning rate ban đầu
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 50             # early stopping: dừng nếu val MSE không giảm sau 50 epoch
    min_lr: float = 1e-5           # lr cuối của CosineAnnealingLR
    weight_decay: float = 0.0
    seed: int = 11
    n_train: int = 2000

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d
