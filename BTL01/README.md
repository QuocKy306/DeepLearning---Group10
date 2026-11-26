# BTL01 – Xấp xỉ hàm Gaussian 2D bằng mạng MLP (PyTorch)

Mạng perceptron đa tầng (MLP) học hàm mật độ Gaussian hai chiều

f(x) = 1 / (2π·√det C) · exp(−½ (x − μ)ᵀ C⁻¹ (x − μ)),  μ = (0, 0),  C = [[0.3, −0.5], [−0.5, 2]],

trên miền −2 ≤ x1 ≤ 2, −4 ≤ x2 ≤ 4, và khảo sát ảnh hưởng của số mẫu huấn luyện **A**, số nơ-ron tầng ẩn **B** và số tầng ẩn **C**.

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Yêu cầu Python ≥ 3.10. Chạy được trên CPU; nếu có GPU, code tự dùng CUDA.

## Chạy lại toàn bộ

```bash
python main.py all                 # dữ liệu → A, B, C, A_steps, B_small, C_full (3 seed) → phân tích, chọn mô hình
```

Thời gian tham khảo: khoảng 30–35 phút trên 1 nhân CPU (142 lần huấn luyện). Kiểm tra nhanh code (1 seed, 30 epoch, ghi vào thư mục riêng):

```bash
python main.py all --quick --results results_quick
```

## Chạy từng bước

| Lệnh | Việc làm |
|---|---|
| `python main.py data` | Sinh `data/train.csv`, `val.csv`, `test.csv` (seed 2026), thống kê và hình dữ liệu |
| `python main.py run --groups A` | Nhóm A: số mẫu train A = 200, 400, …, 2000 |
| `python main.py run --groups B` | Nhóm B: MLP 1 tầng ẩn, B = 1…10 (mở rộng 16, 32, 64) |
| `python main.py run --groups C` | Nhóm C: C = 1, 2, 3, 4 tầng ẩn × 32 nơ-ron (mở rộng 6, 8) |
| `python main.py run --groups A_steps` | Đối chứng: A = 200, 400, 800, 1200 với cùng số bước cập nhật như A = 2000 |
| `python main.py run --groups B_small` | Đối chứng: 1 tầng ẩn, chỉ 200 mẫu train, B = 2…256, đủ số bước (quan sát quá khớp) |
| `python main.py run --groups C_full` | Đối chứng: như C nhưng tắt early stopping (vẫn checkpoint theo validation) |
| `python scripts/preliminary_tuning.py` | Khảo sát sơ bộ khởi tạo/chuẩn hóa/lr trên validation (Bảng 4 báo cáo) |
| `python main.py analyze` | Bảng tổng hợp, hình, chọn mô hình theo validation, đánh giá test |
| `python main.py run --groups A --seeds 11 22 33 44 55` | Chạy với seed tùy chọn |

Lệnh `run` ghi nối tiếp vào `results/runs.csv` và bỏ qua các lần chạy đã có, nên có thể chạy tiếp nếu bị dừng giữa chừng. Muốn chạy lại từ đầu thì xóa thư mục `results/`.

## Cấu hình chuẩn (baseline)

| Tham số | Giá trị |
|---|---|
| Dữ liệu | train pool 2000, validation 400, test 100; lấy mẫu đều; `DATA_SEED = 2026` |
| Kiến trúc | 2 → 32 → 32 → 1, tanh ở tầng ẩn, đầu ra tuyến tính (1 185 tham số) |
| Chuẩn hóa | z-score đầu vào theo mean/std của tập train trong từng lần chạy (lưu trong mô hình) |
| Hàm mất mát | `nn.MSELoss` |
| Tối ưu | Adam, lr 1e-2, CosineAnnealingLR về 1e-5, batch 64 |
| Dừng | tối đa 300 epoch; early stopping patience 50 theo validation MSE |
| Checkpoint | trạng thái có validation MSE nhỏ nhất |
| Seed mô hình | 11, 22, 33 (khởi tạo trọng số + thứ tự mini-batch) |

Mỗi nhóm thực nghiệm chỉ thay đổi đúng một yếu tố. Mô hình được chọn: **6 tầng ẩn × 32 nơ-ron, không early stopping**, test MSE = 5.42e-8 (xem `results/selection.json`). Tập test **không** dùng cho early stopping, checkpoint hay chọn cấu hình; mô hình cuối được chọn theo validation MSE trung bình 3 seed.

## Cấu trúc thư mục

```
BTL01/
├── main.py                  # CLI: data | run | analyze | all
├── requirements.txt
├── mlp_gauss/
│   ├── config.py            # hằng số bài toán, giá trị A/B/C, cấu hình baseline
│   ├── data.py              # hàm Gaussian, sinh/lưu/nạp CSV, DataLoader, thống kê
│   ├── model.py             # lớp MLP (nn.Module) tùy biến số tầng/nơ-ron
│   ├── train.py             # vòng lặp train/validation, early stopping, checkpoint, đánh giá
│   ├── metrics.py           # MSE, RMSE, MAE, sai số lớn nhất, R²
│   ├── experiments.py       # định nghĩa và chạy nhóm A, B, C, B_small; tổng hợp mean ± std
│   ├── analysis.py          # hình/bảng A-B-C, chọn mô hình, đánh giá cuối
├── scripts/preliminary_tuning.py   # khảo sát sơ bộ chọn baseline
│   └── plots.py             # hình dữ liệu và sơ đồ kiến trúc
├── data/                    # train.csv, val.csv, test.csv
└── results/
    ├── runs.csv             # mỗi dòng = 1 lần chạy (cấu hình + độ đo train/val/test + thời gian)
    ├── summary.csv          # trung bình ± độ lệch chuẩn theo (nhóm, giá trị)
    ├── data_stats.csv       # thống kê mô tả x1, x2, y
    ├── selection.json       # mô hình được chọn và độ đo cuối
    ├── test_predictions.csv # y thật, ŷ dự đoán trên tập test
    ├── tables/              # bảng A, B, C, B_small, xếp hạng chọn mô hình
    ├── figures/             # toàn bộ hình trong báo cáo
    ├── logs/                # loss theo epoch của từng lần chạy
    ├── checkpoints/         # trọng số .pt tốt nhất (theo validation) của từng lần chạy
    └── models/best_model.pt # mô hình được chọn
```

## Nạp mô hình đã huấn luyện

```python
import torch
from mlp_gauss.model import MLP

ck = torch.load("results/models/best_model.pt", map_location="cpu", weights_only=False)
model = MLP(hidden=tuple(ck["config"]["hidden"]), activation=ck["config"]["activation"])
model.load_state_dict(ck["state_dict"]); model.eval()
with torch.no_grad():
    print(model(torch.tensor([[0.0, 0.0]])))   # ≈ f(0, 0) = 0.2690
```

## Thành viên và đóng góp

| Họ và tên | MSSV | Nhiệm vụ | Đóng góp |
|---|---|---|---|
| Nguyễn Văn A (nhóm trưởng) | 2xxxxxxx | Thiết kế thực nghiệm, train/experiments, tổng hợp báo cáo | 25% |
| Trần Thị B | 2xxxxxxx | Sinh và khảo sát dữ liệu, mục 2 báo cáo | 25% |
| Lê Văn C | 2xxxxxxx | Mô hình MLP, nhóm B và C | 25% |
| Phạm Thị D | 2xxxxxxx | Nhóm A, đánh giá cuối, slide và demo | 25% |
| **Tổng** | | | **100%** |

## Tài liệu nhóm

[Báo cáo](../BTL01_Nhom10_BaoCao.docx) · [Slide](../BTL01_Nhom10_Slide.pptx)
