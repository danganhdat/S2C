# CAM-SAM Visualization System

Hệ thống visualization kết hợp Class Activation Mapping (CAM) và Segment Anything Model (SAM) để phân tích và phân đoạn ảnh.

## 📁 Cấu trúc thư mục

```
project/
├── main.py              # FastAPI backend
├── app.py               # Streamlit UI
├── model.py             # Model loader & utilities
├── requirements.txt     # Dependencies
├── README.md           # Documentation (file này)
└── checkpoints/        # Model weights (sẽ tạo tự động)
```

## 🚀 Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. (Tùy chọn) Cài đặt CUDA cho GPU

Nếu bạn có GPU NVIDIA và muốn tăng tốc:

```bash
# Kiểm tra CUDA version
nvidia-smi

# Cài đặt PyTorch với CUDA (ví dụ CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Sử dụng

### Bước 1: Khởi động Backend (FastAPI)

Mở terminal đầu tiên:

```bash
python main.py
```

Backend sẽ chạy tại: `http://localhost:8000`

Kiểm tra API docs: `http://localhost:8000/docs`

### Bước 2: Khởi động Frontend (Streamlit)

Mở terminal thứ hai:

```bash
streamlit run app.py
```

UI sẽ mở tại: `http://localhost:8501`

### Bước 3: Sử dụng

1. Upload ảnh qua giao diện Streamlit
2. Click "Process Image"
3. Xem kết quả ở 3 cột:
   - **Ảnh gốc**: Ảnh đầu vào
   - **CAM**: Class Activation Map (vùng quan trọng)
   - **SAM**: Segmentation (đang phát triển)

## 🔧 API Endpoints

### POST `/process`

Xử lý ảnh và trả về CAM + SAM

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image)

**Response:**
```json
{
  "original": "base64_encoded_image",
  "cam": "base64_encoded_cam_image",
  "sam": "base64_encoded_sam_image",
  "predicted_class": 281,
  "message": "Processing status"
}
```

## 📊 Tính năng hiện tại

- ✅ Upload và hiển thị ảnh
- ✅ Class Activation Mapping (CAM) với ResNet50
- ✅ Overlay CAM lên ảnh gốc
- ✅ Download kết quả
- ⏳ SAM segmentation (đang phát triển)

## 🔮 Tính năng sắp tới

- [ ] Tích hợp SAM model (vit-h)
- [ ] Sử dụng CAM làm prompt cho SAM
- [ ] Hỗ trợ nhiều kiến trúc ResNet (34, 101)
- [ ] Batch processing
- [ ] Export masks

## 🐛 Troubleshooting

### Lỗi: Cannot connect to backend

**Nguyên nhân:** Backend chưa chạy

**Giải pháp:**
```bash
python main.py
```

### Lỗi: CUDA out of memory

**Nguyên nhân:** GPU không đủ memory

**Giải pháp:** Sử dụng CPU
```python
# Trong model.py, force CPU
device = torch.device('cpu')
```

### Lỗi: Module not found

**Nguyên nhân:** Thiếu dependencies

**Giải pháp:**
```bash
pip install -r requirements.txt
```

## 📝 Notes

- Model ResNet50 sẽ tự động download lần đầu chạy (~100MB)
- SAM model sẽ cần download checkpoint riêng (~2.5GB cho vit-h)
- Khuyến nghị dùng GPU để xử lý nhanh hơn

## 🔗 Tham khảo

- CAM paper: https://arxiv.org/abs/1512.04150
- SAM paper: https://arxiv.org/abs/2304.02643
- Tutorial: https://zilliz.com/learn/class-activation-mapping-CAM

## 📧 Liên hệ

Nếu có vấn đề, vui lòng tạo issue hoặc liên hệ developer.