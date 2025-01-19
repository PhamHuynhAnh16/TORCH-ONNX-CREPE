<div align="center">

# TORCH & ONNX CREPE

**Trình theo dõi cao độ đơn âm Crepe được triển khai bằng pytorch và onnx**

</div>

**Dự án được tạo ra nhầm phục vụ mục đích cho một dự án khác:V**
**Tôi đã loại bỏ một số thứ do dự án đó của tôi không sử dụng đến**

```python
import os, sys, torch, numpy as np
import soundfile as sf

os.path.append(os.getcwd())
from .core import predict

model = "full" # "full", "large", "medium", "small", "tiny"
device = "cpu"
hop_length = 160
f0_min = 50
f0_max = 1100
sample_rate = 16000
providers = get_providers() 
onnx = True # Dùng onnx thì để True và thêm provider còn không thì để False

x, _ = sf.read(...)

x = x.astype(np.float32)
x /= np.quantile(np.abs(x), 0.999)

audio = torch.unsqueeze(torch.from_numpy(x).to(device, copy=True), dim=0)
if audio.ndim == 2 and audio.shape[0] > 1: audio = torch.mean(audio, dim=0, keepdim=True).detach()

p_len = x.shape[0] // hop_length

source = np.array(predict(audio.detach(), sample_rate, hop_length, f0_min, f0_max, model, batch_size=hop_length * 2, device=device, pad=True, providers=providers, onnx=onnx).squeeze(0).cpu().float().numpy())

source[source < 0.001] = np.nan

f0 = np.nan_to_num(np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source))

def get_providers():
    import onnxruntime

    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers
```

**Cách dùng khác**

```python
import os, sys, torch, numpy as np
import soundfile as sf

os.path.append(os.getcwd())
from .core import predict, mean, median

model = "full" # "full", "large", "medium", "small", "tiny"
device = "cpu"
f0_min = 50
f0_max = 1100
sample_rate = 16000
providers = get_providers() 
onnx = True # Dùng onnx thì để True và thêm provider còn không thì để False

x, _ = sf.read(...)
        
f0, pd = predict(torch.tensor(np.copy(x))[None].float(), sample_rate, 160, f0_min, f0_max, model, batch_size=512, device=device, return_periodicity=True, providers=providers, onnx=onnx)
f0, pd = mean(f0, 3), median(pd, 3)
f0[pd < 0.1] = 0

f0 = f0[0].cpu().numpy()

def get_providers():
    import onnxruntime

    ort_providers = onnxruntime.get_available_providers()

    if "CUDAExecutionProvider" in ort_providers: providers = ["CUDAExecutionProvider"]
    elif "CoreMLExecutionProvider" in ort_providers: providers = ["CoreMLExecutionProvider"]
    else: providers = ["CPUExecutionProvider"]

    return providers
```

# Thứ này được dựng dựa trên các công cụ
- **[torchcrepe](https://github.com/maxrmorrison/torchcrepe) : mô hình "full" và "tiny" và khung chính cho dự án**
- **[onnxcrepe](https://github.com/yqzhishen/onnxcrepe) : các mô hình onnx và một số mã**
- **[crepe](https://github.com/marl/crepe) : các mô hình "large", "medium", "small"**
