import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"CUDA версия: {torch.version.cuda}")
print(f"Название GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU недоступен'}")
