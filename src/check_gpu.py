import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print the number of GPUs (should be 1 for a laptop)
print(torch.cuda.get_device_name(0))  # Should print RTX 2080/4090 or your actual GPU
