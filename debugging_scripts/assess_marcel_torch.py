import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

# Create tensors
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
y = torch.tensor([4.0, 5.0, 6.0]).to(device)

# Perform GPU operations
z = x + y
print("Result tensor on", device, ":", z)

# Verify GPU computation
if device.type == "cuda":
    z_cpu = z.to("cpu")
    print("Result tensor on CPU:", z_cpu)
