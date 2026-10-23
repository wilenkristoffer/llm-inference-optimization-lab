"""Stage 0 go/no-go: does torch inside the container see and use the GPU?"""
import time

import torch

print("torch", torch.__version__, "hip", torch.version.hip)
print("cuda.is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("NO GPU")

print("device", torch.cuda.get_device_name(0))
free, total = torch.cuda.mem_get_info()
print(f"vram free {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB")

# Correctness: GPU matmul must match CPU.
a = torch.randn(1024, 1024)
b = torch.randn(1024, 1024)
err = (torch.mm(a.cuda(), b.cuda()).cpu() - torch.mm(a, b)).abs().max().item()
print(f"matmul max abs err vs cpu {err:.2e}")

# Rough throughput, to compare against the same script run natively in WSL.
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(3):
    x @ x
torch.cuda.synchronize()
n = 50
t0 = time.perf_counter()
for _ in range(n):
    x @ x
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"fp16 4096 matmul {2 * 4096**3 * n / dt / 1e12:.1f} TFLOP/s")
