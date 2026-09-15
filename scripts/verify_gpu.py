import sys
import time
import psutil
import torch

def run_gpu_verification():
    print("=" * 60)
    print("SYNCNET — GPU & CUDA RUNTIME DIAGNOSTIC")
    print("=" * 60)
    
    python_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    
    print(f"Python Version    : {python_ver}")
    print(f"PyTorch Version   : {torch_ver}")
    print(f"CUDA Available    : {cuda_available}")
    print(f"CUDA Build Version: {cuda_version}")
    
    # System RAM Diagnostic
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    ram_avail_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)
    print(f"System Total RAM  : {ram_gb} GB")
    print(f"System Avail RAM  : {ram_avail_gb} GB (Project Target Budget: ~16 GB)")
    
    if not cuda_available:
        print("\n[!] CRITICAL ERROR: CUDA is NOT available to PyTorch!")
        print("    Diagnose NVIDIA drivers, CUDA toolkit, and PyTorch CUDA wheel.")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = round(vram_bytes / (1024 ** 3), 2)
    
    print(f"GPU Device Count  : {device_count}")
    print(f"GPU Device Name   : {device_name}")
    print(f"Total VRAM Memory : {vram_gb} GB (Project Hardware Spec: RTX 5050 8GB VRAM)")
    
    # Perform GPU Tensor Computation
    print("\n--- Executing GPU Tensor Operation Test ---")
    start_time = time.time()
    
    device = torch.device("cuda:0")
    matrix_size = 2000
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    c = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    mem_allocated_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
    mem_reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
    
    print(f"Tensor Shape      : {c.shape}")
    print(f"Tensor Device     : {c.device}")
    print(f"Execution Time    : {elapsed * 1000:.2f} ms")
    print(f"VRAM Allocated    : {mem_allocated_mb} MB")
    print(f"VRAM Reserved     : {mem_reserved_mb} MB")
    
    print("\n[x] SUCCESS: PyTorch CUDA GPU tensor execution verified on RTX 5050!")
    print("=" * 60)

if __name__ == "__main__":
    run_gpu_verification()
