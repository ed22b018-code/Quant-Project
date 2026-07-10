import torch
import lightgbm as lgb
import numpy as np
import time
import subprocess
import sys

def get_best_opencl_platform():
    for i in range(3):
        script = f"import lightgbm as lgb, numpy as np; lgb.train({{'device_type':'gpu', 'gpu_platform_id':{i}, 'gpu_device_id':0, 'verbose':1}}, lgb.Dataset(np.random.rand(10,2), label=np.random.rand(10)), num_boost_round=1)"
        try:
            out = subprocess.check_output([sys.executable, "-c", script], stderr=subprocess.STDOUT, text=True)
            if "NVIDIA" in out: return i
        except Exception as e:
            pass
    return None

def main():
    print("=== PyTorch GPU Test ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available to PyTorch: {cuda_available}")
    if cuda_available:
        print(f"Device count: {torch.cuda.device_count()}")
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Name detected: {device_name}")
        
        # Tiny computation
        try:
            print("Running tiny computation on GPU...")
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            end = time.time()
            print(f"Computation successful! (Output sum: {z.sum().item():.2f})")
            print(f"Time taken: {end - start:.4f} seconds")
        except Exception as e:
            print(f"Failed to run computation on GPU: {e}")
    else:
        print("GPU NOT detected by PyTorch.")

    print("\n=== LightGBM GPU Test ===")
    plat_id = get_best_opencl_platform()
    if plat_id is not None:
        try:
            # Create a tiny dataset
            X = np.random.rand(100, 10)
            y = np.random.rand(100)
            train_data = lgb.Dataset(X, label=y)
            
            # Parameters for GPU
            params = {
                'objective': 'regression',
                'device_type': 'gpu',
                'gpu_platform_id': plat_id,
                'gpu_device_id': 0,
                'verbose': -1
            }
            
            print(f"Attempting to train tiny model on LightGBM with OpenCL (Platform {plat_id}, NVIDIA)...")
            gbm = lgb.train(params, train_data, num_boost_round=2)
            print("LightGBM successfully trained a model using NVIDIA GPU!")
        except Exception as e:
            print(f"LightGBM GPU test failed: {e}")
    else:
        print("LightGBM GPU test failed: Could not detect NVIDIA OpenCL platform.")

if __name__ == '__main__':
    main()
