import subprocess
def get_best_opencl_platform():
    for i in range(3):
        script = f"import lightgbm as lgb, numpy as np; lgb.train({{'device_type':'gpu', 'gpu_platform_id':{i}, 'gpu_device_id':0, 'verbose':1}}, lgb.Dataset(np.random.rand(10,2), label=np.random.rand(10)), num_boost_round=1)"
        try:
            out = subprocess.check_output([r".\.venv\Scripts\python.exe", "-c", script], stderr=subprocess.STDOUT, text=True)
            if "NVIDIA" in out: return i
        except Exception as e:
            pass
    return None
print('Best platform:', get_best_opencl_platform())
