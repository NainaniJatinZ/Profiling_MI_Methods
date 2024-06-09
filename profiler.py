import timeit
import memory_profiler
import psutil
import torch
import logging
from functools import wraps

class Profiler:
    def __init__(self, log_file='profiling_results.log'):
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    def time_profile(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            log_message = f"Time taken by {func.__name__}: {elapsed_time:.4f} seconds"
            logging.info(log_message)
            print(log_message)
            return result
        return wrapper

    def memory_profile(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
            mem_peak = memory_profiler.memory_usage((func, args, kwargs), interval=0.1, max_usage=True)
            mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

            if isinstance(mem_peak, list):
                mem_peak = mem_peak[0]

            log_message = (f"Memory used by {func.__name__}: {mem_after - mem_before:.2f} MB\n"
                           f"Peak memory usage during {func.__name__}: {mem_peak - mem_before:.2f} MB")
            logging.info(log_message)
            print(log_message)
            return func(*args, **kwargs)
        return wrapper

    def log_profile(self, func):
        @self.time_profile
        @self.memory_profile
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    def profile_stages(self, stages):
        profiled_stages = []
        for stage in stages:
            profiled_stages.append(self.log_profile(stage))
        return profiled_stages

    def available_devices(self):
        devices = {'cpu': 'cpu'}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_str = f'cuda:{i}'
                devices[device_name] = device_str

        if torch.backends.mps.is_available():
            devices['MPS'] = 'mps'

        return devices

# Example of how to use the Profiler class with stages
if __name__ == "__main__":
    profiler = Profiler()

    @profiler.log_profile
    def example_function():
        data = stage1()
        tensor = stage2(data)
        result = stage3(tensor)
        return result

    @profiler.time_profile
    @profiler.memory_profile
    def stage1():
        data = [i ** 2 for i in range(100000)]
        return data

    @profiler.time_profile
    @profiler.memory_profile
    def stage2(data):
        tensor = torch.tensor(data, device='cuda' if torch.cuda.is_available() else 'cpu')
        return tensor

    @profiler.time_profile
    @profiler.memory_profile
    def stage3(tensor):
        result = tensor.sum().item()
        return result

    example_function()
