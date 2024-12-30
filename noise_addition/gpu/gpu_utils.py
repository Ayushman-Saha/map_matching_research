import torch
import cupy as cp
import logging
import config
import numpy as np

logger = logging.getLogger(__name__)

class GPUManager:
    def __init__(self):
        self._setup_gpu()
        
    def _setup_gpu(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available. This script requires GPU support.")
        
        try:
            self.device = torch.device('cuda')
            self.gpu_name = torch.cuda.get_device_name(0)
            cp.cuda.Device(0).use()
            
            # Set memory limit
            memory_limit = int(torch.cuda.get_device_properties(0).total_memory * 
                             config.GPU_MEMORY_FRACTION)
            torch.cuda.set_per_process_memory_fraction(config.GPU_MEMORY_FRACTION)
            
            logger.info(f"Using GPU: {self.gpu_name}")
            logger.info(f"Memory limit set to {memory_limit / 1e9:.2f} GB")
            
        except Exception as e:
            logger.error(f"Error initializing GPU: {str(e)}")
            raise

    def to_gpu(self, data):
        """Transfer data to GPU"""
        try:
            if isinstance(data, np.ndarray):
                return cp.array(data)
            elif isinstance(data, torch.Tensor):
                return data.to(self.device)
            else:
                return cp.array(np.array(data))
        except Exception as e:
            logger.error(f"Error transferring data to GPU: {str(e)}")
            raise

    def to_cpu(self, data):
        """Transfer data back to CPU"""
        try:
            if isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
            elif isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            return data
        except Exception as e:
            logger.error(f"Error transferring data to CPU: {str(e)}")
            raise