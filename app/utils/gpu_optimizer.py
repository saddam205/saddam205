"""
gpu_optimizer.py
Part of the app/utils module.
GPU optimization utilities for machine learning models.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the best available device (CUDA GPU, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU detected)")
    
    return device


class GPUOptimizer:
    """
    GPU optimization utilities for model training and inference.
    Handles device management, memory optimization, and batch processing.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize GPU optimizer
        
        Args:
            device: Torch device (auto-detected if not provided)
        """
        self.device = device or get_device()
        self.is_cuda = self.device.type == "cuda"
        self.is_mps = self.device.type == "mps"
        
        # GPU properties
        self.gpu_name = None
        self.gpu_memory_total = None
        self.gpu_compute_capability = None
        
        if self.is_cuda:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.gpu_compute_capability = torch.cuda.get_device_capability(0)
            logger.info(f"GPU: {self.gpu_name} ({self.gpu_memory_total:.1f} GB)")
    
    def to_device(self, data: Any) -> Any:
        """
        Move data to the appropriate device
        
        Args:
            data: Tensor, numpy array, or list of tensors
        
        Returns:
            Data moved to device
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item) for item in data)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        else:
            return data
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy array (handles device transfer)
        
        Args:
            tensor: PyTorch tensor
        
        Returns:
            NumPy array
        """
        if tensor.is_cuda:
            return tensor.cpu().detach().numpy()
        return tensor.detach().numpy()
    
    def optimize_memory(self):
        """Clear GPU memory cache"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU memory cache cleared")
        elif self.is_mps:
            torch.mps.empty_cache()
    
    def get_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in GB"""
        if self.is_cuda:
            return torch.cuda.memory_allocated() / 1e9
        return None
    
    def get_memory_reserved(self) -> Optional[float]:
        """Get reserved GPU memory in GB"""
        if self.is_cuda:
            return torch.cuda.memory_reserved() / 1e9
        return None
    
    def batch_process(self, data: np.ndarray, batch_size: int, 
                      process_func: callable) -> np.ndarray:
        """
        Process data in batches for memory efficiency
        
        Args:
            data: Input data array
            batch_size: Size of each batch
            process_func: Function to process each batch
        
        Returns:
            Concatenated results
        """
        results = []
        n_samples = len(data)
        
        for i in range(0, n_samples, batch_size):
            batch = data[i:i + batch_size]
            batch_tensor = self.to_device(batch)
            result = process_func(batch_tensor)
            results.append(self.to_numpy(result))
            
            # Clear memory after each batch
            self.optimize_memory()
        
        return np.concatenate(results, axis=0) if results else np.array([])
    
    def get_optimal_batch_size(self, sample_size: int, 
                               memory_per_sample_mb: float = 10) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            sample_size: Size of one sample in MB
            memory_per_sample_mb: Memory per sample in MB
        
        Returns:
            Optimal batch size
        """
        if not self.is_cuda:
            return 32  # Default for CPU
        
        available_memory_mb = (self.gpu_memory_total * 0.8) * 1024  # 80% of total
        max_batch = int(available_memory_mb / memory_per_sample_mb)
        
        return max(1, min(max_batch, 256))
    
    def enable_amp(self) -> bool:
        """
        Enable Automatic Mixed Precision training
        
        Returns:
            Whether AMP is available
        """
        if self.is_cuda:
            try:
                from torch.cuda.amp import autocast, GradScaler
                logger.info("AMP (Automatic Mixed Precision) enabled")
                return True
            except ImportError:
                pass
        return False
    
    def profile_model(self, model: torch.nn.Module, input_shape: Tuple) -> dict:
        """
        Profile model performance on GPU
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
        
        Returns:
            Profiling results
        """
        model = model.to(self.device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        self.optimize_memory()
        
        # Measure inference time
        import time
        n_iterations = 100
        start_time = time.time()
        
        for _ in range(n_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if self.is_cuda:
            torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / n_iterations) * 1000
        
        # Get model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        
        return {
            'device': self.device.type,
            'avg_inference_time_ms': avg_time_ms,
            'inferences_per_second': 1000 / avg_time_ms,
            'model_size_mb': param_size,
            'gpu_memory_used_gb': self.get_memory_usage(),
            'gpu_name': self.gpu_name
        }


def optimize_for_gpu(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model for GPU inference
    
    Args:
        model: PyTorch model
    
    Returns:
        Optimized model
    """
    device = get_device()
    model = model.to(device)
    
    # Enable inference optimizations
    model.eval()
    
    if device.type == "cuda":
        # Use torch.jit for optimization
        try:
            model = torch.jit.script(model)
            logger.info("Model optimized with TorchScript")
        except:
            logger.warning("TorchScript optimization failed")
    
    logger.info(f"Model optimized for {device.type}")
    return model