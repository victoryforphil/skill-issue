"""Depth Anything 3 model wrapper."""

import torch
import numpy as np
from typing import Optional
from pathlib import Path


class DepthAnything3:
    """Wrapper for Depth Anything 3 model."""
    
    MODELS = {
        "small": "depth-anything/DA3-SMALL",
        "base": "depth-anything/DA3-BASE", 
        "large": "depth-anything/DA3-LARGE",
        "giant": "depth-anything/DA3-GIANT",
        "nested": "depth-anything/DA3NESTED-GIANT-LARGE",  # Best quality with metric depth
    }
    
    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        """
        Initialize Depth Anything 3 model.
        
        Args:
            model_size: Model size - "small", "base", "large", "giant", or "nested"
            device: Device to run on. If None, will auto-detect CUDA/CPU
        """
        self.model_size = model_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.use_official = False
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model."""
        try:
            # Try to use the official Depth Anything 3 implementation
            from depth_anything_3.api import DepthAnything3 as DA3Official
            
            model_name = self.MODELS.get(self.model_size)
            if model_name is None:
                raise ValueError(f"Unknown model size: {self.model_size}")
            
            print(f"Loading Depth Anything 3 '{self.model_size}' model from {model_name}...")
            print("This may take a while on first run (downloading weights)...")
            
            # Load model using official API
            self.model = DA3Official.from_pretrained(model_name)
            self.model = self.model.to(device=self.device)
            self.use_official = True
            
            print(f"✅ Loaded Depth Anything 3 {self.model_size} model on {self.device}")
            
        except ImportError as e:
            print("⚠️  Depth Anything 3 package not found. Using placeholder implementation.")
            print("To use the real model, install from:")
            print("  git clone https://github.com/ByteDance-Seed/Depth-Anything-3")
            print("  cd Depth-Anything-3")
            print("  pip install -e .")
            print("")
            
            # Placeholder model for demonstration
            class PlaceholderModel(torch.nn.Module):
                def forward(self, x):
                    # Simple placeholder that returns inverse of grayscale as "depth"
                    # x shape: (B, C, H, W)
                    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
                    return 1.0 - gray
            
            self.model = PlaceholderModel().to(self.device).eval()
            self.use_official = False
            print(f"Using placeholder model on {self.device}")
            
        except Exception as e:
            print(f"⚠️  Error loading official model: {e}")
            print("Falling back to placeholder model...")
            
            class PlaceholderModel(torch.nn.Module):
                def forward(self, x):
                    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
                    return 1.0 - gray
            
            self.model = PlaceholderModel().to(self.device).eval()
            self.use_official = False
            print(f"Using placeholder model on {self.device}")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict depth for an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            
        Returns:
            Depth map as numpy array (H, W)
        """
        with torch.no_grad():
            if self.use_official:
                # Official Depth Anything 3 API
                # The API expects list of images or single image
                prediction = self.model.inference([image])
                # Extract depth from prediction
                depth = prediction.depth[0]  # Get first image's depth
            else:
                # Fallback for placeholder
                img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(self.device)
                depth = self.model(img_tensor)[0].cpu().numpy()
        
        return depth
    
    def predict_batch(self, images: list) -> list:
        """
        Predict depth for a batch of images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of depth maps
        """
        if self.use_official:
            with torch.no_grad():
                prediction = self.model.inference(images)
                return [prediction.depth[i] for i in range(len(images))]
        else:
            return [self.predict(img) for img in images]

