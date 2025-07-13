import torch

class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    FEATURE_LAYERS = [4, 5, 6]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

