"""
Convolutional Autoencoder for anomaly detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import numpy as np
from tqdm import tqdm

from src.models.base import BaseAnomalyModel, ModelFactory
from src.core.logging import get_logger

logger = get_logger(__name__)

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, activation: str = 'relu',
                 use_batchnorm: bool = True):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvTransposeBlock(nn.Module):
    """Transposed convolutional block with BatchNorm and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, activation: str = 'relu',
                 use_batchnorm: bool = True):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Encoder(nn.Module):
    """Encoder network"""
    
    def __init__(self, input_dim: int = 3, latent_dim: int = 512):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1: 3 -> 32
            ConvBlock(input_dim, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 256 -> 128
            
            # Block 2: 32 -> 64
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            
            # Block 3: 64 -> 128
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Block 4: 128 -> latent_dim
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, latent_dim, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    """Decoder network"""
    
    def __init__(self, latent_dim: int = 512, output_dim: int = 3):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # Block 1: latent_dim -> 256
            ConvBlock(latent_dim, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32 -> 64
            
            # Block 2: 128 -> 128
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64 -> 128
            
            # Block 3: 64 -> 64
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 128 -> 256
            
            # Output block: 32 -> output_dim
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, output_dim, kernel_size=3, stride=1, padding=1, activation='sigmoid'),
        )
    
    def forward(self, x):
        return self.decoder(x)

class ConvAutoEncoder(BaseAnomalyModel):
    """Convolutional Autoencoder for anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 3)
        self.latent_dim = config.get('latent_dim', 512)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        
        # Build encoder and decoder
        self.encoder = Encoder(self.input_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized ConvAutoEncoder with latent_dim={self.latent_dim}")
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image"""
        return self.decoder(z)
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error as anomaly score"""
        self.eval()
        with torch.no_grad():
            reconstructed = self(x)
            # Compute MSE per sample
            mse = F.mse_loss(reconstructed, x, reduction='none')
            anomaly_scores = mse.view(mse.size(0), -1).mean(dim=1)
            return anomaly_scores
    
    def compute_reconstruction_error(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Compute reconstruction error"""
        reconstructed = self(x)
        return F.mse_loss(reconstructed, x, reduction=reduction)
    
    def fit(self, dataloader: torch.utils.data.DataLoader, 
            val_dataloader: Optional[torch.utils.data.DataLoader] = None,
            epochs: int = 50, patience: int = 10) -> Dict[str, Any]:
        """Train the autoencoder"""
        logger.info("Starting autoencoder training")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                
                optimizer.zero_grad()
                loss = self.compute_reconstruction_error(x)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_dataloader is not None:
                self.eval()
                val_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        else:
                            x = batch
                        
                        x = x.to(self.device)
                        loss = self.compute_reconstruction_error(x)
                        val_loss += loss.item()
                        num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                val_losses.append(avg_val_loss)
                
                scheduler.step(avg_val_loss)
                
                logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
                
                # Early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'Early stopping at epoch {epoch+1}')
                        break
            else:
                scheduler.step(avg_train_loss)
                logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
        
        self.is_trained = True
        logger.info("Training completed")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': best_loss,
            'epochs_trained': epoch + 1
        }

class VAE(BaseAnomalyModel):
    """Variational Autoencoder for anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 3)
        self.latent_dim = config.get('latent_dim', 512)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.beta = config.get('beta', 1.0)  # KL divergence weight
        
        # Build encoder
        self.encoder = Encoder(self.input_dim, self.latent_dim * 2)  # For mu and logvar
        
        # Build decoder
        self.decoder = Decoder(self.latent_dim, self.input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized VAE with latent_dim={self.latent_dim}")
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def encode(self, x: torch.Tensor):
        """Encode input to latent parameters"""
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor):
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error as anomaly score"""
        self.eval()
        with torch.no_grad():
            reconstructed, mu, logvar = self(x)
            # Compute reconstruction error
            recon_error = F.mse_loss(reconstructed, x, reduction='none')
            recon_error = recon_error.view(recon_error.size(0), -1).mean(dim=1)
            return recon_error
    
    def compute_loss(self, x: torch.Tensor):
        """Compute VAE loss (reconstruction + KL divergence)"""
        reconstructed, mu, logvar = self(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (x.size(0) * x.size(1) * x.size(2) * x.size(3))
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def fit(self, dataloader: torch.utils.data.DataLoader, 
            val_dataloader: Optional[torch.utils.data.DataLoader] = None,
            epochs: int = 50, patience: int = 10) -> Dict[str, Any]:
        """Train the VAE"""
        logger.info("Starting VAE training")
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
        
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                x = x.to(self.device)
                
                optimizer.zero_grad()
                total_loss, recon_loss, kl_loss = self.compute_loss(x)
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'recon': recon_loss.item(),
                    'kl': kl_loss.item()
                })
            
            avg_train_loss = train_loss / num_batches
            avg_train_recon_loss = train_recon_loss / num_batches
            avg_train_kl_loss = train_kl_loss / num_batches
            
            train_losses.append({
                'total': avg_train_loss,
                'reconstruction': avg_train_recon_loss,
                'kl': avg_train_kl_loss
            })
            
            # Validation phase
            if val_dataloader is not None:
                self.eval()
                val_loss = 0.0
                val_recon_loss = 0.0
                val_kl_loss = 0.0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        if isinstance(batch, (list, tuple)):
                            x = batch[0]
                        else:
                            x = batch
                        
                        x = x.to(self.device)
                        total_loss, recon_loss, kl_loss = self.compute_loss(x)
                        
                        val_loss += total_loss.item()
                        val_recon_loss += recon_loss.item()
                        val_kl_loss += kl_loss.item()
                        num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                avg_val_recon_loss = val_recon_loss / num_val_batches
                avg_val_kl_loss = val_kl_loss / num_val_batches
                
                val_losses.append({
                    'total': avg_val_loss,
                    'reconstruction': avg_val_recon_loss,
                    'kl': avg_val_kl_loss
                })
                
                scheduler.step(avg_val_loss)
                
                logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
                
                # Early stopping
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'Early stopping at epoch {epoch+1}')
                        break
            else:
                scheduler.step(avg_train_loss)
                logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
        
        self.is_trained = True
        logger.info("Training completed")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_loss': best_loss,
            'epochs_trained': epoch + 1
        }

# Register models
ModelFactory.register_model('autoencoder', ConvAutoEncoder)
ModelFactory.register_model('vae', VAE)