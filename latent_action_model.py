#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_latent_action_model(model_path, device):
    model = LatentActionVQVAE(grayscale=True)
    checkpoint = torch.load(model_path, map_location=device)
    # Handle different checkpoint formats
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    # Remove '_orig_mod.' prefix if present
    fixed_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(fixed_state_dict, strict=False)
    step = checkpoint.get('step', 0)
    return model, step

class Encoder(nn.Module):
    """
    Encoder for VQ-VAE latent action model.
    - Input: Concatenated current and next frames (B, 2, 160, 210) for grayscale or (B, 6, 160, 210) for RGB.
    - Output: Latent feature map (B, 128, 5, 7)
    """
    def __init__(self, in_channels=2, hidden_dims=[64, 128, 256, 512, 512], out_dim=128):
        super().__init__()
        layers = []
        c_in = in_channels
        for i, c_out in enumerate(hidden_dims):
            if i < 4:
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=(4,7), stride=2, padding=(1,3)))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.project = nn.Conv2d(hidden_dims[-1], out_dim, kernel_size=1)
    def forward(self, x):
        x = self.conv(x)
        x = self.project(x)
        return x  # (B, 128, 5, 7)

class VectorQuantizer(nn.Module):
    """
    Vector quantization layer for VQ-VAE.
    - Codebook size: 256
    - Embedding dim: 128
    """
    def __init__(self, num_embeddings=256, embedding_dim=128, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    def forward(self, z):
        z_flat = z.permute(0,2,3,1).contiguous().view(-1, self.embedding_dim)
        d = (z_flat.pow(2).sum(1, keepdim=True)
             - 2 * z_flat @ self.embeddings.weight.t()
             + self.embeddings.weight.pow(2).sum(1))
        encoding_indices = torch.argmin(d, dim=1)
        quantized = self.embeddings(encoding_indices).view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0,3,1,2).contiguous()
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)
        codebook_loss = F.mse_loss(quantized, z.detach())
        quantized = z + (quantized - z).detach()
        return quantized, encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]), commitment_loss, codebook_loss

class Decoder(nn.Module):
    """
    Decoder for VQ-VAE latent action model.
    - Input: Quantized latent (B, 128, 5, 7) and current frame (B, 1, 160, 210) or (B, 3, 160, 210)
    - Output: Reconstructed next frame (B, 1, 160, 210) or (B, 3, 160, 210)
    """
    def __init__(self, in_channels=128, cond_channels=1, hidden_dims=[512, 512, 256, 128, 64], out_channels=1):
        super().__init__()
        self.cond_conv = nn.Sequential(
            nn.Conv2d(cond_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Conv2d(in_channels+128, hidden_dims[0], kernel_size=1)
        up_layers = []
        c_in = hidden_dims[0]
        for c_out in hidden_dims[1:]:
            up_layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1))
            up_layers.append(nn.BatchNorm2d(c_out))
            up_layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        up_layers.append(nn.ConvTranspose2d(c_in, out_channels, kernel_size=4, stride=2, padding=1))
        self.up = nn.Sequential(*up_layers)
    def forward(self, z, cond):
        cond_feat = self.cond_conv(cond)
        cond_feat = F.interpolate(cond_feat, size=z.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([z, cond_feat], dim=1)
        x = self.fc(x)
        x = self.up(x)
        x = F.interpolate(x, size=(160, 210), mode='bilinear', align_corners=False)
        return x

class LatentActionVQVAE(nn.Module):
    """
    VQ-VAE model for latent action prediction.
    """
    def __init__(self, codebook_size=256, embedding_dim=128, commitment_cost=0.25, grayscale=True):
        super().__init__()
        in_channels = 2 if grayscale else 6
        cond_channels = 1 if grayscale else 3
        out_channels = 1 if grayscale else 3
        self.encoder = Encoder(in_channels=in_channels)
        self.vq = VectorQuantizer(num_embeddings=codebook_size, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = Decoder(cond_channels=cond_channels, out_channels=out_channels)

    def forward(self, frame_t, frame_tp1, return_latent=False):
        frame_t_permuted = frame_t.permute(0, 1, 3, 2)  # (B, C, 210, 160) -> (B, C, 160, 210)
        frame_tp1_permuted = frame_tp1.permute(0, 1, 3, 2)
        x = torch.cat([frame_t_permuted, frame_tp1_permuted], dim=1)
        z = self.encoder(x)
        quantized, indices, commitment_loss, codebook_loss = self.vq(z)
        recon_permuted = self.decoder(quantized, frame_t_permuted)
        recon = recon_permuted.permute(0, 1, 3, 2)  # (B, C, 160, 210) -> (B, C, 210, 160)
        if return_latent:
            return recon, indices, commitment_loss, codebook_loss, z
        return recon, indices, commitment_loss, codebook_loss

class ActionToLatentMLP(nn.Module):
    def __init__(self, input_dim=9, hidden1=512, hidden2=256, latent_dim=35, codebook_size=256, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, self.latent_dim, self.codebook_size)
        return out

    def sample_latents(self, logits, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)
        batch, latent_dim, codebook_size = probs.shape
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples

class ActionStateToLatentMLP(nn.Module):
    def __init__(self, action_dim=9, hidden1=512, hidden2=256, latent_dim=35, codebook_size=256, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 8, 128),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(action_dim + 128, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, latent_dim * codebook_size)
        )

    def forward(self, action, frames):
        frame_features = self.frame_encoder(frames)
        combined = torch.cat([action, frame_features], dim=1)
        out = self.net(combined)
        return out.view(-1, self.latent_dim, self.codebook_size)

    def sample_latents(self, logits, temperature=1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be > 0")
        probs = F.softmax(logits / temperature, dim=-1)
        batch, latent_dim, codebook_size = probs.shape
        samples = torch.multinomial(probs.view(-1, codebook_size), 1).view(batch, latent_dim)
        return samples
