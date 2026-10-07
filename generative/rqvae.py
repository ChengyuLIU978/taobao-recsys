from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RQVAEOutput:
    reconstruction: torch.Tensor
    quantized: torch.Tensor
    code_ids: torch.Tensor
    residuals: list[torch.Tensor]
    reconstruction_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    total_loss: torch.Tensor


class ResidualQuantizer(nn.Module):
    def __init__(self, num_codebooks: int, codebook_size: int, embedding_dim: int):
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        scale = 1.0 / max(1, codebook_size)
        self.codebooks = nn.Parameter(
            torch.empty(num_codebooks, codebook_size, embedding_dim).uniform_(-scale, scale)
        )

    @staticmethod
    def nearest_code(residual: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        distances = (
            residual.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * residual @ codebook.t()
        )
        return distances.argmin(dim=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        residual = z
        quantized_parts: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for level in range(self.num_codebooks):
            residuals.append(residual)
            code_ids = self.nearest_code(residual, self.codebooks[level])
            q = F.embedding(code_ids, self.codebooks[level])
            ids.append(code_ids)
            quantized_parts.append(q)
            residual = residual - q.detach()
        quantized = torch.stack(quantized_parts, dim=0).sum(dim=0)
        return quantized, torch.stack(ids, dim=1), residuals


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_codebooks: int = 3,
        codebook_size: int = 128,
        beta: float = 0.25,
    ):
        super().__init__()
        self.beta = float(beta)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.quantizer = ResidualQuantizer(num_codebooks, codebook_size, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> RQVAEOutput:
        z = self.encoder(x)
        quantized, code_ids, residuals = self.quantizer(z)
        z_q_st = z + (quantized - z).detach()
        reconstruction = self.decoder(z_q_st)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        codebook_loss = torch.stack(
            [F.mse_loss(F.embedding(code_ids[:, i], self.quantizer.codebooks[i]), residuals[i].detach())
             for i in range(self.quantizer.num_codebooks)]
        ).sum()
        commitment_loss = torch.stack(
            [F.mse_loss(residuals[i], F.embedding(code_ids[:, i], self.quantizer.codebooks[i]).detach())
             for i in range(self.quantizer.num_codebooks)]
        ).sum()
        total_loss = reconstruction_loss + codebook_loss + self.beta * commitment_loss
        return RQVAEOutput(
            reconstruction=reconstruction,
            quantized=quantized,
            code_ids=code_ids,
            residuals=residuals,
            reconstruction_loss=reconstruction_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
            total_loss=total_loss,
        )

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        _, ids, _ = self.quantizer(z)
        return ids


def utilization_metrics(code_ids: torch.Tensor, codebook_size: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for level in range(code_ids.shape[1]):
        counts = torch.bincount(code_ids[:, level].cpu(), minlength=codebook_size).double()
        active = int((counts > 0).sum().item())
        probabilities = counts[counts > 0] / counts.sum().clamp_min(1)
        entropy = -(probabilities * probabilities.log()).sum()
        rows.append(
            {
                "level": int(level),
                "active_codes": active,
                "utilization": active / codebook_size,
                "perplexity": float(entropy.exp().item()),
                "most_used_code_share": float(counts.max().item() / counts.sum().clamp_min(1).item()),
            }
        )
    return rows
