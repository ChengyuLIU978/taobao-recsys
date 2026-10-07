from __future__ import annotations

import torch
import torch.nn as nn


class GenerativeRetriever(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        encoder_layers: int,
        decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_history_items: int,
        decoder_max_length: int = 6,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pad_token_id = int(pad_token_id)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.encoder_position = nn.Embedding(max_history_items, d_model)
        self.decoder_position = nn.Embedding(decoder_max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.scale = d_model ** 0.5

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)

    def encode(self, history_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history_tokens.ndim != 3 or history_tokens.shape[-1] != 6:
            raise ValueError("history_tokens must have shape [batch, history_items, 6]")
        event_padding = history_tokens[:, :, 0].eq(self.pad_token_id)
        embedded = self.token_embedding(history_tokens).sum(dim=2) / (6.0 ** 0.5)
        positions = torch.arange(history_tokens.shape[1], device=history_tokens.device)
        embedded = embedded * self.scale + self.encoder_position(positions).unsqueeze(0)
        memory = self.encoder(embedded, src_key_padding_mask=event_padding)
        return memory, event_padding

    def decode(self, memory: torch.Tensor, memory_padding: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(decoder_tokens.shape[1], device=decoder_tokens.device)
        embedded = self.token_embedding(decoder_tokens) * self.scale + self.decoder_position(positions).unsqueeze(0)
        decoder_padding = decoder_tokens.eq(self.pad_token_id)
        decoded = self.decoder(
            embedded, memory,
            tgt_mask=self.causal_mask(decoder_tokens.shape[1], decoder_tokens.device),
            tgt_key_padding_mask=decoder_padding,
            memory_key_padding_mask=memory_padding,
        )
        return self.output(decoded)

    def forward(self, history_tokens: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        memory, padding = self.encode(history_tokens)
        return self.decode(memory, padding, decoder_tokens)

