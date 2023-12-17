import torch
from torch import nn

from models.retnet.modeling_retnet import RetNetModel
from models.transformer.model import LayerNormalization, ResidualConnection, MultiHeadAttentionBlock, ProjectionLayer


class RetNetEncoderDecoder(nn.Module):

    def __init__(self, encoder: RetNetModel, decoder: RetNetModel, features,
                 cross_attention_block: MultiHeadAttentionBlock, dropout: float,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.cross_attention_block = cross_attention_block
        self.norm = LayerNormalization(features)
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        return self.encoder(src)[0]

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.decoder(tgt)[0]
        x = self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        return self.norm(x)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
