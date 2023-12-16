import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)


class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output,
                                                                                 src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):

    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder, src_embeds: nn.ModuleList, # list  of embeddings
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeds = src_embeds
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src_list : torch.Tensor, src_mask: torch.Tensor):
        # src_list is of size (B, K, I); where K is the length of the "list"
        assert src_list.size(1) == len(self.src_embeds)
        all_embeddings = [self.src_embeds[i](src_list[:,i,:]) for i in range(len(self.src_embeds))]
        src = torch.hstack(all_embeddings)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (B, O', D)
        embeds = self.tgt_embed(tgt)
        pos = self.tgt_pos(embeds)
        return self.decoder(pos, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


# Returns either Transformer or RetNet
def build_model(config, in_vocab_size: int | list[int], out_vocab_size: int, pos_in_len: int | list[int], pos_out_len: int):
    # TODO: some things hereunder might be Transformer specific, and might need to be factorized when we implement RetNet
    model_pms = config["attention_model_params"]

    emb_type = config["embedding_technique"]
    # Create the embedding layers, differently depending on the embedding_technique
    if emb_type in ["tok_concat","onetext"]:
        # We will use one embedding for the concatenated tokens (the concat of the tokens will be
        # done afterwards/in Dataset). Note: caller must set src_vocab_size and target_vocab_size accordingly
        src_embeds = nn.ModuleList([InputEmbeddings(model_pms.d_model, in_vocab_size)])
    else:
        assert emb_type in ["embed_concat","meta_transformer"]
        assert type(in_vocab_size) == list and type(pos_in_len) == list
        if emb_type == "embed_concat":
            src_embeds = nn.ModuleList([InputEmbeddings(model_pms.d_model, voc_size) for voc_size in in_vocab_size])
        else:
            raise NotImplementedError

        # "meta_transformer" and "embed_concat" also adds pos. encodings after the embedding, so all good
        # (c.f. equation 7, section 3.3 of meta-transformer paper)
        pos_in_len = sum(pos_in_len)

    tgt_embed = InputEmbeddings(model_pms.d_model, out_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(model_pms.d_model, pos_in_len, model_pms.dropout)
    tgt_pos = PositionalEncoding(model_pms.d_model, pos_out_len, model_pms.dropout)

    att_model = config["attention_model"]
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(model_pms.T):
        if att_model == "transformer":
            encoder_self_attention_block = MultiHeadAttentionBlock(model_pms.d_model, model_pms.H, model_pms.dropout)
            feed_forward_block = FeedForwardBlock(model_pms.d_model, model_pms.d_ff, model_pms.dropout)
            encoder_block = EncoderBlock(model_pms.d_model, encoder_self_attention_block, feed_forward_block,
                                         model_pms.dropout)
        else:
            assert att_model == "retnet"
            raise NotImplementedError
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(model_pms.T):
        if att_model == "transformer":
            decoder_self_attention_block = MultiHeadAttentionBlock(model_pms.d_model, model_pms.H, model_pms.dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(model_pms.d_model, model_pms.H, model_pms.dropout)
            feed_forward_block = FeedForwardBlock(model_pms.d_model, model_pms.d_ff, model_pms.dropout)
            decoder_block = TransformerDecoderBlock(model_pms.d_model, decoder_self_attention_block,
                                                    decoder_cross_attention_block, feed_forward_block,
                                                    model_pms.dropout)
        else:
            raise NotImplementedError
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    if att_model == "transformer":
        encoder = TransformerEncoder(model_pms.d_model, nn.ModuleList(encoder_blocks))
        decoder = TransformerDecoder(model_pms.d_model, nn.ModuleList(decoder_blocks))
    else:
        raise NotImplementedError

    # Create the projection layer
    if att_model == "transformer":
        projection_layer = ProjectionLayer(model_pms.d_model, out_vocab_size)
    else:
        raise NotImplementedError

    # Create the transformer
    if att_model == "transformer":
        model = Transformer(encoder, decoder, src_embeds, tgt_embed, src_pos, tgt_pos, projection_layer)
    else:
        raise NotImplementedError

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p, -6 ** 0.5, 6 ** 0.5)

    return model
