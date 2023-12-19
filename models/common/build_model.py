# Returns either Transformer or retnet
from torch import nn

import models.transformer.model as TransformerModel
from models.retnet.configuration_retnet import load_config_from_json
from models.retnet.modeling_retnet import RetNetForCausalLM


def build_model(config, in_vocab_size: int, out_vocab_size: int, pos_in_len: int, pos_out_len: int, pad_token_id: int, eos_token_id: int):
    # TODO: some things hereunder might be Transformer specific, and might need to be factorized when we implement retnet
    model_pms = config["attention_model_params"]
    att_model = config["attention_model"]

    if att_model == "transformer":
        embedding_type = config["embedding_technique"]
        # Create the embedding layers, differently depending on the embedding_technique
        if embedding_type != "meta_transformer":  # TODO: this will have to change if we decide to implement "embed_concat"
            # We will use one embedding for the concatenated tokens (the concat of the tokens will be
            # done afterwards/in Dataset). Note: caller must set src_vocab_size and target_vocab_size accordingly
            src_embed = TransformerModel.InputEmbeddings(model_pms.d_model, in_vocab_size)
            tgt_embed = TransformerModel.InputEmbeddings(model_pms.d_model, out_vocab_size)
        else:
            raise NotImplementedError

        src_pos = TransformerModel.PositionalEncoding(model_pms.d_model, pos_in_len, model_pms.dropout)
        tgt_pos = TransformerModel.PositionalEncoding(model_pms.d_model, pos_out_len, model_pms.dropout)

        # Create the positional encoding layers
        # "meta_transformer" and "embed_concat" if we implement it also adds pos. encodings after the embedding, so no worries

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(model_pms.T):
            encoder_self_attention_block = TransformerModel.MultiHeadAttentionBlock(model_pms.d_model, model_pms.H,
                                                                   model_pms.dropout)
            feed_forward_block = TransformerModel.FeedForwardBlock(model_pms.d_model, model_pms.d_ff, model_pms.dropout)
            encoder_block = TransformerModel.EncoderBlock(model_pms.d_model, encoder_self_attention_block, feed_forward_block,
                                         model_pms.dropout)
            encoder_blocks.append(encoder_block)

        # Create the decoder blocks
        decoder_blocks = []
        for _ in range(model_pms.T):
            decoder_self_attention_block = TransformerModel.MultiHeadAttentionBlock(model_pms.d_model, model_pms.H,
                                                                   model_pms.dropout)
            decoder_cross_attention_block = TransformerModel.MultiHeadAttentionBlock(model_pms.d_model, model_pms.H,
                                                                    model_pms.dropout)
            feed_forward_block = TransformerModel.FeedForwardBlock(model_pms.d_model, model_pms.d_ff, model_pms.dropout)
            decoder_block = TransformerModel.TransformerDecoderBlock(model_pms.d_model, decoder_self_attention_block,
                                                    decoder_cross_attention_block, feed_forward_block,
                                                    model_pms.dropout)
            decoder_blocks.append(decoder_block)

        # Create the encoder and decoder
        encoder = TransformerModel.TransformerEncoder(model_pms.d_model, nn.ModuleList(encoder_blocks))
        decoder = TransformerModel.TransformerDecoder(model_pms.d_model, nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer = TransformerModel.ProjectionLayer(model_pms.d_model, out_vocab_size)

        # Create the transformer
        model = TransformerModel.Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    else:
        conf = load_config_from_json(f"models/retnet/configs/retnet-{config['model_size']}/config.json")
        conf.vocab_size = in_vocab_size
        conf.decoder_embed_dim = model_pms.d_model
        conf.decoder_num_layers = model_pms.T
        conf.decoder_num_attention_heads = model_pms.H
        conf.decoder_retention_heads = model_pms.H
        conf.dropout = model_pms.dropout
        conf.pad_token_id = pad_token_id
        conf.eos_token_id = eos_token_id
        conf.resize_layer_dim = pos_out_len
        model = RetNetForCausalLM(conf)
        print(out_vocab_size, in_vocab_size, pos_out_len, pos_in_len)

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p, -6 ** 0.5, 6 ** 0.5)

    return model