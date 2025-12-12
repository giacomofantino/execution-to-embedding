import torch
import torch.nn as nn

class ModelConfig:
    def __init__(self):
        # Encoder settings
        self.code_hidden_size = 768  # CodeBERT hidden size
        self.data_hidden_size = 768  # DistilBERT hidden size
        self.bi_hidden_size = 768    # Size after projection layers
        
        # Co-attention settings
        self.bi_num_attention_heads = 6
        self.attention_probs_dropout_prob = 0.1
        self.v_attention_probs_dropout_prob = 0.1
        self.num_co_attention_layers = 3
        
        # Decoder settings
        self.decoder_num_layers = 6
        self.decoder_nhead = 12
        self.decoder_dim_feedforward = 2048
        self.decoder_dropout = 0.1
        
        # Generation settings
        self.max_length = 128
        self.max_length_decoder = 514
        self.beam_size = 5
        
        # Training settings
        self.mlm_probability = 0.15

class CommentDecoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        # Token embeddings for decoder
        self.token_embedding = nn.Embedding(vocab_size, config.bi_hidden_size)
        #nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.position_embedding = nn.Embedding(config.max_length_decoder, config.bi_hidden_size)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.bi_hidden_size,
            nhead=config.decoder_nhead,
            dim_feedforward=config.decoder_dim_feedforward,
            dropout=config.decoder_dropout
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config.decoder_num_layers
        )

        # Output projection
        self.dense = nn.Linear(config.bi_hidden_size, config.bi_hidden_size)
        self.lm_head = nn.Linear(config.bi_hidden_size, vocab_size, bias=False)
        
        # Save config
        self.config = config
        self.vocab_size = vocab_size

        #tie the lm_head and token_embedding weights to ensure they use the same space
        self.token_embedding.weight = self.lm_head.weight
        
    def forward(self, tgt_ids, encoded_src, src_padding_mask, inference=False):
        # Get embeddings for target tokens
        seq_length = tgt_ids.size(1)
        positions = torch.arange(seq_length, device=tgt_ids.device).unsqueeze(0).expand(tgt_ids.size(0), -1)
        
        tgt_emb = self.token_embedding(tgt_ids) + self.position_embedding(positions)
        
        # Run through decoder
        attn_mask = self.generate_square_subsequent_mask(tgt_ids.shape[1]).to(tgt_ids.device)

        out = self.decoder(
            tgt=tgt_emb.permute([1,0,2]).contiguous(),
            memory=encoded_src.permute([1,0,2]).contiguous(),
            tgt_mask=attn_mask,
            memory_key_padding_mask=(1-src_padding_mask).bool() #need to negate the mask
        )
        # Apply final linear layer and activation
        out = torch.tanh(self.dense(out))

        if inference:
            # During inference, return only the last token's logits
            hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
            out = (self.lm_head(hidden_states)).detach()
            return out
        else:
            # During training, return logits for all time steps
            hidden_states = out.permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            return lm_logits
    
    def generate_square_subsequent_mask(self, sz):
        # Generate mask to prevent attending to future positions
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask