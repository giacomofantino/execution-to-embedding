import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, RobertaConfig, RobertaModel
import os
from components import CommentDecoder
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attn_code_to_data = nn.MultiheadAttention(
            embed_dim=config.bi_hidden_size,
            num_heads=config.bi_num_attention_heads,
            dropout=config.v_attention_probs_dropout_prob,
            batch_first=True
        )
        self.cross_attn_data_to_code = nn.MultiheadAttention(
            embed_dim=config.bi_hidden_size,
            num_heads=config.bi_num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.norm_code = nn.LayerNorm(config.bi_hidden_size)
        self.norm_data = nn.LayerNorm(config.bi_hidden_size)

    def forward(self, code_emb, code_mask, data_emb, data_mask):
        code_key_padding = ~code_mask.bool()
        data_key_padding = ~data_mask.bool()
        
        attn_code, _ = self.cross_attn_code_to_data(
            query=code_emb, 
            key=data_emb, 
            value=data_emb, 
            key_padding_mask=data_key_padding
        )
        
        attn_data, _ = self.cross_attn_data_to_code(
            query=data_emb, 
            key=code_emb, 
            value=code_emb, 
            key_padding_mask=code_key_padding
        )            
        
        code_emb = self.norm_code(code_emb + attn_code)
        data_emb = self.norm_data(data_emb + attn_data)

        return code_emb, data_emb
    
class SimpleDiffEncoder(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        
        num_hidden_layers = 6
        hidden_size = 768
        num_attention_heads = 8
        intermediate_size = 3072
        max_position_embeddings = 514  # RoBERTa default is 514
        layer_norm_eps = 1e-5
        dropout = 0.1
        attention_dropout = 0.1

        self.tokenizer = tokenizer

        ## Config a new Roberta model from scratch
        config = RobertaConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=attention_dropout,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        self.model = RobertaModel(config)

        if self.model.embeddings.word_embeddings.num_embeddings != len(self.tokenizer):
            self.model.resize_token_embeddings(len(self.tokenizer))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
    def forward(
        self,
        input_ids,
        attention_mask,
        return_dict = False,
        output_hidden_states = False
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        return outputs

class MultimodalEncoder(nn.Module):
    def __init__(self, config, data_tokenizer):
        super().__init__()
        # Load pre-trained models
        self.code_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.data_encoder = SimpleDiffEncoder(config, data_tokenizer)
        
        # Co-attention layer
        self.co_attention_layers = nn.ModuleList([
            CoAttention(config) for _ in range(config.num_co_attention_layers)
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.bi_hidden_size, config.bi_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.bi_hidden_size)
        )
        
    def forward(self, code_ids, code_mask, data_ids, data_mask):
        # Encode code
        code_outputs = self.code_encoder(
            input_ids=code_ids,
            attention_mask=code_mask,
            return_dict=True
        )
        code_emb = code_outputs.last_hidden_state
        #code_emb = self.code_projection(code_emb)

        # Encode data diff
        data_outputs = self.data_encoder(
            input_ids=data_ids,
            attention_mask=data_mask,
            return_dict=True
        )
        data_emb = data_outputs.last_hidden_state
        #data_emb = self.data_projection(data_emb)
        
        # Apply co-attention
        for layer in self.co_attention_layers:
            code_emb, data_emb = layer(code_emb, code_mask, data_emb, data_mask)
        
        # Concatenate and fuse embeddings
        concat_emb = torch.cat([code_emb, data_emb], dim=1)
        fused_emb = self.fusion_layer(concat_emb)
        
        return fused_emb

class MultimodalCommentGenerator(nn.Module):
    def __init__(self, config, vocab_size, tokenizer_code, tokenizer_data, tokenizer_comment):
        super().__init__()
        self.config = config
        self.encoder = MultimodalEncoder(config, tokenizer_data)
        self.decoder = CommentDecoder(config, vocab_size)

        #resize because we added new tokens
        self.encoder.data_encoder.model.resize_token_embeddings(len(tokenizer_data.get_vocab()))

        self.tokenizer_code = tokenizer_code
        self.tokenizer_data = tokenizer_data
        self.tokenizer_comment = tokenizer_comment
        self.padding_token = tokenizer_comment.pad_token_id

        #later used for correlation loss
        no_diff_ids = tokenizer_data.convert_tokens_to_ids(tokenizer_data.tokenize("<no_diff>"))
        self.no_diff_ids = torch.tensor(no_diff_ids, device="cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, code_ids, code_mask, data_ids, data_mask, 
                tgt_ids):

        # Normal forward pass for fine-tuning or inference with actual data-diff
        fused_emb = self.encoder(code_ids, code_mask, data_ids, data_mask)
        fused_mask = torch.cat([code_mask, data_mask], dim=1)

        logits = self.decoder(tgt_ids, fused_emb, fused_mask, inference=False)
        return logits
    
    def train_step(self, batch, optimizer, device):
        optimizer.zero_grad()
        
        # Forward pass
        code_ids,code_mask,diff_ids,diff_mask,target_ids,target_mask = batch

        logits = self.forward(code_ids,code_mask,diff_ids,diff_mask,target_ids).contiguous()
        
        # Calculate loss
        # Reshape logits and target_ids for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
        input = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
        target = shift_labels.view(-1)[active_loss]

        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_token)
        generation_loss = loss_fct(input, target)

        generation_loss.backward()
        optimizer.step()
        return generation_loss.item()
    
    def train_epoch(self, train_dataloader, optimizer, device, epoch):
        """
        Train for one epoch
        """
        self.train()
        total_loss = 0
        
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Epochs")
        for batch in bar:
            # Move batch to device
            batch = tuple(t.to(device) for t in batch)
            
            # Training step
            loss = self.train_step(batch, optimizer, device)
            bar.set_description(f"epoch {epoch} loss {loss:.4f}")
            total_loss += loss
            
        return total_loss / len(train_dataloader)

    def generate_comment(self, batch):
        beam_size = 5
        temperature = 1.0
        max_length = 128
        repetition_penalty = 1.2
        
        bos_id = self.tokenizer_comment.cls_token_id
        eos_id = self.tokenizer_comment.sep_token_id

        code_ids, code_mask, data_ids, data_mask, _, _ = batch

        # Encode inputs
        fused_emb = self.encoder(code_ids, code_mask, data_ids, data_mask)
        
        #fused_mask = ((code_mask + data_mask) > 0).long()
        fused_mask = torch.cat([code_mask, data_mask], dim=1)

        device = code_ids.device

        batch_size = code_ids.size(0)
        all_generated_sequences = []
        for i in range(batch_size):
            code_e = fused_emb[i:i+1]
            code_m = fused_mask[i:i+1,:]

            sequences = [[bos_id] for _ in range(beam_size)]
            scores = torch.zeros(beam_size, device=device)
            ended = torch.zeros(beam_size, dtype=torch.bool, device=device)

            for _ in range(max_length):
                # Temporary list to store candidates for the next generation step
                input_tensor = torch.tensor(sequences, dtype=torch.long, device=device)
                logits = self.decoder(input_tensor, code_e.repeat(beam_size, 1, 1), code_m.repeat(beam_size, 1), inference=True)

                for i, seq in enumerate(sequences):
                    token_counts = Counter(seq)
                    for token_id, count in token_counts.items():
                        if count > 1:
                            logits[i, token_id] /= (repetition_penalty ** (count - 1))

                logits = logits / temperature
                
                for i in range(beam_size):
                    if ended[i]:
                        logits[i, :] = float('-inf')
                        logits[i, eos_id] = 0.0  # Only allow eos if already ended

                top_probs, top_ids = torch.softmax(logits, dim=-1).topk(beam_size, dim=-1)

                new_sequences = []
                new_scores = []
                new_ended = []

                # Expand each current sequence into `beam_size` new ones
                for i in range(beam_size):
                    for j in range(beam_size):
                        new_seq = sequences[i] + [top_ids[i, j].item()]
                        new_score = scores[i] + torch.log(top_probs[i, j])
                        new_sequences.append(new_seq)
                        new_scores.append(new_score)
                        new_ended.append(top_ids[i, j].item() == eos_id)

                # Keep top beam_size overall sequences
                new_scores = torch.stack(new_scores)
                topk = torch.topk(new_scores, beam_size)
                indices = topk.indices

                sequences = [new_sequences[i] for i in indices]
                scores = new_scores[indices]
                ended = torch.tensor([new_ended[i] for i in indices], device=device)

                if ended.all():
                    break

            all_generated_sequences.append(sequences[0])

        return [self.tokenizer_comment.decode(torch.tensor(p), skip_special_tokens=True) for p in all_generated_sequences]

    def fit(self, train_dataset, optimizer, device, num_epochs=10, val_dataset=None, batch_size=16, num_workers=4, 
            base_path=None, last_epoch=None):
    
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        validation_dataloader = None
        if val_dataset:
            print("Creating validation dataloader...")
            validation_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

        best_val_loss = float('inf')
        training_history = {
            'train_loss': [],
            'val_loss': [] if validation_dataloader else None
        }
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_dataloader, optimizer, device, epoch)
            print(f"  Training - Loss: {train_loss:.4f}")
            
            training_history['train_loss'].append(train_loss)
            
            # Validation phase
            if validation_dataloader:
                self.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in validation_dataloader:
                        batch = tuple(t.to(device) for t in batch)
                        code_ids, code_mask, data_ids, data_mask, target_ids, target_mask = batch
                        
                        # Forward pass
                        logits = self.forward(code_ids, code_mask, data_ids, data_mask, target_ids)
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = target_ids[..., 1:].contiguous()

                        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
                        input = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
                        target = shift_labels.view(-1)[active_loss]

                        # Calculate validation loss
                        loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_token)
                        loss = loss_fct(input, target)
                        val_loss += loss.item()
                
                val_loss = val_loss / len(validation_dataloader)
                training_history['val_loss'].append(val_loss)
                print(f"  Validation - Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if base_path:
                save_path = f"{base_path}_epoch{epoch+int(last_epoch)+1}.pt" if last_epoch is not None else f"{base_path}_epoch{epoch+1}.pt"

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, save_path)

                print(f"Model saved to {save_path}")
        
        return training_history