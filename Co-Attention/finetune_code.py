import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel
import os
from components import CommentDecoder
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CodeOnlyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Load pre-trained models
        self.code_encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        
    def forward(self, code_ids, code_mask):
        # Encode code
        code_outputs = self.code_encoder(
            input_ids=code_ids,
            attention_mask=code_mask,
            return_dict=True
        )
        code_emb = code_outputs.last_hidden_state
        #code_emb = self.code_projection(code_emb)
        return code_emb

class CodeFinetuneCommentGenerator(nn.Module):
    def __init__(self, config, vocab_size, tokenizer_code, tokenizer_comment):
        super().__init__()
        self.config = config
        self.encoder = CodeOnlyEncoder(config)
        self.decoder = CommentDecoder(config, vocab_size)
        
        self.tokenizer_code = tokenizer_code
        self.tokenizer_comment = tokenizer_comment
        self.padding_token = tokenizer_comment.pad_token_id
        
    def forward(self, code_ids, code_mask, tgt_ids):

        # Normal forward pass for fine-tuning or inference with actual data-diff
        code_emb = self.encoder(code_ids, code_mask)

        logits = self.decoder(tgt_ids, code_emb, code_mask, inference=False)
        return logits
    
    def train_step(self, batch, optimizer, device):
        """
        Perform a single training step
        """
        optimizer.zero_grad()
        
        # Forward pass
        code_ids, code_mask, _, _, target_ids, target_mask = batch

        logits = self.forward(code_ids,code_mask,target_ids).contiguous()
        
        # Calculate loss
        # Reshape logits and target_ids for loss calculation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()

        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
        input = shift_logits.view(-1, shift_logits.size(-1))[active_loss]
        target = shift_labels.view(-1)[active_loss]

        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.padding_token)
        loss = loss_fct(input, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
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

        code_ids, code_mask, _, _, _, _ = batch

        # Encode inputs
        code_emb = self.encoder(code_ids, code_mask)

        device = code_ids.device

        batch_size = code_ids.size(0)
        all_generated_sequences = []
        for i in range(batch_size):
            code_e = code_emb[i:i+1]
            code_m = code_mask[i:i+1,:]

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
                        code_ids, code_mask, _, _, target_ids, target_mask = batch
                        
                        # Forward pass
                        logits = self.forward(code_ids, code_mask, target_ids)
                        
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
                save_path = f"{base_path}_epoch{epoch+last_epoch+1}.pt" if last_epoch is not None else f"{base_path}_epoch{epoch+1}.pt"

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, save_path)

                print(f"Model saved to {save_path}")
        
        return training_history