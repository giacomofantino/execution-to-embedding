import torch
import torch.nn as nn
from tqdm import tqdm
from model_ft import MultimodalEncoder
import os
import math
from torch.utils.data import Subset
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        return self.layer_norm(self.activation(self.linear(x)))

class MultimodalPreTrainJoint(nn.Module):
    def __init__(self, config, vocab_size, tokenizer_code, tokenizer_data, tokenizer_comment):
        super().__init__()
        self.config = config

        # variable to test different type of encoders
        self.encoder = MultimodalEncoder(config, tokenizer_data)

        #resize because we added new tokens
        #self.encoder.data_encoder.resize_token_embeddings(len(tokenizer_data.get_vocab()))
        
        # Additional heads for pretraining tasks
        #self.modality_correlation_head = nn.Linear(config.bi_hidden_size, 2)  # Binary classification
        mlm_decoder_code = nn.Linear(config.bi_hidden_size, len(tokenizer_code.get_vocab()))
        mlm_decoder_data = nn.Linear(config.bi_hidden_size, len(tokenizer_data.get_vocab()))

        # for optimizing tie the mlm embeddings with corresponding encoder
        mlm_decoder_code.weight = self.encoder.code_encoder.embeddings.word_embeddings.weight
        #self.mlm_head_data.weight = self.encoder.data_encoder.embeddings.word_embeddings.weight
        mlm_decoder_data.weight = self.encoder.data_encoder.model.embeddings.word_embeddings.weight

        self.mlm_head_code = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(config.bi_hidden_size, config.bi_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.bi_hidden_size),
            mlm_decoder_code
        )

        self.mlm_head_data = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(config.bi_hidden_size, config.bi_hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.bi_hidden_size),
            mlm_decoder_data
        )

        OPT = ['<MODIFIED>', '<ADDED>', '<REMOVED>','<MODIFIEDCOL>']
        self.ops_ids = torch.tensor(tokenizer_data.convert_tokens_to_ids(OPT), dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.F = self.ops_ids.numel()

        self.edit_head = nn.Sequential(
            nn.Linear(config.bi_hidden_size, config.bi_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.bi_hidden_size, len(OPT))
        )

        self.loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=-100)

        self.tokenizer_code = tokenizer_code
        self.tokenizer_data = tokenizer_data
        self.mask_id = self.tokenizer_data.mask_token_id
        self.no_diff_token_id = tokenizer_data.convert_tokens_to_ids("<NO_DIFF>")

        
    def forward(self, code_ids, code_mask, data_ids, data_mask):
        # Normal forward pass for fine-tuning or inference with actual data-diff
        fused_emb = self.encoder(code_ids, code_mask, data_ids, data_mask)

        return fused_emb

    # as these parts are common for both modalities, let's re use it
    def mask_modality_tokens(self, ids, special_tokens, modality, mlm_probability=0.15):
        labels = ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability, device=ids.device)

        special_tokens_mask = [[
                val in special_tokens
                for val in sample
            ]
            for sample in labels.tolist() 
        ]

        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=ids.device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        if modality == 'data-diff':
            #higher prob for <op>
            is_op = (ids.unsqueeze(-1) == self.ops_ids.view(1, 1, self.F)).to(special_tokens_mask.device)
            is_op = is_op & (~special_tokens_mask).unsqueeze(-1)
            is_op = is_op.any(dim=-1)
            probability_matrix[is_op] = mlm_probability * 2

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # returns special_tokens_mask so that we can use it later to avoid masking special tokens
        return masked_indices, labels, special_tokens_mask

    def apply_mlm_strategy(self, ids, mask, labels, tokenizer, vocab_size, device):
        # labels: real id where masked, else -100
        labels[mask] = ids[mask]
        labels[~mask] = -100

        # 80 % → [MASK]
        replace_prob = torch.rand_like(ids.float())
        indices_replaced = (replace_prob < 0.8) & mask
        ids[indices_replaced] = tokenizer.mask_token_id

        # 10 % → random id
        indices_random  = (replace_prob >= 0.8) & (replace_prob < 0.9) & mask
        #indices_random  = (replace_prob >= 0.9) & mask
        random_words    = torch.randint(vocab_size, ids.shape, dtype=torch.long, device=device)
        ids[indices_random] = random_words[indices_random]

        # 10 % → no changes

        return ids, labels
    
    def mask_tokens(self, code_ids, diff_ids, mlm_probability_code, mlm_probability_data):
        SPECIAL_CODE = [self.tokenizer_code.cls_token_id, self.tokenizer_code.sep_token_id, self.tokenizer_code.pad_token_id, self.tokenizer_code.bos_token_id, self.tokenizer_code.eos_token_id]
        SPECIAL_DIFF = [self.tokenizer_data.cls_token_id, self.tokenizer_data.sep_token_id, self.tokenizer_data.pad_token_id, self.tokenizer_data.bos_token_id, self.tokenizer_data.eos_token_id]

        masked_code = code_ids.clone()
        masked_diff = diff_ids.clone()
        device = code_ids.device
        batch_size = code_ids.size(0)
        seq_len = code_ids.size(1) + diff_ids.size(1)

        masked_indices_code, code_labels, special_tokens_mask_code = self.mask_modality_tokens(
            code_ids, 
            SPECIAL_CODE,
            'code', 
            mlm_probability_code
        )
        masked_indices_diff, diff_labels, special_tokens_mask_diff = self.mask_modality_tokens(
            diff_ids, 
            SPECIAL_DIFF,
            'data-diff', 
            mlm_probability_data
        )

        # Ensure at least one token is masked per sequence
        no_mask = (masked_indices_code.sum(dim=1) == 0) & (masked_indices_diff.sum(dim=1) == 0)
        
        if no_mask.any():
            valid_candidates_code = ~special_tokens_mask_code
            valid_candidates_diff = ~special_tokens_mask_diff

            # considering code and diff as a single sequence, we can generate random numbers and mask the highest number in the valid candidates
            valid_candidates = torch.concat([valid_candidates_code, valid_candidates_diff], dim=1)

            fix_indices = no_mask.nonzero(as_tuple=False).squeeze(1)
            # select masked token using the maximum value of a random tensor
            rand = torch.rand((batch_size, seq_len), device=device)
            rand[~valid_candidates] = -1 # Ensure only non-special tokens are considered

            _, random_token_idx = rand.max(dim=1)

            # Apply fix only to sequences that had no mask
            masked_indices = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            masked_indices[fix_indices, random_token_idx[fix_indices]] = True

            masked_indices_code = masked_indices_code | masked_indices[:, :code_ids.size(1)]
            masked_indices_diff = masked_indices_diff | masked_indices[:, code_ids.size(1):]

        # Apply masking strategy
        masked_code, code_labels = self.apply_mlm_strategy(
            masked_code, 
            masked_indices_code, 
            code_labels, 
            self.tokenizer_code, 
            self.tokenizer_code.vocab_size, 
            device
        )
        masked_diff, diff_labels = self.apply_mlm_strategy(
            masked_diff, 
            masked_indices_diff, 
            diff_labels, 
            self.tokenizer_data, 
            self.tokenizer_data.vocab_size, 
            device
        )

        return masked_code, masked_diff, code_labels, diff_labels
    
    def build_edit_presence_labels(self, diff_ids, diff_mask):
        # compare every position of diff_ids with self.ops_ids
        is_op = (diff_ids.unsqueeze(-1) == self.ops_ids.view(1, 1, self.F))
        is_op = is_op & diff_mask.unsqueeze(-1).bool()
        return is_op.any(dim=1).long() # shape (batch_size, num_ops)


    def forward_coatt(self, code_ids, code_mask, diff_ids, diff_mask):
        code_outputs = self.encoder.code_encoder(
            input_ids=code_ids,
            attention_mask=code_mask,
            return_dict=True
        )
        code_emb = code_outputs.last_hidden_state

        data_outputs = self.encoder.data_encoder(
            input_ids=diff_ids,
            attention_mask=diff_mask,
            return_dict=True
        )
        data_emb = data_outputs.last_hidden_state
        
        # Apply co-attention
        for layer in self.encoder.co_attention_layers:
            code_emb, data_emb = layer(code_emb, code_mask, data_emb, diff_mask)
        
        return code_emb, data_emb

    def train_step(self, batch):
        """
        Perform a single training step
        """
        
        # Forward pass
        code_ids,code_mask,diff_ids,diff_mask,_,_ = batch

        strategy = random.random()

        if strategy < 0.5:
            # Focus on diff_ids
            code_ids_masked, diff_ids_masked, code_labels, diff_labels = self.mask_tokens(
                code_ids, 
                diff_ids, 
                mlm_probability_code = 0.2,
                mlm_probability_data = 0.05
            )
        else:
            #focus on code_ids
            code_ids_masked, diff_ids_masked, code_labels, diff_labels = self.mask_tokens(
                code_ids, 
                diff_ids, 
                mlm_probability_code = 0.05,
                mlm_probability_data = 0.35
            )

        # Use the code inside the forward of the encoder to control the flow of the masked tokens
        code_emb, data_emb = self.forward_coatt(code_ids_masked, code_mask, diff_ids_masked, diff_mask)
        
        # Now we can use code_emb and data_emb to compute the loss

        # compute loss weighting by the number of masked tokens
        masked_code_tokens = (code_labels != -100).sum().item()
        masked_diff_tokens = (diff_labels != -100).sum().item()
        total_masked_tokens = masked_code_tokens + masked_diff_tokens

        if masked_code_tokens > 0:
            code_logits = self.mlm_head_code(code_emb)
            code_loss = self.loss_fct_mlm(code_logits.view(-1, code_logits.size(-1)), code_labels.view(-1))
            code_loss = code_loss * (masked_code_tokens / total_masked_tokens)
        else:
            code_loss = 0.0
        
        if masked_diff_tokens > 0:
            diff_logits = self.mlm_head_data(data_emb)
            diff_loss = self.loss_fct_mlm(diff_logits.view(-1, diff_logits.size(-1)), diff_labels.view(-1))
            diff_loss = diff_loss * (masked_diff_tokens / total_masked_tokens)
        else:
            diff_loss = 0.0
        
        mlm_loss = code_loss + diff_loss

        ## let's also introduce here the edit loss, but now the diff ids ops are masked
        y = self.build_edit_presence_labels(diff_ids, diff_mask)

        #masking the <op> in diff_ids
        is_op = (diff_ids.unsqueeze(-1) == self.ops_ids.view(1, 1, self.F))
        is_op = (is_op & diff_mask.unsqueeze(-1).bool()).any(dim=-1)

        diff_ids_op_masked = diff_ids.clone()
        diff_ids_op_masked[is_op] = self.mask_id

        _, data_emb = self.forward_coatt(code_ids, code_mask, diff_ids_op_masked, diff_mask)

        # use masked mean pooling
        diff_mask_expanded = diff_mask.unsqueeze(-1).float()
        data_pooled = (data_emb * diff_mask_expanded).sum(dim=1) / diff_mask_expanded.sum(dim=1).clamp(min=1e-6)

        edit_logits = self.edit_head(data_pooled)
        edit_loss = nn.functional.binary_cross_entropy_with_logits(edit_logits, y.float())

        loss = mlm_loss + 0.5 * edit_loss
        return loss
    
    def train_epoch(self, train_dataloader, optimizer, device, epoch, batch_size):
        """
        Train for one epoch
        """
        self.train()
        optimizer.zero_grad()
        total_loss = 0
        total_batch_size = 0
        
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Epochs")
        for batch in bar:
            # Move batch to device
            batch = tuple(t.to(device) for t in batch)

            total_batch_size += batch_size
            
            # Training step
            loss = self.train_step(batch)
            bar.set_description(f"epoch {epoch} loss {loss:.4f}")

            if math.isnan(loss) or math.isinf(loss):
                print(f"error epoch {epoch} loss {loss:.4f}")
            
            loss = loss / 4 # since we compute the step after 4 epoch each loss is weighted
            loss.backward()

            total_loss += loss.item() * 4

            if total_batch_size == 16:
                total_batch_size = 0
                # Update model parameters
                optimizer.step()
                optimizer.zero_grad()                
        return total_loss / len(train_dataloader)

    def fit(self, train_dataset, optimizer, device, num_epochs=10, val_dataset=None, batch_size=16, num_workers=4, 
            base_path=None, last_epoch=None):

        ## test using a smaller batch size (4) and using gradient accumulation to get to 16
        batch_size = 4
        #print(train_dataset.tensors)
        all_diff_ids = train_dataset.tensors[2] #retrieve all_diff_ids tensor
        is_no_diff = (all_diff_ids == self.no_diff_token_id)
        is_no_diff = is_no_diff.any(dim=1)  # Check if any token in the sequence is <NO_DIFF>
        no_diff_indices = torch.nonzero(is_no_diff, as_tuple=False).view(-1)
        has_diff_indices = torch.nonzero(~is_no_diff, as_tuple=False).view(-1)
        N = len(train_dataset)
        print(f"Total samples: {N}")
        print(f"  no-diff samples : {len(no_diff_indices)} ({100*len(no_diff_indices)/N:.1f}%)")
        print(f"  has-diff samples: {len(has_diff_indices)} ({100*len(has_diff_indices)/N:.1f}%)")
        H = len(has_diff_indices)
        ratios = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        freeze_epoch = 0 #number of epochs where code encoder is frozen
        
        ## the main idea is that for each epoch the subset of token with no_diff is changed, so that the code encoder still sees the entire dataset
        ## but the no_diff samples avoid to bias the loss computation
        for epoch in range(num_epochs):
            ratio = ratios[epoch]
            if ratio == 0:
                r = 0.0
            elif ratio == 1:
                r = 1.0
            else:
                r = ratio / (1.0 - ratio)
            
            X_target = int(r * H)
            X_target = min(X_target, len(no_diff_indices))  # if no_diff ratio is already < dont remove
            perm = torch.randperm(len(no_diff_indices))
            chosen_no_diff = no_diff_indices[perm[:X_target]]

            # Combine and sort the final index list
            final_indices = torch.cat([has_diff_indices, chosen_no_diff])
            final_indices, _ = torch.sort(final_indices)

            train_dataset_new = Subset(train_dataset, final_indices)
            all_diff_ids_new = all_diff_ids[final_indices] #from the original diff_ids, check the subset
            count_no_diff = (all_diff_ids_new == self.no_diff_token_id)
            count_no_diff = count_no_diff.any(dim=1)

            print(f"New training size: {len(train_dataset_new)}")
            print(f"  new no-diff % = {100 * sum(count_no_diff) / len(train_dataset_new):.1f}%")

            #for some epochs no diff tokens and no code encoder updating
            if epoch < freeze_epoch:
                for param in self.encoder.code_encoder.parameters():
                    param.requires_grad = False
            else:
                for param in self.encoder.code_encoder.parameters():
                    param.requires_grad = True

        
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset_new,
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
        
            train_loss = self.train_epoch(train_dataloader, optimizer, device, epoch, batch_size)
            print(f"  Training - Loss: {train_loss:.4f}")
            
            training_history['train_loss'].append(train_loss)
            
            # Validation phase
            if validation_dataloader:
                self.eval()
                val_loss = 0
                val_loss_mlm = 0
                val_loss_gate = 0
                with torch.no_grad():
                    for batch in validation_dataloader:
                        batch = tuple(t.to(device) for t in batch)
                        loss = self.train_step(batch)

                        val_loss += loss.item()
                        #val_loss_align += loss_align.item()
                
                val_loss = val_loss / len(validation_dataloader)
                print(f"  Validation - Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if base_path:
                if isinstance(last_epoch, int):
                    #last epoch specified as int, must preserve the numbering
                    save_path = f"{base_path}_epoch{epoch+last_epoch+1}.pt"
                else:
                    save_path = f"{base_path}_epoch{epoch+1}.pt"

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, save_path)

                print(f"Model saved to {save_path}")
        
        return training_history