import torch
import torch.nn as nn
from tqdm import tqdm
from model_ft import MultimodalEncoder
import os
import math
from torch.utils.data import Subset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class MultimodalPreTrain(nn.Module):
    def __init__(self, config, vocab_size, tokenizer_code, tokenizer_data, phase=None):
        super().__init__()
        self.config = config

        # variable to test different type of encoders
        self.encoder = MultimodalEncoder(config, tokenizer_data)
        
        # Additional heads for pretraining tasks
        # MLM head for data (diff) modality
        self.mlm_dropout = nn.Dropout(p=0.1)
        self.mlm_dense = nn.Linear(config.bi_hidden_size, config.bi_hidden_size)
        self.mlm_act = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(config.bi_hidden_size)

        # Projection to vocab, **tied** to input embeddings:
        self.mlm_decoder = nn.Linear(config.bi_hidden_size, vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.mlm_decoder.weight = self.encoder.data_encoder.model.get_input_embeddings().weight

        self.loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=-100)

        # head for edit loss (using simplediff)
        OPT = ['<MODIFIED>', '<ADDED>', '<REMOVED>','<MODIFIEDCOL>']
        self.ops_ids = torch.tensor(tokenizer_data.convert_tokens_to_ids(OPT), dtype=torch.long, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.edit_head = nn.Sequential(
            nn.Linear(config.bi_hidden_size, config.bi_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.bi_hidden_size, len(OPT))
        )

        # Head for BCE loss on samples with no diff
        self.no_head = nn.Sequential(
            nn.Linear(config.bi_hidden_size, config.bi_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.bi_hidden_size, 1)
        )

        self.tokenizer_code = tokenizer_code
        self.tokenizer_data = tokenizer_data

        self.no_diff_token_id = self.tokenizer_data.convert_tokens_to_ids("<NO_DIFF>")

        '''
        1: diff data embeddings
        2: Introduce no-diff samples
        '''
        self.phase = phase
        
    def forward(self, data_ids, data_mask):
        ## we will compute the MLM loss only on the diff tokens
        data_outputs = self.encoder.data_encoder(
            input_ids=data_ids,
            attention_mask=data_mask,
            return_dict=True
        )
        data_emb = data_outputs.last_hidden_state
        return data_emb
    
    def mask_diff_tokens(self, diff_ids, tokenizer, mlm_probability=0.1):
        batch_size, seq_len = diff_ids.shape
        labels = diff_ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)

        #token not to be masked
        SPECIAL_DIFF = [self.tokenizer_data.cls_token_id, self.tokenizer_data.sep_token_id, self.tokenizer_data.pad_token_id]

        special_tokens_mask = [[
                val in SPECIAL_DIFF
                for val in sample
            ]
            for sample in labels.tolist() 
        ]

        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # instead of simply masking 15% of the tokens, for <op> the probability is 30%
        F = self.ops_ids.numel()
        is_op = (diff_ids.unsqueeze(-1) == self.ops_ids.view(1, 1, F)).to(special_tokens_mask.device)
        is_op = is_op & (~special_tokens_mask).unsqueeze(-1)
        is_op = is_op.any(dim=-1)
        probability_matrix[is_op] = 0.3

        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Ensure at least one token is masked per sequence
        no_mask = masked_indices.sum(dim=1) == 0

        if no_mask.any():
            valid_candidates = ~special_tokens_mask

            fix_indices = no_mask.nonzero(as_tuple=False).squeeze(1)
            # select masked token using the maximum value of a random tensor
            rand = torch.rand((batch_size, seq_len), device=masked_indices.device)
            rand[~valid_candidates] = -1 # Ensure only non-special tokens are considered

            _, random_token_idx = rand.max(dim=1)

            # Apply fix only to sequences that had no mask
            masked_indices[fix_indices, random_token_idx[fix_indices]] = True

        labels[~masked_indices] = -100 # Ignore non-masked tokens in loss calculation

        # 80% masked, 10% random, 10% unchanged
        masked_diff_ids = diff_ids.clone()
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        masked_diff_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=diff_ids.device)
        masked_diff_ids[indices_random] = random_words[indices_random]

        return masked_diff_ids, labels
    
    def build_edit_presence_labels(self, diff_ids, diff_mask):
        # compare every position of diff_ids with self.ops_ids
        F = self.ops_ids.numel()
        is_op = (diff_ids.unsqueeze(-1) == self.ops_ids.view(1, 1, F))
        is_op = is_op & diff_mask.unsqueeze(-1).bool()
        return is_op.any(dim=1).long() # shape (batch_size, num_ops)

    def train_step(self, batch, device):
        """
        Perform a single training step
        """
        _, _, diff_ids,diff_mask,_,_ = batch

        
        # for this case we have to select a subset of samples as we dont want to include no_diff samples for mlm
        has_diff = (diff_ids != self.no_diff_token_id).all(dim=1)
        # get indices of positive samples
        pos_indices = has_diff.nonzero(as_tuple=False).squeeze(-1)

        if len(pos_indices) > 0:
            diff_ids_mlm = diff_ids[pos_indices]
            diff_mask_mlm = diff_mask[pos_indices]

            # MLM loss, mask some tokens in diff_ids
            diff_ids_masked, diff_labels = self.mask_diff_tokens(diff_ids_mlm, self.tokenizer_data)

            data_emb = self.forward(diff_ids_masked, diff_mask_mlm)
            
            # MLM head
            x = self.mlm_dropout(data_emb)
            x = self.mlm_dense(x)
            x = self.mlm_act(x)
            x = self.mlm_layer_norm(x)
            logits = self.mlm_decoder(x) + self.mlm_bias

            # Compute loss only on masked positions
            mlm_loss = self.loss_fct_mlm(
                logits.view(-1, logits.size(-1)),
                diff_labels.view(-1)
            )
        else:
            mlm_loss = torch.tensor(0.0, device=device)

        # Edit loss, from diff_ids check which operations were detected and use the edit head
        y = self.build_edit_presence_labels(diff_ids, diff_mask)

        data_emb = self.forward(diff_ids, diff_mask)

        # use masked mean pooling
        diff_mask_expanded = diff_mask.unsqueeze(-1).float()
        data_pooled = (data_emb * diff_mask_expanded).sum(dim=1) / diff_mask_expanded.sum(dim=1).clamp(min=1e-6)

        edit_logits = self.edit_head(data_pooled)
        edit_loss = nn.functional.binary_cross_entropy_with_logits(edit_logits, y.float())

        loss = mlm_loss + 0.5 * edit_loss
        
        if self.phase == 2:
            ## also add cross-entropy on NO_DIFF samples
            # positive samples are those with no <NO_DIFF> token
            no_diff_labels = (diff_ids == self.no_diff_token_id).any(dim=1)

            data_emb = self.forward(diff_ids, diff_mask)

            # Pooling: use masked mean
            data_mask_expanded = diff_mask.unsqueeze(-1).float()
            data_pooled = (data_emb * data_mask_expanded).sum(dim=1) / data_mask_expanded.sum(dim=1).clamp(min=1e-6)

            logits = self.no_head(data_pooled).squeeze(-1)
            num_pos = max(1, int(no_diff_labels.sum().item()))
            num_neg = len(no_diff_labels) - num_pos
            # weight for positive samples is the ratio of neg/pos samples
            # in case pos samples are few, their weight is higher
            pw = max(1.0, min(5.0, num_neg / num_pos)) # to avoid extreme weights, range between 1 and 5     
            bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
            no_diff_loss = bce(logits, no_diff_labels.float())

            loss = loss + 0.5 * no_diff_loss
        
        return loss
    
    def train_epoch(self, train_dataloader, optimizer, device, epoch, batch_size):
        """
        Train for one epoch
        """
        self.train()
        total_loss = 0
        optimizer.zero_grad()
        total_batch_size = 0
        
        bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Epochs")
        for batch in bar:
            # Move batch to device
            batch = tuple(t.to(device) for t in batch)

            total_batch_size += batch_size
            
            # Training step
            loss = self.train_step(batch, device)
            #loss, mlm_loss, edit_loss, no_diff_loss = self.train_step(batch, optimizer, device)

            # loss may be 0.0 if we skip the batch
            if loss is None:
                continue
            
            bar.set_description(f"epoch {epoch} loss {loss.item():.4f}")#, mlm {mlm_loss:.4f}, edit {edit_loss:.4f}, no_diff {no_diff_loss:.4f}")

            loss = loss / 2
            loss.backward()

            total_loss += loss.item()*2

            if total_batch_size == batch_size * 2:
                total_batch_size = 0
                
                optimizer.step()
                optimizer.zero_grad()

            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f"epoch {epoch} loss {loss.item():.4f}")

        return total_loss / len(train_dataloader)

    def fit(self, train_dataset, optimizer, device, num_epochs=10, val_dataset=None, batch_size=16, num_workers=4, 
            base_path=None, last_epoch=None):
        
        if self.phase is None or self.phase not in [1, 2]:
            raise ValueError("Please specify the training phase (1 or 2) for pretraining the encoders")
        batch_size = 8

        for param in self.encoder.code_encoder.parameters():
            param.requires_grad = False

        all_diff_ids = train_dataset.tensors[2]
        is_no_diff = (all_diff_ids == self.no_diff_token_id)
        is_no_diff = is_no_diff.any(dim=1)  # Check if any token in the sequence is <NO_DIFF>
        no_diff_indices = torch.nonzero(is_no_diff, as_tuple=False).view(-1)
        has_diff_indices = torch.nonzero(~is_no_diff, as_tuple=False).view(-1)

        N = len(train_dataset)
        print(f"Total samples: {N}")
        print(f"  no-diff samples : {len(no_diff_indices)} ({100*len(no_diff_indices)/N:.1f}%)")
        print(f"  has-diff samples: {len(has_diff_indices)} ({100*len(has_diff_indices)/N:.1f}%)")
        H = len(has_diff_indices)

        if val_dataset:
            all_diff_ids_val = val_dataset.tensors[2]
            is_no_diff_val = (all_diff_ids_val == self.no_diff_token_id)
            is_no_diff_val = is_no_diff_val.any(dim=1)
            no_diff_indices_val = torch.nonzero(is_no_diff_val, as_tuple=False).view(-1)
            has_diff_indices_val = torch.nonzero(~is_no_diff_val, as_tuple=False).view(-1)
            H_val = len(has_diff_indices_val)

        if self.phase == 1:
            print("No no-diff samples in A0 phase")
            ratios = [0 for _ in range(10)]
        elif self.phase == 2:
            print("Mix some no-diff samples in A1 phase")
            #ratios = [0.2 for _ in range(10)]
            ratios = [0.2, 0.1, 0.05]
        
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
        
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset_new,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )

            validation_dataloader = None
            if val_dataset:
                #print("Creating validation dataloader...")
                # we have to apply the same logic to the validation set for the data diff subsampling

                #use same r as training
                X_target_val = int(r * H_val)
                X_target_val = min(X_target_val, len(no_diff_indices_val))  # if no_diff ratio is already < dont remove
                perm_val = torch.randperm(len(no_diff_indices_val))
                chosen_no_diff_val = no_diff_indices_val[perm_val[:X_target_val]]
                final_indices_val = torch.cat([has_diff_indices_val, chosen_no_diff_val])
                final_indices_val, _ = torch.sort(final_indices_val)
                val_dataset_new = Subset(val_dataset, final_indices_val)

                validation_dataloader = torch.utils.data.DataLoader(
                    val_dataset_new,
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
                with torch.no_grad():
                    for batch in validation_dataloader:
                        batch = tuple(t.to(device) for t in batch)
                        loss = self.train_step(batch, device)

                        if loss is None:
                            continue

                        if math.isnan(loss) or math.isinf(loss):
                            print(f"epoch {epoch} loss {loss:.4f}")# mlm_loss {mlm_loss:.4f}")

                        val_loss += loss.item()
                
                val_loss = val_loss / len(validation_dataloader)
                training_history['val_loss'].append(val_loss)
                print(f"  Validation - Loss: {val_loss:.4f}")#, mlm {mlm_loss:.4f}, edit {edit_loss:.4f}, no_diff {no_diff_loss:.4f}")
            
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