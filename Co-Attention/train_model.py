from model_ft import MultimodalCommentGenerator
from model_pt import CodeOnlyCommentGenerator
from pretrain_diff import MultimodalPreTrain
from pretrain_encoders import MultimodalPreTrainJoint
from finetune_code import CodeFinetuneCommentGenerator
from components import ModelConfig
from transformers import AutoTokenizer, RobertaTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch
from tqdm import tqdm
import json
import argparse
import evaluate
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import bert_score
import code_bert_score

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 diff
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.diff = diff

def read_examples(filename, keep_no_diff=True, keep_no_comment=False):
    """Read examples from filename."""

    #print(f"Keep sample without comment = {keep_no_comment}")

    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())

            if 'diff_tokens' not in js or len(js['diff_tokens']) == 0 or js['diff_tokens'] == []:
                if keep_no_diff:
                    diff = '<NO_DIFF>'
                else:
                    #we skip this example
                    continue
            else:
                js['diff_tokens'] = [token for token in js['diff_tokens'] if isinstance(token, str)]
                diff = ' '.join(js['diff_tokens'])
                diff = diff.replace('\n',' ')
                diff=' '.join(diff.strip().split())
            
            if not keep_no_comment and len(js['docstring_tokens']) == 0:
                continue

            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())

            examples.append(
                Example(
                    idx = idx,
                    source=code,
                    target = nl,
                    diff = diff
                    ) 
            )
    
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
        example_id,
        source_ids,
        target_ids,
        diff_ids,
        diff_mask,
        source_mask,
        target_mask,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.diff_ids = diff_ids
        self.diff_mask = diff_mask
        self.source_mask = source_mask
        self.target_mask = target_mask       
        
def convert_examples_to_features(examples, tokenizer_code, tokenizer_diff,stage=None):
    features = []
    max_source_length = 256
    max_target_length = 128
    
    for example_index, example in tqdm(enumerate(examples),total=len(examples)):
        #source
        source_tokens = tokenizer_code.tokenize(example.source)[:max_source_length-2]
        source_tokens =[tokenizer_code.cls_token]+source_tokens+[tokenizer_code.sep_token]
        source_ids =  tokenizer_code.convert_tokens_to_ids(source_tokens) 
        source_mask = [1] * (len(source_tokens))
        padding_length = max_source_length - len(source_ids)
        source_ids+=[tokenizer_code.pad_token_id]*padding_length
        source_mask+=[0]*padding_length

        #diff (use fast tokenization)
        enc = tokenizer_diff(example.diff,
                            max_length=max_source_length,
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=True,
                            return_tensors=None)
        diff_ids = enc['input_ids']
        diff_mask = enc['attention_mask']

        #target
        target_tokens = tokenizer_code.tokenize(example.target)[:max_target_length-2]
        target_tokens = [tokenizer_code.cls_token]+target_tokens+[tokenizer_code.sep_token]            
        target_ids = tokenizer_code.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = max_target_length - len(target_ids)
        target_ids+=[tokenizer_code.pad_token_id]*padding_length
        target_mask+=[0]*padding_length
        
        features.append(
            InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 diff_ids,
                 diff_mask,
                 source_mask,
                 target_mask,
            )
        )
    return features

def transfer_weights_to_multimodal_model(pretrained_model, multimodal_model, pretrain_encoders=False):
    """
    Transfer weights from pretrained CodeOnlyCommentGenerator to MultimodalCommentGenerator
    """
    # Transfer code encoder weights
    multimodal_model.encoder.code_encoder.load_state_dict(
        pretrained_model.code_encoder.state_dict()
    )
    
    if not pretrain_encoders:
        # Transfer decoder weights
        multimodal_model.decoder.load_state_dict(
            pretrained_model.decoder.state_dict()
        )

        # in the pre trained model the lm_head of the decoder  and the position embedding was shared with the word embedding weight of the encdoder
        # since we are adding a new encoder with a new modality, we need to make sure the lm_head is not shared
        multimodal_model.decoder.lm_head.weight = torch.nn.Parameter(pretrained_model.decoder.lm_head.weight.clone())
        config = multimodal_model.decoder.config
        multimodal_model.decoder.position_embedding = torch.nn.Embedding(config.max_length_decoder, config.bi_hidden_size)
        multimodal_model.decoder.position_embedding.weight = torch.nn.Parameter(pretrained_model.decoder.position_embedding.weight.clone())

    # the code encoder will not be trained, thus we need to freeze its parameters
    for param in multimodal_model.encoder.code_encoder.parameters():
        param.requires_grad = False
        
    print("Successfully transferred weights from pretrained model to multimodal model")
    return multimodal_model

def train_model(model, code_tokenizer, data_tokenizer, device, pre_train_code=True, save_path=None, last_epoch=None, skip_diff=False, 
                pretrain_encoders=False, pretrain_diff=False):
    if pre_train_code:
        train_batch_size = 8
        lr = 5e-5
        weight_decay = 0.0
        epochs = 5
        adam_epsilon = 1e-8
    else:
        train_batch_size = 8
        lr = 5e-5
        weight_decay = 0.0
        epochs = 12
        adam_epsilon = 1e-8

    if pre_train_code:
        print('CodeXGlue dataset')
        train_filename = '../dataset/python/train.jsonl'
        dev_filename = '../dataset/python/valid.jsonl'
    elif pretrain_diff:
        print('Pretrain dataset for diff pretraining')
        train_filename = '../dataset/my_data/train_data.jsonl'
        dev_filename = '../dataset/my_data/val_data.jsonl'
    elif pretrain_encoders:
        print('Pretrain dataset for encoders pretraining')
        train_filename = '../dataset/my_data/train_data.jsonl'
        dev_filename = '../dataset/my_data/val_data.jsonl'
    else:
        print('My dataset')
        train_filename = '../dataset/my_data/train_data.jsonl'
        dev_filename = '../dataset/my_data/val_data.jsonl'

    #keep samples without comment when pretraining the encoders or the diff encoder
    keep_no_comment = (pretrain_encoders or pretrain_diff)

    train_examples = read_examples(train_filename, keep_no_diff= not skip_diff, keep_no_comment=keep_no_comment)
    train_features = convert_examples_to_features(train_examples, code_tokenizer, data_tokenizer,stage='train')
    all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
    all_diff_ids = torch.tensor([f.diff_ids for f in train_features], dtype=torch.long)
    all_diff_mask = torch.tensor([f.diff_mask for f in train_features], dtype=torch.float)
    all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_source_ids,all_source_mask,all_diff_ids, all_diff_mask, all_target_ids,all_target_mask)
    
    #eval data
    eval_examples = read_examples(dev_filename, keep_no_diff=not skip_diff, keep_no_comment=keep_no_comment)
    eval_features = convert_examples_to_features(eval_examples, code_tokenizer, data_tokenizer, stage='dev')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
    all_diff_ids = torch.tensor([f.diff_ids for f in eval_features], dtype=torch.long)
    all_diff_mask = torch.tensor([f.diff_mask for f in eval_features], dtype=torch.float)
    all_target_ids = torch.tensor([f.target_ids for f in eval_features], dtype=torch.long)
    all_target_mask = torch.tensor([f.target_mask for f in eval_features], dtype=torch.long)      
    eval_data = TensorDataset(all_source_ids,all_source_mask,all_diff_ids, all_diff_mask,all_target_ids,all_target_mask)   
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)

    #start training 
    return model.fit(train_data, optimizer, device, num_epochs=epochs, batch_size=train_batch_size, base_path=save_path, val_dataset=eval_data, last_epoch=last_epoch)

def test_model(model, code_tokenizer, data_tokenizer, device, skip_diff=False):
    test_filenames = ['../dataset/my_data/test_data_no_diff.jsonl', '../dataset/my_data/test_data_diff.jsonl']
    #test_filenames = ['../dataset/my_data/test_data_new.jsonl']
    
    eval_batch_size = 16
    model.eval() 
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    for i, test_filename in enumerate(test_filenames):
        test_examples = read_examples(test_filename,keep_no_diff=not skip_diff, keep_no_comment=False)
        
        test_examples_tok = convert_examples_to_features(test_examples, code_tokenizer, data_tokenizer,stage='test')
        all_source_ids = torch.tensor([f.source_ids for f in test_examples_tok], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in test_examples_tok], dtype=torch.long)
        all_diff_ids = torch.tensor([f.diff_ids for f in test_examples_tok], dtype=torch.long)
        all_diff_mask = torch.tensor([f.diff_mask for f in test_examples_tok], dtype=torch.float)
        all_target_ids = torch.tensor([f.target_ids for f in test_examples_tok], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in test_examples_tok], dtype=torch.long)
        test_data = TensorDataset(all_source_ids,all_source_mask,all_diff_ids, all_diff_mask, all_target_ids,all_target_mask)
        
        eval_sampler = SequentialSampler(test_data)
        eval_dataloader = DataLoader(test_data, sampler=eval_sampler, batch_size=eval_batch_size)

        sources = [sample.source for sample in test_examples]
        references = []
        predictions=[]

        for batch in tqdm(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)                 
            with torch.no_grad():
                res = model.generate_comment(batch)
                predictions = predictions + res
            
            _, _, _, _, target_ids, _ = batch
            labels = model.tokenizer_comment.batch_decode(target_ids, skip_special_tokens=True)
            references.extend(labels)
            #sources.extend(source_id)

        print(f'For test file {test_filename}:')

        # to make sure the metrics are calculated correctly, lower the case of the predictions and references
        predictions = [pred.lower() for pred in predictions]
        references = [ref.lower() for ref in references]

        bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
        rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(pred)) for pred, ref in zip(predictions, references)]
        P, R, F1 = bert_score.score(predictions, references, lang='en', verbose=False)
        P_code, R_code, F1_code, _ = code_bert_score.score(predictions, references, model_type='microsoft/codebert-base', verbose=False, sources=sources)
        print(f'BLEU: {bleu_score["bleu"]*100:.2f}%')
        print(f'ROUGE-1: {rouge_result["rouge1"]*100:.2f}%')
        print(f'ROUGE-2: {rouge_result["rouge2"]*100:.2f}%')
        print(f'ROUGE-L: {rouge_result["rougeL"]*100:.2f}%')
        print(f'METEOR: {sum(meteor_scores)/len(meteor_scores)*100:.2f}%')
        print(f'Roberta: F1 {F1.mean().item() * 100:.2f}%, P {P.mean().item() * 100:.2f}%, R {R.mean().item() * 100:.2f}%')
        print(f'CodeBERTScore: F1 {F1_code.mean().item() * 100:.2f}%, P {P_code.mean().item() * 100:.2f}%, R {R_code.mean().item() * 100:.2f}%')
        print("--------------------------------------------------")
    
        '''for j, pred in enumerate(predictions):
            print(f"Prediction {j}: {pred}")
        print("--------------------------------------------------")'''
    return

def main():
    parser = argparse.ArgumentParser(description="Train or test the comment generation model")
    #the user can choose which type of model to use (pretrain or finetune) and the task (train or test)
    parser.add_argument('--mode', type=str, choices=['pretrain_code', 'pretrain_diff', 'finetune', 'finetune_code', 'pretrain_encoders'], required=True, help="Specify the mode: pretrain or finetune")
    parser.add_argument('--task', type=str, choices=['train', 'test'], required=True, help="Specify the task: train or test")
    #in case of training a finetuned model, the user can specify to use a pre trained model from a specific epoch
    parser.add_argument('--retrieve_pretrain', type=int, default=None, help="Use a pre-trained model")
    parser.add_argument('--epoch', type=str, default=None, help="Specify the epoch number to retrieve the model from")
    parser.add_argument('--diff_epoch', type=str, default=None, help="Use a pre-trained model for diff pretraining from a specific epoch")
    parser.add_argument('--skip_diff', action='store_true', help="Use examples with no diff in pretraining")
    parser.add_argument('--encoders_epoch', type=str, default=None, help="Use a pre trained joint encoders for code and data")
    parser.add_argument('--phase', type=int, default=None, help="Training phase (1 or 2) for pretraining the data-diff encoder")
    args = parser.parse_args()

    checkpoint_pt_code = './model/pretrained/pytorch_model'
    checkpoint_pt_diff = './model/pretrained_diff/pytorch_model'
    checkpoint_pt_encoders = './model/pretrained_encoders/pytorch_model'
    checkpoint_ft = './model/finetuned/pytorch_model'
    checkpoint_ft_code = './model/finetuned_code/pytorch_model'

    # Initialize configuration
    config = ModelConfig()
    
    # Initialize tokenizers
    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    comment_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    data_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", use_fast=True)
    data_tokenizer.add_special_tokens({'additional_special_tokens': ['<NO_DIFF>', '<MODIFIED>', '<ADDED>', '<REMOVED>', '<TYPECHANGED>',
                                 '<NAFILLED>', '<MODIFIEDCOL>', '<RENAMED>', '<ELEMENT_ADDED>', '<ELEMENT_REMOVED>', '<PRINTED>']})

    if args.mode == 'pretrain_code':
        model = CodeOnlyCommentGenerator(config, comment_tokenizer.vocab_size, code_tokenizer, comment_tokenizer)
        save_path = checkpoint_pt_code
        
        if args.epoch:
            print(f"Loading checkpoint from epoch {args.epoch}")
            model.load_state_dict(torch.load(checkpoint_pt_code + f'_epoch{args.epoch}.pt')['model_state_dict'])
    elif args.mode == 'pretrain_diff':
        if args.phase is None or args.phase not in [1, 2]:
            raise ValueError("Please specify the training phase (1 or 2) for pretraining the encoder using --phase")
        model = MultimodalPreTrain(config, len(data_tokenizer), code_tokenizer, data_tokenizer, phase=args.phase)
        save_path = checkpoint_pt_diff

        if args.retrieve_pretrain:
            print("Loading pre-trained model weights for fine-tuning...")
            model_pt = CodeOnlyCommentGenerator(config, comment_tokenizer.vocab_size, code_tokenizer, comment_tokenizer)
            model_pt.load_state_dict(torch.load(checkpoint_pt_code + f'_epoch{args.retrieve_pretrain}.pt')['model_state_dict'])
            transfer_weights_to_multimodal_model(model_pt, model, pretrain_encoders=True)
        if args.epoch:
            print(f"Loading checkpoint from epoch {args.epoch}")
            model.load_state_dict(torch.load(checkpoint_pt_diff + f'_epoch{args.epoch}.pt')['model_state_dict'], strict=False)
    elif args.mode == "pretrain_encoders":
        model = MultimodalPreTrainJoint(config, comment_tokenizer.vocab_size, code_tokenizer, data_tokenizer, comment_tokenizer)
        save_path = checkpoint_pt_encoders

        if args.retrieve_pretrain:
            print("Loading pre-trained model weights for fine-tuning...")
            model_pt = CodeOnlyCommentGenerator(config, comment_tokenizer.vocab_size, code_tokenizer, comment_tokenizer)
            model_pt.load_state_dict(torch.load(checkpoint_pt_code + f'_epoch{args.retrieve_pretrain}.pt')['model_state_dict'])
            transfer_weights_to_multimodal_model(model_pt, model, pretrain_encoders=True)

            for param in model.encoder.code_encoder.parameters():
                param.requires_grad = False
        
        if args.diff_epoch:
            print(f"Loading pre-trained model weights for fine-tuning from diff pretraining epoch{args.diff_epoch}...")
            model_pt_diff = MultimodalPreTrain(config, len(data_tokenizer), code_tokenizer, data_tokenizer)
            model_pt_diff.load_state_dict(torch.load(checkpoint_pt_diff + f'_epoch{args.diff_epoch}.pt')['model_state_dict'], strict=False)

            model.encoder.data_encoder.load_state_dict(
                model_pt_diff.encoder.data_encoder.state_dict()
            )

            for param in model.encoder.data_encoder.parameters():
                param.requires_grad = True

            model.encoder.co_attention_layers.load_state_dict(
                model_pt_diff.encoder.co_attention_layers.state_dict()
            )
        
        if args.epoch:
            print(f"Loading checkpoint from epoch {args.epoch}")
            model.load_state_dict(torch.load(checkpoint_pt_encoders + f'_epoch{args.epoch}.pt')['model_state_dict'])
    elif args.mode == 'finetune':
        model = MultimodalCommentGenerator(config, comment_tokenizer.vocab_size, code_tokenizer, data_tokenizer,comment_tokenizer)
        save_path = checkpoint_ft

        if args.retrieve_pretrain:
            print("Loading pre-trained model weights for fine-tuning...")
            model_pt = CodeOnlyCommentGenerator(config, comment_tokenizer.vocab_size, code_tokenizer, comment_tokenizer)
            model_pt.load_state_dict(torch.load(checkpoint_pt_code + f'_epoch{args.retrieve_pretrain}.pt')['model_state_dict'])
            transfer_weights_to_multimodal_model(model_pt, model)
        
        if args.diff_epoch:
            print(f"Loading pre-trained model weights for fine-tuning from diff pretraining epoch{args.diff_epoch}...")
            model_pt_diff = MultimodalPreTrain(config, comment_tokenizer.vocab_size, code_tokenizer, data_tokenizer, comment_tokenizer)
            model_pt_diff.load_state_dict(torch.load(checkpoint_pt_diff + f'_epoch{args.diff_epoch}.pt')['model_state_dict'])

            model.encoder.data_encoder.load_state_dict(
                model_pt_diff.encoder.data_encoder.state_dict()
            )

            for param in model.encoder.data_encoder.parameters():
                param.requires_grad = True

            model.encoder.co_attention_layers.load_state_dict(
                model_pt_diff.encoder.co_attention_layers.state_dict()
            )
        
        if args.encoders_epoch:
            print(f"Loading pre-trained joint model weights for fine-tuning from encoder pretraining epoch{args.encoders_epoch}...")

            model_pt = MultimodalPreTrainJoint(config, comment_tokenizer.vocab_size, code_tokenizer, data_tokenizer, comment_tokenizer)
            state_dict = torch.load(checkpoint_pt_encoders + f'_epoch{args.encoders_epoch}.pt')['model_state_dict']
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("code_encoder_pt")}
            model_pt.load_state_dict(state_dict)

            model.encoder.code_encoder.load_state_dict(
                    model_pt.encoder.code_encoder.state_dict()
                )
            
            for param in model.encoder.code_encoder.parameters():
                param.requires_grad = False

            model.encoder.data_encoder.load_state_dict(
                model_pt.encoder.data_encoder.state_dict()
            )

            model.encoder.co_attention_layers.load_state_dict(
                model_pt.encoder.co_attention_layers.state_dict()
            )
        
        if args.epoch:
            print(f"Loading checkpoint from epoch {args.epoch}")
            model.load_state_dict(torch.load(checkpoint_ft + f'_epoch{args.epoch}.pt')['model_state_dict'], strict=False)
            for param in model.encoder.code_encoder.parameters():
                param.requires_grad = False
    elif args.mode == 'finetune_code':
        model = CodeFinetuneCommentGenerator(config, comment_tokenizer.vocab_size, code_tokenizer, comment_tokenizer)
        save_path = checkpoint_ft_code

        if args.retrieve_pretrain:
            print("Loading pre-trained model weights for fine-tuning...")
            model_pt = CodeOnlyCommentGenerator(config, comment_tokenizer.vocab_size, code_tokenizer, comment_tokenizer)
            model_pt.load_state_dict(torch.load(checkpoint_pt_code + f'_epoch{args.retrieve_pretrain}.pt')['model_state_dict'])
            transfer_weights_to_multimodal_model(model_pt, model)
            for param in model.encoder.code_encoder.parameters():
                param.requires_grad = True
        elif args.epoch:
            print(f"Loading checkpoint from epoch {args.epoch}")
            model.load_state_dict(torch.load(checkpoint_ft_code + f'_epoch{args.epoch}.pt')['model_state_dict'])

    if args.task == 'train':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        pre_train_code = args.mode == 'pretrain_code'
        pretrain_encoders = args.mode == 'pretrain_encoders'
        pretrain_diff = args.mode == 'pretrain_diff'
        
        training_history = train_model(model, code_tokenizer, data_tokenizer, device, pre_train_code=pre_train_code,save_path=save_path, last_epoch=args.epoch,
                                       skip_diff=args.skip_diff, pretrain_encoders=pretrain_encoders, pretrain_diff=pretrain_diff)
    
        print("Training History:")
        for i, loss in enumerate(training_history['train_loss']):
            print(f"Epoch {i+1} - Train Loss: {loss:.4f}", end='')
            if training_history['val_loss'] is not None:
                print(f", Validation Loss: {training_history['val_loss'][i]:.4f}")
            else:
                print()

    elif args.task == 'test':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print(f"Test for pytorch_model_epoch{args.epoch}:")
        test_model(model, code_tokenizer, data_tokenizer, device, skip_diff=args.skip_diff)

if __name__ == "__main__":
    main()