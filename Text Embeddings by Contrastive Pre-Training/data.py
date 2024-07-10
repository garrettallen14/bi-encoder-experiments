import torch
from torch.utils.data import Dataset
from datasets import load_dataset

class MSMARCODataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize with truncation, leaving room for EOS
        query = self.tokenizer(item['query'], truncation=True, max_length=self.max_length-1, add_special_tokens=False)
        answer = self.tokenizer(item['answer'], truncation=True, max_length=self.max_length-1, add_special_tokens=False)
        
        # Add EOS token
        query['input_ids'].append(self.tokenizer.eos_token_id)
        query['attention_mask'].append(1)
        answer['input_ids'].append(self.tokenizer.eos_token_id)
        answer['attention_mask'].append(1)
        
        # Pad sequences to max_length
        query_padding = [self.tokenizer.pad_token_id] * (self.max_length - len(query['input_ids']))
        answer_padding = [self.tokenizer.pad_token_id] * (self.max_length - len(answer['input_ids']))
        
        query['input_ids'] = query['input_ids'] + query_padding
        query['attention_mask'] = query['attention_mask'] + [0] * len(query_padding)
        answer['input_ids'] = answer['input_ids'] + answer_padding
        answer['attention_mask'] = answer['attention_mask'] + [0] * len(answer_padding)

        return {
            'query_input_ids': torch.tensor(query['input_ids']),
            'query_attention_mask': torch.tensor(query['attention_mask']),
            'answer_input_ids': torch.tensor(answer['input_ids']),
            'answer_attention_mask': torch.tensor(answer['attention_mask']),
        }

class SentEvalDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize with truncation, leaving room for EOS
        encoded = self.tokenizer(item['sentence'], truncation=True, max_length=self.max_length-1, add_special_tokens=False)
        
        # Add EOS token
        encoded['input_ids'].append(self.tokenizer.eos_token_id)
        encoded['attention_mask'].append(1)
        
        # Pad sequence to max_length
        padding_length = self.max_length - len(encoded['input_ids'])
        encoded['input_ids'] = encoded['input_ids'] + [self.tokenizer.pad_token_id] * padding_length
        encoded['attention_mask'] = encoded['attention_mask'] + [0] * padding_length

        return {
            'input_ids': torch.tensor(encoded['input_ids']),
            'attention_mask': torch.tensor(encoded['attention_mask']),
            'label': torch.tensor(item['label'])
        }

def load_datasets(tokenizer):
    msmarco = load_dataset("ms_marco", "v1.1")
    msmarco = msmarco.map(lambda example: {
        'query': example['query'],
        'answer': example['passages']['passage_text'][example['passages']['is_selected'].index(1) if 1 in example['passages']['is_selected'] else 0]
    }, remove_columns=msmarco['train'].column_names)
    
    senteval_mr = load_dataset("rahulsikder223/SentEval-SUBJ")
    
    msmarco_train_dataset = MSMARCODataset(msmarco['train'], tokenizer)
    senteval_train_dataset = SentEvalDataset(senteval_mr['train'], tokenizer)
    senteval_test_dataset = SentEvalDataset(senteval_mr['test'], tokenizer)
    
    return msmarco_train_dataset, senteval_train_dataset, senteval_test_dataset
