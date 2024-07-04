import torch
from torch.utils.data import Dataset
from utils.common_utils import load_pickle
from ..config import Config

class IRDataset(Dataset):
    def __init__(self, data_path, corpus_path, neg_path, tokenizer, max_length=320, is_train=True, is_symmetric=False):
        self.data = load_pickle(data_path)
        self.corpus = load_pickle(corpus_path)
        self.neg_samples = load_pickle(neg_path) if is_train else None
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.is_symmetric = is_symmetric

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, pos_doc_id = self.data[idx]
        pos_doc = self.corpus[pos_doc_id]
        
        if self.is_train:
            neg_doc_ids = self.neg_samples[query][:Config.NUM_NEGATIVES]
            neg_docs = [self.corpus[neg_id] for neg_id in neg_doc_ids]
            return query, pos_doc, neg_docs, self.is_symmetric
        return query, pos_doc, self.is_symmetric

def collate_fn(batch):
    if len(batch[0]) == 4:  # Training
        queries, pos_docs, neg_docs, is_symmetric = zip(*batch)
        neg_docs = [doc for sublist in neg_docs for doc in sublist]  # Flatten neg_docs
    else:  # Validation
        queries, pos_docs, is_symmetric = zip(*batch)
        neg_docs = []

    return {
        'queries': queries,
        'pos_docs': pos_docs,
        'neg_docs': neg_docs,
        'is_symmetric': is_symmetric
    }