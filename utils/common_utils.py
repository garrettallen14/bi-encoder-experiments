import random
import torch
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from ..config import Config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def write_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def mine_hard_negatives(queries, corpus, relevant_docs, K=100):
    tokenized_corpus = [doc.split() for doc in corpus.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bi_encoder = SentenceTransformer('distilbert-base-nli-mean-tokens')
    
    hard_negatives = {}
    for query, relevant_doc_ids in queries.items():
        bm25_scores = bm25.get_scores(query.split())
        top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:K]
        
        query_embedding = bi_encoder.encode([query], convert_to_tensor=True)
        doc_embeddings = bi_encoder.encode([corpus[i] for i in top_bm25], convert_to_tensor=True)
        similarity_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        
        hard_negs = []
        for idx, score in sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True):
            doc_id = top_bm25[idx]
            if doc_id not in relevant_doc_ids and len(hard_negs) < Config.NUM_NEGATIVES:
                hard_negs.append(doc_id)
        
        hard_negatives[query] = hard_negs
    
    return hard_negatives