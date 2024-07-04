import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from data.dataset import IRDataset, collate_fn
from model.student import StudentModel
from config import Config
from tqdm import tqdm

def calculate_metrics(scores, k=10):
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    position = (sorted_indices == 0).nonzero()[:, 1] + 1
    
    mrr = (1 / position).mean().item()
    recall = (position <= k).float().mean().item()
    
    return mrr, recall

def evaluate():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    val_datasets = []
    for dataset_name, dataset_info in Config.DATASETS.items():
        val_dataset = IRDataset(
            data_path=f"{dataset_info['path']}/val.pkl",
            corpus_path=f"{dataset_info['path']}/corpus.pkl",
            neg_path=None,
            tokenizer=tokenizer,
            is_train=False,
            is_symmetric=dataset_info['is_symmetric']
        )
        val_datasets.append(val_dataset)

    val_dataset = ConcatDataset(val_datasets)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    model = StudentModel(Config.MODEL_NAME)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(Config.DEVICE)
    model.eval()

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            queries = batch['queries']
            pos_docs = batch['pos_docs']

            query_inputs = tokenizer(queries, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(Config.DEVICE)
            pos_doc_inputs = tokenizer(pos_docs, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(Config.DEVICE)

            scores = model(query_inputs.input_ids, query_inputs.attention_mask,
                           pos_doc_inputs.input_ids, pos_doc_inputs.attention_mask)
            
            all_scores.append(scores)
            all_labels.append(torch.ones(scores.size(0), device=scores.device))

    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mrr, recall = calculate_metrics(all_scores)

    print(f"MRR@10: {mrr:.4f}")
    print(f"Recall@10: {recall:.4f}")

if __name__ == "__main__":
    evaluate()