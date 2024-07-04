import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from data.dataset import IRDataset, collate_fn
from model.teacher import TeacherModel
from model.student import StudentModel
from utils.common_utils import set_seed, mine_hard_negatives, load_pickle, write_pickle
from utils.losses import contrastive_imitation_loss, rank_imitation_loss_ph, rank_imitation_loss_hi, feature_imitation_loss
from config import Config
from tqdm import tqdm

def prepare_dataset(dataset_name, dataset_info):
    corpus = load_pickle(f"{dataset_info['path']}/corpus.pkl")
    train_data = load_pickle(f"{dataset_info['path']}/train.pkl")
    val_data = load_pickle(f"{dataset_info['path']}/val.pkl")

    hard_neg_path = f"{dataset_info['path']}/hard_negatives.pkl"
    try:
        hard_negatives = load_pickle(hard_neg_path)
    except FileNotFoundError:
        print(f"Mining hard negatives for {dataset_name}...")
        queries = {query: [pos_doc] for query, pos_doc in train_data}
        hard_negatives = mine_hard_negatives(queries, corpus, queries)
        write_pickle(hard_negatives, hard_neg_path)

    return corpus, train_data, val_data, hard_negatives

def train():
    set_seed(Config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_datasets = []
    val_datasets = []
    for dataset_name, dataset_info in Config.DATASETS.items():
        corpus, train_data, val_data, hard_negatives = prepare_dataset(dataset_name, dataset_info)
        
        train_dataset = IRDataset(
            data_path=f"{dataset_info['path']}/train.pkl",
            corpus_path=f"{dataset_info['path']}/corpus.pkl",
            neg_path=f"{dataset_info['path']}/hard_negatives.pkl",
            tokenizer=tokenizer,
            is_train=True,
            is_symmetric=dataset_info['is_symmetric']
        )
        val_dataset = IRDataset(
            data_path=f"{dataset_info['path']}/val.pkl",
            corpus_path=f"{dataset_info['path']}/corpus.pkl",
            neg_path=f"{dataset_info['path']}/hard_negatives.pkl",
            tokenizer=tokenizer,
            is_train=False,
            is_symmetric=dataset_info['is_symmetric']
        )
        
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_fn)

    teacher_model = TeacherModel(Config.MODEL_NAME).to(Config.DEVICE)
    student_model = StudentModel(Config.MODEL_NAME).to(Config.DEVICE)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()  # For mixed precision training
    
    for epoch in range(Config.NUM_EPOCHS):
        student_model.train()
        teacher_model.eval()
        
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}"):
            queries = batch['queries']
            pos_docs = batch['pos_docs']
            neg_docs = batch['neg_docs']
            is_symmetric = batch['is_symmetric']

            # Teacher predictions
            with torch.no_grad(), autocast(enabled=Config.MIXED_PRECISION):
                teacher_pos_scores = teacher_model(queries, pos_docs, is_symmetric)
                teacher_neg_scores = teacher_model([q for q in queries for _ in range(Config.NUM_NEGATIVES)],
                                                   neg_docs,
                                                   [sym for sym in is_symmetric for _ in range(Config.NUM_NEGATIVES)])

            # Student predictions
            with autocast(enabled=Config.MIXED_PRECISION):
                query_inputs = tokenizer(queries, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(Config.DEVICE)
                pos_doc_inputs = tokenizer(pos_docs, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(Config.DEVICE)
                neg_doc_inputs = tokenizer(neg_docs, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(Config.DEVICE)

                student_pos_scores = student_model(query_inputs.input_ids, query_inputs.attention_mask,
                                                   pos_doc_inputs.input_ids, pos_doc_inputs.attention_mask)
                student_neg_scores = student_model(query_inputs.input_ids.repeat_interleave(Config.NUM_NEGATIVES, dim=0),
                                                   query_inputs.attention_mask.repeat_interleave(Config.NUM_NEGATIVES, dim=0),
                                                   neg_doc_inputs.input_ids, neg_doc_inputs.attention_mask)

                # Calculate losses
                ci_loss = contrastive_imitation_loss(student_pos_scores, teacher_pos_scores)
                ri_ph_loss = rank_imitation_loss_ph(student_pos_scores, teacher_pos_scores)
                ri_hi_loss = rank_imitation_loss_hi(student_neg_scores, teacher_neg_scores)
                
                student_features = student_model.encode(query_inputs.input_ids, query_inputs.attention_mask)
                with torch.no_grad():
                    teacher_features = teacher_model.get_features(query_inputs.input_ids)
                fi_loss = feature_imitation_loss(student_features, teacher_features)

                loss = ci_loss + Config.ALPHA * ri_ph_loss + Config.BETA * ri_hi_loss + Config.GAMMA * fi_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

        # Validation
        student_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                queries = batch['queries']
                pos_docs = batch['pos_docs']
                is_symmetric = batch['is_symmetric']

                query_inputs = tokenizer(queries, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(Config.DEVICE)
                pos_doc_inputs = tokenizer(pos_docs, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(Config.DEVICE)

                student_scores = student_model(query_inputs.input_ids, query_inputs.attention_mask,
                                               pos_doc_inputs.input_ids, pos_doc_inputs.attention_mask)
                
                teacher_scores = teacher_model(queries, pos_docs, is_symmetric)
                
                val_loss += contrastive_imitation_loss(student_scores, teacher_scores).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    torch.save(student_model.state_dict(), Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()