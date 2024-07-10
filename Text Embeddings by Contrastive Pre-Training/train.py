import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from schedulefree import AdamWScheduleFree
import os

from data import load_datasets
from models import load_model, LinearProbe, contrastive_loss
from utils import setup_logging, get_device, configure_tokenizer
from config import NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, LOG_INTERVAL

from transformers import AutoTokenizer

logger = setup_logging()
device = get_device()

def save_checkpoint(model, optimizer, epoch, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Delete the previous checkpoint if it exists
    previous_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch-1}.pt')
    if os.path.exists(previous_checkpoint):
        os.remove(previous_checkpoint)
    
    # Save the new checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch
    return 0

def train():
    print('Loading model.')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = load_model(device)
    tokenizer, model = configure_tokenizer(tokenizer, model)

    print('Loading datasets.')
    msmarco_train_dataset, senteval_train_dataset, senteval_test_dataset = load_datasets(tokenizer)

    msmarco_train_loader = DataLoader(msmarco_train_dataset, batch_size=BATCH_SIZE, num_workers=32, pin_memory=True, shuffle=True)
    senteval_train_loader = DataLoader(senteval_train_dataset, batch_size=BATCH_SIZE, num_workers=32, pin_memory=True, shuffle=True)
    senteval_test_loader = DataLoader(senteval_test_dataset, batch_size=BATCH_SIZE, num_workers=32, pin_memory=True)

    optimizer = AdamWScheduleFree(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()  # Create a GradScaler for mixed precision training

    start_epoch = load_checkpoint(model, optimizer, 'checkpoints/checkpoint_latest.pt')

    print('Beginning Training.')
    for epoch in range(start_epoch, NUM_EPOCHS):
        if epoch % 2 == 0 and epoch != 0:
            train_senteval(model, senteval_train_loader, senteval_test_loader, epoch, scaler)
        train_msmarco(model, msmarco_train_loader, optimizer, epoch, scaler)
        
        save_checkpoint(model, optimizer, epoch)

def train_msmarco(model, dataloader, optimizer, epoch, scaler):
    model.train()
    print(f'Epoch: {epoch}, Training on MS MARCO')
    loss_avg = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for idx, batch in enumerate(progress_bar):
        query_inputs = {
            'input_ids': batch['query_input_ids'].to(device),
            'attention_mask': batch['query_attention_mask'].to(device),
            'output_hidden_states': True
        }
        answer_inputs = {
            'input_ids': batch['answer_input_ids'].to(device),
            'attention_mask': batch['answer_attention_mask'].to(device),
            'output_hidden_states': True
        }
        
        optimizer.zero_grad()
        
        with autocast():
            query_output = model(**query_inputs)
            answer_output = model(**answer_inputs)
            
            query_emb = get_last_token_embeddings(query_output, query_inputs['attention_mask'])
            answer_emb = get_last_token_embeddings(answer_output, answer_inputs['attention_mask'])
            
            loss = contrastive_loss(query_emb, answer_emb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters())
        scaler.step(optimizer)
        scaler.update()
    
        loss_avg += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if idx % LOG_INTERVAL == 0 and idx > 0:
            logger.info(f"[Epoch {epoch + 1}/{NUM_EPOCHS}], [Batch Index {idx}/{len(dataloader)}], MS MARCO Average Loss: {loss_avg / LOG_INTERVAL:.4f}")
            loss_avg = 0
    
    logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}, MS MARCO Final Average Loss: {loss_avg / (idx % LOG_INTERVAL + 1):.4f}")

def train_senteval(model, train_dataloader, test_dataloader, main_epoch, scaler, num_probe_epochs=5):
    for param in model.parameters():
        param.requires_grad = False
    
    probe = LinearProbe(model.config.hidden_size).to(device)
    probe_optimizer = AdamWScheduleFree(probe.parameters(), lr=LEARNING_RATE)
    
    print(f'Main Epoch: {main_epoch}, Training on SentEval SUBJ for {num_probe_epochs} probe epochs.')
    
    for probe_epoch in range(num_probe_epochs):
        probe.train()
        probe_loss_avg = 0
        progress_bar = tqdm(train_dataloader, desc=f"Main Epoch {main_epoch+1}/{NUM_EPOCHS}, Probe Epoch {probe_epoch+1}/{num_probe_epochs}")
        for idx, batch in enumerate(progress_bar):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'output_hidden_states': True
            }
            labels = batch['label'].to(device)
            
            probe_optimizer.zero_grad()
            
            with autocast():
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = get_last_token_embeddings(outputs, inputs['attention_mask'])
                
                logits = probe(embeddings.to(torch.float32))
                probe_loss = torch.nn.functional.cross_entropy(logits, labels)

            scaler.scale(probe_loss).backward()
            scaler.unscale_(probe_optimizer)
            torch.nn.utils.clip_grad_norm_(probe.parameters())
            scaler.step(probe_optimizer)
            scaler.update()
        
            probe_loss_avg += probe_loss.item()
            progress_bar.set_postfix({'loss': f'{probe_loss.item():.4f}'})
            
            if idx % LOG_INTERVAL == 0 and idx > 0:
                logger.info(f"Main Epoch {main_epoch + 1}/{NUM_EPOCHS}, Probe Epoch {probe_epoch + 1}/{num_probe_epochs}, [Batch Index {idx}/{len(train_dataloader)}], SentEval SUBJ Probe Training Loss: {probe_loss_avg / LOG_INTERVAL:.4f}")
                probe_loss_avg = 0
        
        logger.info(f"Main Epoch {main_epoch + 1}/{NUM_EPOCHS}, Probe Epoch {probe_epoch + 1}/{num_probe_epochs}, SentEval SUBJ Probe Training Loss: {probe_loss_avg / (idx % LOG_INTERVAL + 1):.4f}")
        
        evaluate_senteval(model, probe, test_dataloader, main_epoch, probe_epoch)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    del probe
    del probe_optimizer
    
    for param in model.parameters():
        param.requires_grad = True

def evaluate_senteval(model, probe, dataloader, main_epoch, probe_epoch):
    probe.eval()
    correct = 0
    total = 0
    print(f'Main Epoch: {main_epoch}, Probe Epoch: {probe_epoch}, Evaluating on SentEval SUBJ test set')
    progress_bar = tqdm(dataloader, desc=f"Main Epoch {main_epoch+1}/{NUM_EPOCHS}, Probe Epoch {probe_epoch+1}")
    for batch in progress_bar:
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'output_hidden_states': True
        }
        labels = batch['label'].to(device)
        
        with torch.no_grad(), autocast():
            outputs = model(**inputs)
            embeddings = get_last_token_embeddings(outputs, inputs['attention_mask'])
            
            logits = probe(embeddings.to(torch.float32))
            predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        accuracy = correct / total
        progress_bar.set_postfix({'accuracy': f'{accuracy:.4f}'})
    
    logger.info(f"Main Epoch {main_epoch + 1}/{NUM_EPOCHS}, Probe Epoch {probe_epoch + 1}, SentEval SUBJ Test Accuracy: {accuracy:.4f}")

def get_last_token_embeddings(model_output, attention_mask):
    # Get the last hidden state
    last_hidden_state = model_output.hidden_states[-1]
    
    # Find the position of the last non-padding token
    last_token_indices = attention_mask.sum(dim=1) - 1
    
    # Get the batch size
    batch_size = last_hidden_state.size(0)

    # Extract the embeddings of the last token for each sequence in the batch
    last_token_embeddings = last_hidden_state[torch.arange(batch_size), last_token_indices]
    
    return last_token_embeddings

if __name__ == "__main__":
    train()