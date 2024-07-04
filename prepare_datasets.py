import os
import pickle
import random
from config import Config

def create_dummy_data(dataset_name, num_samples=1000, corpus_size=10000):
    corpus = {i: f"This is document {i} in the {dataset_name} corpus." for i in range(corpus_size)}
    
    train_data = [(f"Query {i} for {dataset_name}", random.randint(0, corpus_size-1)) for i in range(num_samples)]
    val_data = [(f"Val Query {i} for {dataset_name}", random.randint(0, corpus_size-1)) for i in range(num_samples//10)]
    
    hard_negatives = {query: [random.randint(0, corpus_size-1) for _ in range(8)] for query, _ in train_data}
    
    return corpus, train_data, val_data, hard_negatives

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def prepare_datasets():
    base_path = "/workspace/data"
    os.makedirs(base_path, exist_ok=True)
    
    for dataset_name, dataset_info in Config.DATASETS.items():
        print(f"Preparing {dataset_name} dataset...")
        dataset_path = os.path.join(base_path, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        corpus, train_data, val_data, hard_negatives = create_dummy_data(dataset_name)
        
        save_pickle(corpus, os.path.join(dataset_path, "corpus.pkl"))
        save_pickle(train_data, os.path.join(dataset_path, "train.pkl"))
        save_pickle(val_data, os.path.join(dataset_path, "val.pkl"))
        save_pickle(hard_negatives, os.path.join(dataset_path, "hard_negatives.pkl"))
        
        # Update the path in the Config
        Config.DATASETS[dataset_name]['path'] = dataset_path
    
    # Update the config file
    with open('config.py', 'r') as f:
        config_content = f.read()
    
    for dataset_name, dataset_info in Config.DATASETS.items():
        old_path = f"'path': 'path/to/{dataset_name}'"
        new_path = f"'path': '{dataset_info['path']}'"
        config_content = config_content.replace(old_path, new_path)
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    
    print("All datasets prepared and config updated.")

if __name__ == "__main__":
    prepare_datasets()