import torch

class Config:
    SEED = 42
    MODEL_NAME = "Qwen/Qwen-7B-Chat"
    DATASETS = {
        't2ranking': {'ratio': 0.5, 'path': 'path/to/t2ranking', 'is_symmetric': False},
        'dureader': {'ratio': 0.8, 'path': 'path/to/dureader', 'is_symmetric': False},
        'cmedqa2': {'ratio': 0.8, 'path': 'path/to/cmedqa2', 'is_symmetric': False},
        'mmarco': {'ratio': 0.35, 'path': 'path/to/mmarco', 'is_symmetric': False},
        'snli-zh': {'ratio': 1.0, 'path': 'path/to/snli-zh', 'is_symmetric': True},
        'sts': {'ratio': 1.0, 'path': 'path/to/sts', 'is_symmetric': True}
    }
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "/model.pth"
    MIXED_PRECISION = True
    ALPHA = 1.0  # Weight for RI_PH loss
    BETA = 0.3   # Weight for RI_HI loss
    GAMMA = 0.1  # Weight for FI loss
    MAX_SEQ_LENGTH = 320
    NUM_HEADS = 32  # For student model's PMA
    TEACHER_TEMPERATURE = 1.0
    STUDENT_TEMPERATURE = 1.0
    NUM_NEGATIVES = 8