import torch
import logging

def setup_logging(filename='output.log'):
    logging.basicConfig(filename=filename, level=logging.INFO)
    return logging.getLogger(__name__)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_tokenizer(tokenizer, model):
    special_tokens_dict = {'additional_special_tokens': ['[SOS]', '[EOS]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model