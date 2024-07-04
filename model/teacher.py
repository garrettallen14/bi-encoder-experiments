import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config

class TeacherModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, queries, documents, is_symmetric):
        prompts = []
        for q, d, sym in zip(queries, documents, is_symmetric):
            if sym:
                prompt = f"#Q和#P将分别描述一种事件或问题，它们可能并无关系。仅使用此描述和您对世界的了解，判断#P是不是一个关于#Q中的事件绝对正确的句子，或者#P是不是绝对正确地描述了#Q的事件或问题，请回答是或不是，若您不确定，请回答不是。\n#Q：{q}\n#P：{d}\n回答："
            else:
                prompt = f"#Q将描述一个问题，#A将描述一个网络段落，它们可能并没有关系。仅依据这些描述和您对世界的了解，判断#A能不能正确地回答#Q中提出的问题，请回答能或不能。\n#Q：{q}\n#A：{d}\n回答："
            prompts.append(prompt)

        inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH, return_tensors='pt').to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.logits[:, -1, :]  # Last token logits

    def get_features(self, input_ids):
        return self.model.get_input_embeddings()(input_ids)