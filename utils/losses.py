import torch
import torch.nn.functional as F
from ..config import Config

def contrastive_imitation_loss(student_logits, teacher_scores, temperature=Config.STUDENT_TEMPERATURE):
    exp_student = torch.exp(student_logits / temperature)
    exp_teacher = torch.exp(teacher_scores / Config.TEACHER_TEMPERATURE)
    return -torch.mean(torch.log(exp_student / exp_teacher.sum()))

def rank_imitation_loss_ph(student_logits, teacher_logits):
    return 1 - F.cosine_similarity(student_logits, teacher_logits, dim=0)

def rank_imitation_loss_hi(student_logits, teacher_logits):
    diff = student_logits.unsqueeze(1) - student_logits.unsqueeze(0)
    teacher_diff = teacher_logits.unsqueeze(1) - teacher_logits.unsqueeze(0)
    return -torch.mean(torch.log(torch.sigmoid(diff)) * (teacher_diff > 0).float())

def feature_imitation_loss(student_features, teacher_features):
    student_sim = F.cosine_similarity(student_features.unsqueeze(1), student_features.unsqueeze(0), dim=-1)
    teacher_sim = F.cosine_similarity(teacher_features.unsqueeze(1), teacher_features.unsqueeze(0), dim=-1)
    return F.mse_loss(student_sim, teacher_sim)