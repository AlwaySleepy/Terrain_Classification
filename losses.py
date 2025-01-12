import torch
import torch.nn.functional as F
from helpers import get_device


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def mse0_loss(y, alpha, epoch_num, num_classes, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)
    return loglikelihood

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss

def edl_mse0_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = exp_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse0_loss(target, alpha, epoch_num, num_classes, device=device)
    )
    return loss
# relu not used in experiment
'''
def edl_relu_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse0_loss(target, alpha, epoch_num, num_classes, device=device)
    )
    return loss
'''
def euc_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = exp_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    uncertainty = num_classes / S
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    _, target = torch.max(target,dim=1, keepdim=True)
    target = target.to(device)
    pred_cls = pred_cls.to(device)
    pred_scores = pred_scores.to(device)
    acc_match = torch.eq(pred_cls, target).float().to(device)
    eps = 10e-6
    acc_uncertain = -pred_scores * torch.log(1-uncertainty+eps)
    inacc_certain = -(1-pred_scores) * torch.log(uncertainty+eps)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    euc_loss = annealing_coef * acc_match * acc_uncertain + (1-annealing_coef) * (1-acc_match)*inacc_certain
    loss = torch.mean(euc_loss)
    return loss
'''
def euc_relu_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    uncertainty = num_classes / S
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    _, target = torch.max(target,dim=1, keepdim=True)
    target = target.to(device)
    pred_cls = pred_cls.to(device)
    pred_scores = pred_scores.to(device)
    acc_match = torch.eq(pred_cls, target).float().to(device)
    eps = 10e-6
    acc_uncertain = -pred_scores * torch.log(1-uncertainty+eps)
    inacc_certain = -(1-pred_scores) * torch.log(uncertainty+eps)
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    euc_loss = annealing_coef * acc_match * acc_uncertain + (1-annealing_coef) * (1-acc_match)*inacc_certain
    loss = torch.mean(euc_loss)
    return loss
'''
def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss



import torch.nn as nn
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        # print("compute focal loss")
        if self.size_average: return loss.mean()
        else: return loss.sum()


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: Tensor of shape (batch_size, feature_dim)
            labels: Tensor of shape (batch_size) with class labels
        Returns:
            loss: Supervised Contrastive Loss
        """
        batch_size = features.shape[0]

        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix (cosine similarity)
        # print(features.shape)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask to remove self-comparisons
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)

        # Get labels similarity mask
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(features.device)

        # Remove self-similarity for positive samples
        positives_mask = label_mask * (~mask).float()

        # Compute loss for all samples
        exp_similarity = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_similarity.sum(dim=1, keepdim=True))
        per_sample_loss = -(positives_mask * log_prob).sum(dim=1) / (positives_mask.sum(dim=1) + 1e-8)
        
        # Average loss over batch
        loss = per_sample_loss.mean()

        return loss
