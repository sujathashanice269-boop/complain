"""
客户重复投诉预测 - 增强训练工具（可选）
包含FocalLoss、EarlyStopping等训练辅助工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLabelSmoothingLoss(nn.Module):
    """
    Focal Loss with Label Smoothing
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 epsilon: float = 0.1, num_classes: int = 2):
        super(FocalLabelSmoothingLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        """
        # Label smoothing
        with torch.no_grad():
            smoothed_targets = torch.zeros_like(logits)
            smoothed_targets.fill_(self.epsilon / (self.num_classes - 1))
            smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)

        # Focal loss with smoothed targets
        log_probs = F.log_softmax(logits, dim=1)
        loss = -smoothed_targets * log_probs

        # Apply focal term
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma
        loss = self.alpha * focal_weight * loss

        return loss.sum(dim=1).mean()

class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    def __init__(self, patience: int = 7, delta: float = 0.001,
                 mode: str = 'max', verbose: bool = True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics like accuracy/AUC, 'min' for loss
            verbose: Whether to print messages
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int = 0) -> bool:
        """
        Args:
            score: Current metric value
            epoch: Current epoch number

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f'EarlyStopping: Initial best score = {score:.4f}')
            return False

        # Check if improved
        if self.mode == 'max':
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f'EarlyStopping: New best score = {score:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: No improvement for {self.counter} epochs')

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'EarlyStopping: Stopping at epoch {epoch}, '
                          f'best was epoch {self.best_epoch} with score {self.best_score:.4f}')

        return self.early_stop

    def reset(self):
        """Reset the early stopping counter"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

class GradientAccumulator:
    """
    Gradient accumulation for simulating larger batch sizes
    """
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_update(self) -> bool:
        """Check if should perform optimizer step"""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation"""
        return loss / self.accumulation_steps

class AdversarialTraining:
    """
    Adversarial training for improved robustness
    """
    def __init__(self, epsilon: float = 0.01, alpha: float = 0.001,
                 num_steps: int = 3):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def generate_adversarial_examples(self, model: nn.Module,
                                     inputs: torch.Tensor,
                                     targets: torch.Tensor,
                                     loss_fn: Callable) -> torch.Tensor:
        """
        Generate adversarial examples using PGD
        """
        # Save original inputs
        original_inputs = inputs.clone().detach()

        # Initialize perturbation
        perturbation = torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        perturbation = perturbation.requires_grad_(True)

        for _ in range(self.num_steps):
            # Add perturbation to inputs
            adv_inputs = inputs + perturbation

            # Forward pass
            outputs = model(adv_inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass
            loss.backward()

            # Update perturbation
            perturbation_grad = perturbation.grad.detach()
            perturbation = perturbation + self.alpha * perturbation_grad.sign()

            # Clamp perturbation
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            perturbation = torch.clamp(original_inputs + perturbation, 0, 1) - original_inputs

            # Clear gradients
            perturbation = perturbation.detach().requires_grad_(True)

        return original_inputs + perturbation

class MixupAugmentation:
    """
    Mixup data augmentation
    Reference: https://arxiv.org/abs/1710.09412
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Apply mixup augmentation
        """
        batch_size = inputs.size(0)

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1

        # Random shuffle indices
        indices = torch.randperm(batch_size).to(inputs.device)

        # Mix inputs and targets
        mixed_inputs = lam * inputs + (1 - lam) * inputs[indices]
        targets_a, targets_b = targets, targets[indices]

        return mixed_inputs, targets_a, targets_b, lam

class CosineAnnealingWarmupRestarts:
    """
    Cosine annealing with warm restarts learning rate scheduler
    """
    def __init__(self, optimizer, first_cycle_steps: int,
                 cycle_mult: float = 1.0, max_lr: float = 0.1,
                 min_lr: float = 0.001, warmup_steps: int = 0,
                 gamma: float = 1.0):
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0

        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self):
        """Update learning rate"""
        if self.step_in_cycle < self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * self.step_in_cycle / self.warmup_steps
        else:
            # Cosine annealing phase
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * (self.step_in_cycle - self.warmup_steps) /
                                  (self.cur_cycle_steps - self.warmup_steps)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Update cycle
        self.step_in_cycle += 1
        if self.step_in_cycle >= self.cur_cycle_steps:
            self.cycle += 1
            self.step_in_cycle = 0
            self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) *
                                      self.cycle_mult) + self.warmup_steps
            self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)

# 导出主要类
__all__ = [
    'FocalLoss',
    'FocalLabelSmoothingLoss',
    'EarlyStopping',
    'GradientAccumulator',
    'AdversarialTraining',
    'MixupAugmentation',
    'CosineAnnealingWarmupRestarts'
]

if __name__ == "__main__":
    # 测试代码
    print("增强训练工具模块")

    # 测试Focal Loss
    focal_loss = FocalLoss()
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 1, 0])
    loss = focal_loss(logits, targets)
    print(f"Focal Loss: {loss.item():.4f}")

    # 测试Early Stopping
    early_stopping = EarlyStopping(patience=3)
    scores = [0.7, 0.72, 0.71, 0.71, 0.70]
    for i, score in enumerate(scores):
        should_stop = early_stopping(score, i)
        if should_stop:
            print(f"Early stopping triggered at epoch {i}")
            break