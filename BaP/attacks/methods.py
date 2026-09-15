from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.clip import LoadedCLIP, normalize_pixels

@dataclass
class ZeroShotCLIPClassifier:
    loaded: LoadedCLIP
    text_features: torch.Tensor

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.loaded.model.get_image_features(
            pixel_values=normalize_pixels(images.to(self.loaded.device))
        )
        image_features = F.normalize(image_features, dim=-1)
        scale = self.loaded.model.logit_scale.exp()
        return image_features @ self.text_features.to(self.loaded.device).T * scale


def pgd_attack(
    classifier: ZeroShotCLIPClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    alpha: float,
) -> torch.Tensor:
    images = images.to(classifier.loaded.device)
    labels = labels.to(classifier.loaded.device)
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, 0.0, 1.0) - images
    delta.requires_grad_(True)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        adv = torch.clamp(images + delta, 0.0, 1.0)
        logits = classifier.logits(adv)
        loss = loss_fn(logits, labels)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * grad.sign(), -epsilon, epsilon)
        delta.grad.zero_()

    return torch.clamp(images + delta.detach(), 0.0, 1.0)


def cw_linf_attack(
    classifier: ZeroShotCLIPClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    alpha: float,
    kappa: float = 0.0,
) -> torch.Tensor:
    images = images.to(classifier.loaded.device)
    labels = labels.to(classifier.loaded.device)
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, 0.0, 1.0) - images
    delta.requires_grad_(True)

    for _ in range(steps):
        adv = torch.clamp(images + delta, 0.0, 1.0)
        logits = classifier.logits(adv)
        correct = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        mask = torch.zeros_like(logits).scatter_(1, labels.view(-1, 1), float("inf"))
        other, _ = torch.max(logits - mask, dim=1)
        loss = torch.clamp(other - correct + kappa, min=-kappa).sum()
        loss.backward()
        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + alpha * grad.sign(), -epsilon, epsilon)
        delta.data = torch.clamp(images + delta.data, 0.0, 1.0) - images
        delta.grad.zero_()

    return torch.clamp(images + delta.detach(), 0.0, 1.0)


def _low_freq_mask(height: int, width: int, radius: float, device: torch.device) -> torch.Tensor:
    y = torch.arange(height, device=device).view(-1, 1)
    x = torch.arange(width, device=device).view(1, -1)
    cy, cx = height // 2, width // 2
    return (torch.sqrt((x - cx) ** 2 + (y - cy) ** 2) <= radius).float()


def low_freq_pgd_attack(
    classifier: ZeroShotCLIPClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    alpha: float,
    freq_radius: float,
) -> torch.Tensor:
    images = images.to(classifier.loaded.device)
    labels = labels.to(classifier.loaded.device)
    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, 0.0, 1.0) - images
    delta.requires_grad_(True)
    loss_fn = nn.CrossEntropyLoss()
    _, _, h, w = images.shape
    mask = _low_freq_mask(h, w, freq_radius, classifier.loaded.device)

    for _ in range(steps):
        adv = torch.clamp(images + delta, 0.0, 1.0)
        logits = classifier.logits(adv)
        loss = loss_fn(logits, labels)
        loss.backward()
        grad = delta.grad.detach()
        proposal = delta + alpha * grad.sign()
        fft = torch.fft.fftshift(torch.fft.fft2(proposal), dim=(-2, -1))
        filtered = fft * mask.view(1, 1, h, w)
        delta.data = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1))).real
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(images + delta.data, 0.0, 1.0) - images
        delta.grad.zero_()

    return torch.clamp(images + delta.detach(), 0.0, 1.0)


def apgd_ce_attack(
    classifier: ZeroShotCLIPClassifier,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
) -> torch.Tensor:
    images = images.to(classifier.loaded.device)
    labels = labels.to(classifier.loaded.device)
    batch_size = images.shape[0]
    loss_best = torch.full((images.shape[0],), -1e9, device=classifier.loaded.device)
    x_best = images.clone()
    x_adv = images + torch.zeros_like(images).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv.requires_grad = True
    step_size = torch.ones(batch_size, 1, 1, 1, device=classifier.loaded.device) * (2.0 * epsilon)
    checkpoints = [max(int(0.22 * steps), 1), max(int(0.48 * steps), 1), max(int(0.78 * steps), 1)]
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for i in range(steps):
        logits = classifier.logits(x_adv)
        loss_indiv = loss_fn(logits, labels)
        loss_total = loss_indiv.sum()
        classifier.loaded.model.zero_grad()
        loss_total.backward()
        grad = x_adv.grad.data

        with torch.no_grad():
            is_better = loss_indiv > loss_best
            loss_best[is_better] = loss_indiv[is_better]
            current_x = x_adv.detach()
            if i == 0:
                x_best = current_x.clone()
            else:
                x_best[is_better] = current_x[is_better]

            if (i + 1) in checkpoints:
                step_size = step_size * 0.5
            x_next = x_adv + step_size * grad.sign()
            delta = torch.clamp(x_next - images, -epsilon, epsilon)
            x_adv.data = torch.clamp(images + delta, 0.0, 1.0)
            x_adv.grad.zero_()

    return x_best.detach()
