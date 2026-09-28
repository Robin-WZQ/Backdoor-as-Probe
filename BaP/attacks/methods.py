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