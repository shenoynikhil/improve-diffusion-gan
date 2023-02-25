"""Utils"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as Fm
from torchvision.utils import save_image


def compute_metrics(
    real_pred,
    fake_pred,
    real_aux,
    fake_aux,
    valid,
    fake,
    labels,
    gen_labels,
    apply_sigmoid: bool = False,
    multi_label: bool = False,
):
    """Utility function to compute a bunch of metrics

    Parameters
    ----------
    real_pred: torch.Tensor
        real/not predictions on real images from discriminator
    fake_pred: torch.Tensor
        real/not predictions on gen images from discriminator
    real_aux: torch.Tensor
        ground-truth label predictions on real images from discriminator
    fake_aux: torch.Tensor
        ground-truth label predictions on gen images from discriminator
    valid: torch.Tensor
        real labels (all 1) real images from discriminator
    fake: torch.Tensor
        fake labels (all 0) real images from discriminator
    labels: torch.Tensor
        ground-truth labels real images from discriminator
    gen_labels: torch.Tensor
        ground-truth labels gen images from discriminator
    apply_sigmoid: bool, default False
        Applies sigmoid on real_pred and fake_pred if raw scores
    multi_label: bool, default False
        In the Multi Label Case, auxillary scores accuracy scores are calculaled differently
    """
    if apply_sigmoid:
        real_pred = torch.sigmoid(real_pred)
        fake_pred = torch.sigmoid(fake_pred)

    # Calculate discriminator accuracy
    pred = np.concatenate([real_pred.data.cpu().numpy(), fake_pred.data.cpu().numpy()], axis=0)
    gt = np.concatenate([valid.data.cpu().numpy(), fake.data.cpu().numpy()], axis=0)
    # d_acc = np.mean(np.argmax(pred, axis=1) == gt)
    pred = np.where(pred >= 0.5, 1, 0)
    d_acc = np.mean(pred == gt)

    if not multi_label:
        class_pred = np.concatenate(
            [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0
        )
        c_gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_class_acc = np.mean(np.argmax(class_pred, axis=1) == c_gt)
    else:
        # considering multi label binary case, therefore num_classes = 2
        class_pred = torch.cat([real_aux, fake_aux], axis=0).detach().cpu()
        c_gt = torch.cat([labels, gen_labels], axis=0).detach().cpu()
        class_pred = 1.0 * (np.sig(class_pred) > 0.5)
        d_class_acc = Fm.accuracy(class_pred, c_gt, num_classes=2)

    return {
        "D Accuracy": d_acc * 100,
        "D Class Accuracy": d_class_acc * 100,
    }


def compute_metrics_no_aux(
    real_pred,
    fake_pred,
    valid,
    fake,
    apply_sigmoid: bool = False,
):
    """Compute Basic Metrics"""
    if apply_sigmoid:

        real_pred = torch.sigmoid(real_pred)
        fake_pred = torch.sigmoid(fake_pred)

    # Calculate discriminator accuracy
    pred = np.concatenate([real_pred.data.cpu().numpy(), fake_pred.data.cpu().numpy()], axis=0)
    gt = np.concatenate([valid.data.cpu().numpy(), fake.data.cpu().numpy()], axis=0)
    # d_acc = np.mean(np.argmax(pred, axis=1) == gt)
    pred = np.where(pred >= 0.5, 1, 0)
    d_acc = np.mean(pred == gt)

    return {"D Accuracy": d_acc * 100}


def sample_image(gen_imgs, n_row: int, epochs_done: int, output_dir: str) -> None:
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Get labels ranging from 0 to n_classes for n rows and generate images from noise
    save_dir = os.path.join(output_dir, "images/")
    os.makedirs(save_dir, exist_ok=True)
    save_image(
        gen_imgs.data,
        os.path.join(save_dir, f"{epochs_done}.png"),
        nrow=n_row,
        normalize=True,
    )


def weights_init_normal(m):
    """Initialize the weights normally"""
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def focal_binary_cross_entropy(inputs, targets, gamma=2, alpha=1, logits=True, reduce=True):
    """Computes focal_loss given inputs and targets"""
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss
