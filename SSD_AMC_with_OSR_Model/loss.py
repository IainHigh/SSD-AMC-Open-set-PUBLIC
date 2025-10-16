# pylint: disable=import-error
"""
loss.py:
This module implements the loss function for the SSD AMC OSR model.
The loss function includes components for:
- IoU
- confidence
- classification
- Triplet loss (for open-set recognition)
The weights for these components are configurable through the `config` module.
It also provides functionality for detailed loss tracking and printing.
"""

import torch
from torch import nn
import torch.nn.functional as F

from config import (
    B,
    NUM_CLASSES,
    IOU_LOSS,
    LAMBDA_NOOBJ,
    LAMBDA_CLASS,
    LAMBDA_TRIPLET,
    TRIPLET_MARGIN,
    DETAILED_LOSS_PRINT,
    EMBED_DIM,
    OPENSET_ENABLE,
)


class Loss(nn.Module):
    """Single-Shot Detector AMC Loss Function

    Args:
        nn (Module): PyTorch neural network module
    """

    def __init__(self):
        """Initialize the loss function."""
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=TRIPLET_MARGIN, p=2)
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        """Reset accumulators used for detailed loss printing."""
        self._epoch_stats = {
            "iou": 0.0,
            "conf_obj": 0.0,
            "conf_noobj": 0.0,
            "cls": 0.0,
            "triplet": 0.0,
        }
        self._samples = 0

    def _update_epoch_stats(self, batch_size, loss_components):
        """Update epoch statistics.

        Args:
            batch_size (int): Number of samples in the batch.
            loss_components (dict): Dictionary containing individual loss components.
        """
        self._epoch_stats["iou"] += loss_components["iou"]
        self._epoch_stats["conf_obj"] += loss_components["conf_obj"]
        self._epoch_stats["conf_noobj"] += loss_components["conf_noobj"]
        self._epoch_stats["cls"] += loss_components["cls"]
        self._epoch_stats["triplet"] += loss_components["triplet"]
        self._samples += batch_size

    def print_epoch_stats(self):
        """Print the averaged loss components for the epoch."""
        if self._samples == 0:
            return
        stats = {k: v / self._samples for k, v in self._epoch_stats.items()}
        print("\tDetailed Loss (avg per sample):")
        print(f"\t\tIoULoss: {stats['iou']:.4f}")
        print(f"\t\tConfLossObj: {stats['conf_obj']:.4f}")
        print(f"\t\tConfLossNoObj: {stats['conf_noobj']:.4f}")
        print(f"\t\tClsLoss: {stats['cls']:.4f}")
        print(f"\t\tTripletLoss: {stats['triplet']:.4f}")

    def _calculate_iou_loss(self, pred, target):
        """Calculate Intersection over Union (IoU) loss.

        Args:
            pred (torch.Tensor): Predicted bounding boxes.
            target (torch.Tensor): Target bounding boxes.

        Returns:
            torch.Tensor: IoU loss.
        """
        # ----- IoU between predicted and target frequency regions -----
        pred_low = pred[..., 0] - (pred[..., 2] / 2.0)
        pred_high = pred[..., 0] + (pred[..., 2] / 2.0)
        tgt_low = target[..., 0] - (target[..., 2] / 2.0)
        tgt_high = target[..., 0] + (target[..., 2] / 2.0)

        inter_low = torch.maximum(pred_low, tgt_low)
        inter_high = torch.minimum(pred_high, tgt_high)
        intersection = (inter_high - inter_low).clamp(min=0.0)

        union_low = torch.minimum(pred_low, tgt_low)
        union_high = torch.maximum(pred_high, tgt_high)
        union = (union_high - union_low).clamp(min=1e-6)

        return intersection / union

    def _calculate_cls_loss(self, pred, target, obj_mask):
        """Calculate the classification loss.

        Args:
            pred (torch.Tensor): Predicted class probabilities.
            target (torch.Tensor): Target class labels.
            obj_mask (torch.Tensor): Object mask indicating presence of objects.

        Returns:
            torch.Tensor: Classification loss.
        """
        cls_tgt = target[..., 3:]
        cls_pred = pred[..., 3:]

        with torch.no_grad():
            tgt_idx = cls_tgt.argmax(dim=-1)
        if obj_mask.sum() > 0:
            ce = F.cross_entropy(
                cls_pred[obj_mask.bool()],
                tgt_idx[obj_mask.bool()],
                reduction="sum",
            )
        else:
            ce = torch.tensor(0.0, device=cls_pred.device)
        return ce

    def _calculate_triplet_loss(self, target, embed, centers, obj_mask):
        """Compute triplet loss using class centres as positives/negatives."""
        if obj_mask.sum() == 0:
            return torch.tensor(0.0, device=embed.device)

        cls_tgt = target[..., 3:]
        with torch.no_grad():
            gt_idx = cls_tgt.argmax(dim=-1)

        obj_mask_flat = obj_mask.bool().view(-1)
        emb_flat = embed.view(embed.size(0), embed.size(1), B, EMBED_DIM)
        anchor = emb_flat.view(-1, EMBED_DIM)[obj_mask_flat]
        labels = gt_idx.view(-1)[obj_mask_flat]

        pos = centers[labels]

        rand = torch.randint(
            0, centers.size(0) - 1, (labels.size(0),), device=embed.device
        )
        neg_idx = rand + (rand >= labels).long()
        neg = centers[neg_idx]

        return self.triplet_loss(anchor, pos, neg)

    def forward(self, pred, target, embed, centers):
        """Compute the loss for the model.

        Args:
            pred (torch.Tensor): The predicted output from the model.
            target (torch.Tensor): The ground truth target output.
            embed (torch.Tensor): The embedding features from the model.
            centers (torch.Tensor): The class centers for the embeddings.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Create a dictionary to hold the loss components
        loss_components = {
            "iou": 0.0,
            "conf_obj": 0.0,
            "conf_noobj": 0.0,
            "cls": 0.0,
            "triplet": 0.0,
        }

        batch_size = pred.shape[0]
        pred = pred.view(batch_size, pred.shape[1], B, 1 + 1 + 1 + NUM_CLASSES)
        target = target.view_as(pred)

        obj_mask = (target[..., 1] > 0).float()

        iou_1d = self._calculate_iou_loss(pred, target)
        loss_components["iou"] = IOU_LOSS * torch.sum(obj_mask * (1.0 - iou_1d))

        conf_pred = pred[..., 1]
        loss_components["conf_obj"] = torch.sum(obj_mask * (conf_pred - iou_1d) ** 2)
        loss_components["conf_noobj"] = LAMBDA_NOOBJ * torch.sum(
            (1.0 - obj_mask) * (conf_pred**2)
        )

        loss_components["cls"] = LAMBDA_CLASS * self._calculate_cls_loss(
            pred, target, obj_mask
        )

        if OPENSET_ENABLE:
            loss_components["triplet"] = LAMBDA_TRIPLET * self._calculate_triplet_loss(
                target, embed, centers, obj_mask
            )
        else:
            loss_components["triplet"] = torch.tensor(0.0, device=pred.device)

        # Calculate the total loss by summing the components
        total_loss = sum(loss_components.values())

        if DETAILED_LOSS_PRINT:
            self._update_epoch_stats(batch_size, loss_components)

        return total_loss / batch_size
