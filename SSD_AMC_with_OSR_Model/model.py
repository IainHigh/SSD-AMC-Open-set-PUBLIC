# pylint: disable=import-error
"""
model.py:
Model definition used for AMC experiments.
"""

# Standard library imports
from math import pi

# Third-party imports
import torch
from torch import nn
import torch.nn.functional as F

# Local application imports
from config import (
    S,
    B,
    NUM_CLASSES,
    NUMTAPS,
    SAMPLING_FREQUENCY,
    EMBED_DIM,
    OPENSET_ENABLE,
    WINDOW_CHOICE,
)

# Should only have one public method, the model's forward pass.
# pylint: disable=too-few-public-methods


# Mapping from window name to the torch function constructing that window.
WINDOW_FUNCTIONS = {
    "Rectangular": lambda M, dtype, device: torch.ones(M, dtype=dtype, device=device),
    "Triangular": lambda M, dtype, device: torch.bartlett_window(
        M, periodic=False, dtype=dtype, device=device
    ),
    "Hanning": lambda M, dtype, device: torch.hann_window(
        M, periodic=False, dtype=dtype, device=device
    ),
    "Hamming": lambda M, dtype, device: torch.hamming_window(
        M, periodic=False, dtype=dtype, device=device
    ),
    "Blackman": lambda M, dtype, device: torch.blackman_window(
        M, periodic=False, dtype=dtype, device=device
    ),
    "Kaiser-Bessel": lambda M, dtype, device: torch.kaiser_window(
        M, beta=8.6, periodic=False, dtype=dtype, device=device
    ),
}


class Model(nn.Module):
    """Single-shot detector architecture for the joint detection and classification of signals.

    Args:
        nn (Module): PyTorch neural network module.

    Returns:
        Module: The model.
    """

    class _ResidualBlock(nn.Module):
        """Small residual block used throughout the network."""

        def __init__(self, in_ch, out_ch):
            """Initializes the residual block.

            Args:
                in_ch (int): Number of input channels.
                out_ch (int): Number of output channels.
            """
            super().__init__()
            # Distribute the output channels across three branches so that the
            # concatenated result matches ``out_ch`` regardless of the embedding
            # dimension. Extra channels (if any) are assigned to the first
            # branch.
            branch_ch = out_ch // 3
            extra = out_ch - 3 * branch_ch
            b1_ch = branch_ch + extra
            b2_ch = branch_ch
            b3_ch = branch_ch

            self.branch1 = nn.Sequential(
                nn.Conv1d(in_ch, b1_ch, kernel_size=1, stride=2),
                nn.BatchNorm1d(b1_ch),
                nn.ReLU(),
            )
            self.branch2 = nn.Sequential(
                nn.Conv1d(in_ch, b2_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(b2_ch),
                nn.ReLU(),
            )
            self.branch3 = nn.Sequential(
                nn.Conv1d(in_ch, b3_ch, kernel_size=1, stride=2),
                nn.BatchNorm1d(b3_ch),
                nn.ReLU(),
            )
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=2),
                nn.BatchNorm1d(out_ch),
            )

        def forward(self, x):
            """Forward pass through the residual block.

            Args:
                x (torch.Tensor): Input tensor of shape [N, in_ch, T].

            Returns:
                torch.Tensor: Output tensor of shape [N, out_ch, T'].
            """
            res = self.residual(x)
            b1 = self.branch1(x)
            b2 = self.branch2(x)
            b3 = self.branch3(x)
            concat = torch.cat([b1, b2, b3], dim=1)
            out = F.relu(concat + res)
            return out

    class _Classifier(nn.Module):
        """Classifier head predicting confidence and classes from baseband."""

        def __init__(self, num_out):
            """Initializes the classifier with the given number of outputs.

            Args:
                num_out (int): Number of output classes.
            """
            super().__init__()
            # Conv Block 1 (as in narrowband model)
            self.conv_block1 = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=8, stride=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            )
            # Block 2 repeated 4 times
            self.block2_layers = nn.ModuleList(
                [
                    self._create_block2(32 if i == 0 else EMBED_DIM, EMBED_DIM)
                    for i in range(4)
                ]
            )
            # Global Average Pooling
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            # EMBED_DIM-D embedding  →  (1+NUM_CLASSES) logits
            self.fc = nn.Linear(EMBED_DIM, num_out)

        def _create_block2(self, in_channels, out_channels):
            """Create a Block 2 module.

            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.

            Returns:
                nn.ModuleDict: A dictionary containing the branches and residual connections.
            """
            branch_ch = out_channels // 3
            extra = out_channels - 3 * branch_ch
            b1_ch = branch_ch + extra
            b2_ch = branch_ch
            b3_ch = branch_ch
            return nn.ModuleDict(
                {
                    "branch1": nn.Sequential(
                        nn.Conv1d(in_channels, b1_ch, kernel_size=1, stride=2),
                        nn.BatchNorm1d(b1_ch),
                        nn.ReLU(),
                    ),
                    "branch2": nn.Sequential(
                        nn.Conv1d(
                            in_channels, b2_ch, kernel_size=3, stride=2, padding=1
                        ),
                        nn.BatchNorm1d(b2_ch),
                        nn.ReLU(),
                    ),
                    "branch3": nn.Sequential(
                        nn.Conv1d(in_channels, b3_ch, kernel_size=1, stride=2),
                        nn.BatchNorm1d(b3_ch),
                        nn.ReLU(),
                    ),
                    "residual": nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2),
                        nn.BatchNorm1d(out_channels),
                    ),
                }
            )

        def forward(self, x):
            """Forward pass through the classifier.

            Args:
                x (torch.Tensor): Input tensor of shape [N, 2, T].

            Returns:
                torch.Tensor: Output a tuple (logits, feat).
            """
            # ``x`` has shape ``[N, 2, T]`` where ``T`` is the number of samples.
            x = self.conv_block1(x)
            for block in self.block2_layers:
                residual = block["residual"](x)
                branch1 = block["branch1"](x)
                branch2 = block["branch2"](x)
                branch3 = block["branch3"](x)
                concatenated = torch.cat([branch1, branch2, branch3], dim=1)
                x = F.relu(concatenated + residual)
            x = self.global_avg_pool(x)
            feat = x.view(x.size(0), -1)  # (B,EMBED_DIM)
            logits = self.fc(feat)
            return (logits, feat)

    def __init__(self, num_samples):
        """Initializes the model.

        Args:
            num_samples (int): The number of samples in the dataset.
        """
        super().__init__()
        self.num_samples = num_samples
        self.class_means = None
        self.inv_cov = None

        # -----------------------
        # Stage-1: Frequency Prediction
        # -----------------------
        # Time-domain branch.
        self.first_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=8, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.stage1_blocks = nn.Sequential(
            self._ResidualBlock(32, EMBED_DIM),
            self._ResidualBlock(EMBED_DIM, EMBED_DIM),
            self._ResidualBlock(EMBED_DIM, EMBED_DIM),
            self._ResidualBlock(EMBED_DIM, EMBED_DIM),
            self._ResidualBlock(EMBED_DIM, EMBED_DIM),
        )
        self.pool_1 = nn.AdaptiveAvgPool1d(1)

        # -----------------------
        # Dynamic Anchor Setup for Frequency Prediction
        # -----------------------
        initial_anchor_values = torch.linspace(
            1 / (B + 1), B / (B + 1), B, dtype=torch.float32
        )  # shape: [B]
        # Store logits so anchors stay in [0,1] after sigmoid.
        anchor_logits = initial_anchor_values.logit()
        self.anchor_logits = nn.Parameter(
            anchor_logits.unsqueeze(0).repeat(S, 1)
        )  # shape: [S, B]
        self.freq_predictor = nn.Linear(EMBED_DIM, S * B)
        self.band_predictor = nn.Linear(EMBED_DIM, S * B)
        self.anchors = torch.sigmoid(self.anchor_logits)

        # -----------------------
        # Stage-2: Confidence and Classification using the new classifier.
        # -----------------------
        # This classifier follows the narrowband architecture (adapted for 1+NUM_CLASSES outputs)
        self.classifier = self._Classifier(num_out=1 + NUM_CLASSES)

    def forward(self, x_time):
        """Run the model on ``x_time`` input.

        Args:
            x_time (torch.Tensor): Time-domain input tensor of shape [N, 2, T].

        Returns:
            tuple: A tuple containing the raw predictions and embeddings.
        """
        bsz = x_time.size(0)
        # -----------------------
        # Stage-1: Coarse Frequency and BW Prediction
        # -----------------------
        h1 = self.first_conv(x_time)
        h1 = self.stage1_blocks(h1)
        h1 = self.pool_1(h1).squeeze(-1)  # [bsz, EMBED_DIM]

        # 1.1: Frequency Prediction
        # Predict delta for frequency offset.
        raw_delta = self.freq_predictor(h1).view(bsz, S, B)  # shape: [bsz, S, B]
        delta = 0.5 * torch.tanh(raw_delta)
        self.anchors = torch.sigmoid(self.anchor_logits)  # [S,B]
        freq_pred = self.anchors.unsqueeze(0) + delta  # [bsz,S,B]
        freq_pred = freq_pred.clamp(0.0, 1.0)

        cell_indices = torch.arange(
            S, device=freq_pred.device, dtype=freq_pred.dtype
        ).view(1, S, 1)
        freq_pred_raw = (cell_indices + freq_pred) * (SAMPLING_FREQUENCY / 2) / S
        freq_pred_flat = freq_pred_raw.view(bsz * S * B)

        # 1.2: BW Prediction
        bw_raw = self.band_predictor(h1)  # [bsz, S*B]
        bw_pred = F.softplus(bw_raw).view(bsz, S, B)
        bw_pred_flat = bw_pred.view(bsz * S * B)

        # -----------------------
        # Stage-2: Downconversion and Classification
        # -----------------------
        x_rep = x_time.unsqueeze(1).unsqueeze(1).expand(-1, S, B, -1, -1)
        x_rep = x_rep.contiguous().view(bsz * S * B, 2, self.num_samples)

        x_filt = self._filter_raw(x_rep, freq_pred_flat, bw_pred_flat)
        x_base = self._downconvert_multiple(x_filt, freq_pred_flat)

        logits, embed = self.classifier(x_base)
        logits = logits.view(bsz, S, B, 1 + NUM_CLASSES)

        final_out = torch.zeros(
            bsz,
            S,
            B,
            1 + 1 + 1 + NUM_CLASSES,
            dtype=logits.dtype,
            device=logits.device,
        )
        final_out[..., 0] = freq_pred  # offset
        final_out[..., 1] = logits[..., 0]  # confidence
        final_out[..., 2] = bw_pred  # bandwidth (norm.)
        final_out[..., 3:] = logits[..., 1:]  # classes
        final_out = final_out.view(bsz, S, B * (1 + 1 + 1 + NUM_CLASSES))

        if OPENSET_ENABLE:
            embed = embed.view(bsz, S, B, -1)  # (bsz,S,B,EMBED_DIM)
            return final_out, embed
        return final_out, None

    def _conv1d_batch(self, x, weight, pad_left, pad_right):
        """Apply 1‑D convolution on a batch using unfolding.

        Args:
            x (torch.Tensor): Input tensor of shape [N, 2, T].
            weight (torch.Tensor): Weight tensor of shape [N, 2, M].
            pad_left (int): Amount of padding to add on the left.
            pad_right (int): Amount of padding to add on the right.

        Returns:
            torch.Tensor: Output tensor of shape [N, 2, T].
        """
        x_padded = F.pad(x, (pad_left, pad_right))
        x_unf = x_padded.unfold(dimension=2, size=weight.shape[-1], step=1)
        weight = weight.unsqueeze(2)
        y = (x_unf * weight).sum(dim=-1)
        return y

    def _filter_raw(self, x_flat, freq_flat, bandwidth_flat):
        """Filter raw model outputs based on confidence and apply bandwidth constraints.

        Args:
            x_flat (torch.Tensor): Input tensor of shape [N, 2, T], where N is the batch size,
            freq_flat (torch.Tensor): Frequency tensor of shape [N], where N is the batch size.
            bandwidth_flat (torch.Tensor): Bandwidth tensor of shape [N], where N is the batch size.

        Returns:
            torch.Tensor: Filtered output tensor of shape [N, 2, T].
        """
        n_batch, _, n_samples = x_flat.shape
        device, dtype = x_flat.device, x_flat.dtype
        # ---------- low‑pass kernel ----------
        alpha = (NUMTAPS - 1) / 2.0
        n = torch.arange(NUMTAPS, device=device, dtype=dtype) - alpha  # [M]
        cutoff_norm = bandwidth_flat.clamp(1e-4, float(S)) / S  # [N]
        x = cutoff_norm.unsqueeze(1) * n  # [N,M]

        # use PyTorch’s numerically‑stable sinc
        sinc = torch.sinc(x)  # sin(pi x)/(pi x)

        h_lp = cutoff_norm.unsqueeze(1) * sinc  # [N,M]
        h_lp = h_lp / h_lp.sum(dim=1, keepdim=True).clamp_min(1e-12)

        win_func = WINDOW_FUNCTIONS.get(WINDOW_CHOICE)
        win = win_func(NUMTAPS, dtype, device)
        h_lp = (h_lp * win) / (h_lp * win).sum(dim=1, keepdim=True)

        # ---------- shift to band‑pass ----------
        f0 = freq_flat.view(n_batch, 1)  # [N,1]
        cos_factor = torch.cos(2 * pi * f0 * n / SAMPLING_FREQUENCY)  # [N,M]

        h_bp_all = h_lp * cos_factor  # [N,M]

        # duplicate for I & Q channels
        h_bp_all_expanded = h_bp_all.unsqueeze(1).repeat(1, 2, 1)  # [N,2,M]

        # ---------- apply FIR by unfolding ----------
        x_reshaped = x_flat.reshape(n_batch * 2, 1, n_samples)  # [N*2,1,T]
        weight = h_bp_all_expanded.reshape(n_batch * 2, 1, NUMTAPS)  # [N*2,1,M]

        pad_left = NUMTAPS // 2
        pad_right = NUMTAPS - 1 - pad_left
        y = self._conv1d_batch(x_reshaped, weight, pad_left, pad_right)  # [N*2,1,T]
        y = y.reshape(n_batch, 2, n_samples)
        return y

    def _downconvert_multiple(self, x_flat, freq_flat):
        """Downconvert a batch of signals by multiple frequencies.

        Args:
            x_flat (torch.Tensor): Input tensor of shape [N, 2, T].
            freq_flat (torch.Tensor): Frequency tensor of shape [N].

        Returns:
            torch.Tensor: Downconverted tensor of shape [N, 2, T].
        """
        device = x_flat.device
        dtype = x_flat.dtype
        _, _, num_samples = x_flat.shape
        t = (
            torch.arange(num_samples, device=device, dtype=dtype).unsqueeze(0)
            / SAMPLING_FREQUENCY
        )
        freq_flat = freq_flat.unsqueeze(-1)
        angle = -2.0 * pi * freq_flat * t
        shift_real = torch.cos(angle)
        shift_imag = torch.sin(angle)
        x_real = x_flat[:, 0, :]
        x_imag = x_flat[:, 1, :]
        y_real = x_real * shift_real - x_imag * shift_imag
        y_imag = x_real * shift_imag + x_imag * shift_real
        x_base = torch.stack([y_real, y_imag], dim=1)
        return x_base
