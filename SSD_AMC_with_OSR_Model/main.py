# pylint: disable=import-error
"""
main.py:
Main function for the single-shot detector architecture model training and testing.
This is the entry point for the training and testing of the SSD AMC OSR model.
"""

# Standard library imports
import argparse
import sys
import os
import gc
import random
from uuid import uuid4
from warnings import filterwarnings
import contextlib
import io

# Third-party imports
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from adjustText import adjust_text
import adjustText as _adjustText
from tqdm import tqdm
from matplotlib.patches import Ellipse
from scipy import special

# Local application imports
import config as cfg
from dataset import Dataset
from model import Model
from loss import Loss

# Suppress debug prints from adjustText's random_shifts function
if hasattr(_adjustText, "random_shifts"):
    _orig_random_shifts = _adjustText.random_shifts

    def _silent_random_shifts(*args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return _orig_random_shifts(*args, **kwargs)

    _adjustText.random_shifts = _silent_random_shifts

# Define global variables
# pylint: disable=invalid-name
# pylint: disable=global-statement
out_dir = None
device = None
model = None
optimizer = None
criterion = None

# Feature plotting accumulators keyed by SNR range tuple (snr_min, snr_max).
feature_store = {}


def _normalise_tsne_snr_ranges():
    """Return validated SNR ranges for t-SNE plotting."""

    snr_ranges = []
    for rng in cfg.TSNE_SNR_RANGES:
        if len(rng) != 2:
            raise ValueError(
                "Each TSNE_SNR_RANGES entry must contain exactly 2 values: (min_snr, max_snr)."
            )
        low, high = float(rng[0]), float(rng[1])
        if low > high:
            raise ValueError(
                f"Invalid SNR range ({low}, {high}) in TSNE_SNR_RANGES: min must be <= max."
            )
        snr_ranges.append((low, high))
    return snr_ranges


def _initialise_feature_store():
    """Initialise feature buffers used for range-specific t-SNE plots."""

    global feature_store
    feature_store = {
        rng: {"train_feat": [], "train_labels": [], "test_feat": [], "test_labels": []}
        for rng in _normalise_tsne_snr_ranges()
    }


def _snr_range_label(snr_range):
    """Return a human-readable SNR range label."""
    return f"SNR {snr_range[0]:g} to {snr_range[1]:g} dB"


def _snr_range_file_suffix(snr_range):
    """Return a filesystem-safe suffix for an SNR range."""

    low = f"{snr_range[0]:g}".replace("-", "m").replace(".", "p")
    high = f"{snr_range[1]:g}".replace("-", "m").replace(".", "p")
    return f"snr_{low}_{high}dB"


def _initialise():
    """Initialises the training environment and sets global variables."""

    global out_dir, device

    # 1) Ignore warning messages that we'd expect to see

    # NumPy’s “Casting complex values to real …”
    # This is to be expected as we're converting the IQ data to real values.
    filterwarnings("ignore", category=np.ComplexWarning)

    # PyTorch DataLoader’s “This DataLoader will create …”
    # This is to be expected as we're using multiple workers for the DataLoader.
    filterwarnings(
        "ignore",
        message=r"This DataLoader will create .* worker processes",
        category=UserWarning,
    )

    # 2) Set random seed for reproducibility
    rng_seed = cfg.RNG_SEED
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # 3) Print the configuration file
    if cfg.PRINT_CONFIG_FILE:
        cfg.print_config_file()
    print("\n=== TRAINING ===")

    # 4) Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5) Set the data directory and output directory
    uuid_str = str(uuid4())
    dir_name = "test_result_plots_" + uuid_str
    out_dir = os.path.join("../OutputFiles", dir_name)
    print("\nWriting results to:", dir_name)
    os.makedirs(out_dir, exist_ok=True)

    if cfg.PLOT_FEATURE_DISTRIBUTION:
        _initialise_feature_store()


def main(train_dir=None, test_dir=None):
    """
    Main function for the SSD AMC OSR model training and testing.
    """

    global model, optimizer, criterion
    _initialise()

    # 1) Build dataset and loaders
    train_dataset = Dataset(train_dir, transform=None, class_list=None)

    cfg.MODULATION_CLASSES = train_dataset.class_list

    test_dataset = Dataset(
        test_dir,
        transform=None,
        class_list=cfg.MODULATION_CLASSES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )

    # 2) Create model & loss
    num_samples = train_dataset.get_num_samples()
    model = Model(num_samples).to(device)
    criterion = Loss().to(device)

    # 3) Optimiser
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
    )

    # 4) Training loop
    _training_loop(train_loader)

    # 5) Test the model
    _run_phase(
        test_loader,
        "test",
        plot=cfg.PLOT_TEST_SAMPLES,
        write=cfg.WRITE_TEST_RESULTS,
    )

    if cfg.PLOT_FEATURE_DISTRIBUTION:
        for snr_range, store in feature_store.items():
            _plot_feature_distribution(
                np.array(store["train_feat"]),
                np.array(store["train_labels"]),
                np.array(store["test_feat"]),
                np.array(store["test_labels"]),
                out_dir,
                snr_range=snr_range,
            )


def _training_loop(train_loader):
    """Run the training loop for the model.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
    """
    for epoch in range(cfg.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{cfg.EPOCHS}]")

        # 1) Adjust learning rate for this epoch
        prog = epoch / (cfg.EPOCHS - 1) if cfg.EPOCHS > 1 else 0.0
        learn_rate = cfg.LEARNING_RATE * (cfg.FINAL_LR_MULTIPLE**prog)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learn_rate

        # 2) Run training phase
        (
            avg_train_loss,
            train_mean_freq_err,
            train_cls_accuracy,
            train_prec,
            train_rec,
            train_f1,
        ) = _run_phase(
            train_loader,
            "train",
            epoch=epoch,
        )

        # 3) Print training results
        train_mean_freq_err = _convert_to_readable(train_mean_freq_err, 0)[0]
        print(
            f"\tTrain: Loss={avg_train_loss:.4f}, "
            f"MeanFreqErr={train_mean_freq_err}, "
            f"ClsAcc={train_cls_accuracy:.2f}%, "
            f"P={train_prec:.3f}, R={train_rec:.3f}, F1={train_f1:.3f}"
        )

        # 4) Garbage collection and emptying CUDA cache.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()


def _setup_phase(mode, plot, write):
    """Setup the training or testing phase.

    Args:
        mode (str): The mode of operation ("train" or "test").
        plot (int): Number of samples to plot results.
        write (bool): Whether to write results to disk.

    Returns:
        dict: A dictionary containing the setup phase information.
    """

    global criterion

    is_train = mode == "train"

    model.train(is_train)

    if criterion is None:
        criterion = Loss().to(device)

    if cfg.DETAILED_LOSS_PRINT:
        criterion.reset_epoch_stats()

    if model.class_means is not None:
        current_centers = model.class_means.to(device)
    else:
        current_centers = (
            torch.randn(cfg.NUM_CLASSES, cfg.EMBED_DIM, device=device) * 0.1
        )

    emb_acc = [[] for _ in range(cfg.NUM_CLASSES)] if is_train else None

    if not is_train:
        overall_true_classes = []
        overall_pred_classes = []
        snr_dicts = {
            "obj_count": {},
            "correct_cls": {},
            "freq_err": {},
            "tp": {},
            "fp": {},
            "fn": {},
            "matched": {},
            "iou": {},
        }
        snr_classes = {}

        if plot or write:
            os.makedirs(
                os.path.join(out_dir, "frequency_domain_representations"), exist_ok=True
            )
        entries = []
    else:
        overall_true_classes = None
        overall_pred_classes = None
        snr_dicts = None
        snr_classes = None
        entries = None

    return {
        "is_train": is_train,
        "is_test": not is_train,
        "require_grad": is_train,
        "current_centers": current_centers,
        "emb_acc": emb_acc,
        "overall_true_classes": overall_true_classes,
        "overall_pred_classes": overall_pred_classes,
        "snr_dicts": snr_dicts,
        "snr_classes": snr_classes,
        "entries": entries,
        "plot": plot,
        "write": write,
    }


def _prepare_data(batch):
    """Prepare data for model input.

    Args:
        batch (tuple): A tuple containing the input data.

    Returns:
        tuple: A tuple containing the prepared input data.
    """

    time_data, label_tensor, snr_tensor = batch

    time_data = time_data.to(device, non_blocking=True)
    label_tensor = label_tensor.to(device, non_blocking=True)

    return time_data, label_tensor, snr_tensor


def _forward_pass(time_data, label_tensor, current_centers, require_grad):
    """Run the model forward (and backward if training).

    Args:
        time_data (torch.Tensor): Time-domain data.
        label_tensor (torch.Tensor): Ground truth labels.
        current_centers (torch.Tensor): Current class centers.
        require_grad (bool): Whether to compute gradients.

    Returns:
        tuple: A tuple containing the model predictions, embeddings, and loss.
    """

    if require_grad:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(require_grad):
        pred, emb = model(time_data)

        if not cfg.OPENSET_ENABLE:
            assert emb is None, "Embedding should be None when OPENSET_ENABLE is False"

        loss = criterion(pred, label_tensor, emb, current_centers)

        if require_grad:
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.anchors.data.clamp_(0.0, 1.0)

    return pred, emb, loss


def _collect_prediction_metrics(pred_r, tgt_r):
    """Collect prediction and ground truth metrics.

    Args:
        pred_r (torch.Tensor): Model predictions.
        tgt_r (torch.Tensor): Ground truth labels.

    Returns:
        tuple: A tuple containing the number of true-positive predictions,
        false positives, false negatives, and the number of ground-truth
        signals that were detected at least once.
    """
    bsize = pred_r.size(0)
    tp = fp = fn = matched = 0
    for i in range(bsize):
        preds, gts = _collect_lists(pred_r, tgt_r, i)
        tpi, fpi, fni, _, match = _tp_fp_fn(preds, gts)
        tp += tpi
        fp += fpi
        fn += fni
        matched += match
    return tp, fp, fn, matched


def _analyze_batch(pred, label_tensor, emb, stats):
    """Analyze a single batch and update statistics.

    Args:
        pred (torch.Tensor): Model predictions.
        label_tensor (torch.Tensor): Ground truth labels.
        emb (torch.Tensor): Embedding features.
        stats (dict): Dictionary to accumulate statistics.

    Returns:
        dict: Updated statistics dictionary.
    """

    bsize = pred.size(0)
    preds = pred.view(bsize, cfg.S, cfg.B, 1 + 1 + 1 + cfg.NUM_CLASSES)
    tgts = label_tensor.view_as(preds)

    counts = _collect_prediction_metrics(preds, tgts)
    stats["tp"] += counts[0]
    stats["fp"] += counts[1]
    stats["fn"] += counts[2]
    stats["matched"] += counts[3]

    obj_mask = tgts[..., 1] > 0

    pred_class_idx = preds[..., 3:].argmax(dim=-1)
    if cfg.OPENSET_ENABLE and cfg.OPENSET_THRESHOLD is not None:
        pred_class_idx = _apply_openset(pred_class_idx, emb)

    true_class_idx = tgts[..., 3:].argmax(dim=-1)
    if cfg.OPENSET_ENABLE:
        gt_unknown_mask = tgts[..., 3:].sum(dim=-1) == 0
        true_class_idx[gt_unknown_mask] = cfg.UNKNOWN_IDX

    correct_cls_mask = pred_class_idx == true_class_idx

    stats["obj_count"] += obj_mask.sum().item()
    stats["correct_cls"] += correct_cls_mask[obj_mask].sum().item()

    freq_pred = preds[..., 0]
    # Prepare tensor to hold per-object frequency error
    freq_err = torch.zeros_like(freq_pred)

    # For each item in batch, compute freq error using closest prediction
    for i in range(bsize):
        gt_mask = obj_mask[i]
        if not gt_mask.any():
            continue

        # Flatten predictions and ground truths for easier indexing
        pred_offsets = freq_pred[i].view(-1)
        tgts_flat = tgts[i].view(-1, tgts.size(-1))
        gt_mask_flat = gt_mask.view(-1)

        gt_offsets = tgts_flat[gt_mask_flat, 0]

        diff = (pred_offsets.unsqueeze(0) - gt_offsets.unsqueeze(1)).abs()
        min_vals, _ = diff.min(dim=1)

        freq_err[i][gt_mask] = min_vals * cfg.CELL_WIDTH
        stats["sum_freq_err"] += (min_vals * cfg.CELL_WIDTH).sum().item()

    return {
        "bsize": bsize,
        "obj_mask": obj_mask,
        "freq_err": freq_err,
        "correct_cls_mask": correct_cls_mask,
        "preds": preds,
        "tgts": tgts,
        "pred_class_idx": pred_class_idx,
        "true_class_idx": true_class_idx,
    }


def _apply_openset(pred_class_idx, emb):
    """Apply open-set thresholding to predictions.

    Args:
        pred_class_idx (torch.Tensor): Predicted class indices.
        emb (torch.Tensor): Embedding features.

    Returns:
        torch.Tensor: Updated predicted class indices.
    """

    flat_idx = pred_class_idx.reshape(-1)
    means_sel = model.class_means[flat_idx].to(device)
    inv_cov_sel = model.inv_cov[flat_idx].to(device)
    d = _mahalanobis_dist(
        emb.reshape(-1, cfg.EMBED_DIM), means_sel, inv_cov_sel
    ).view_as(pred_class_idx)
    tau = cfg.OPENSET_THRESHOLD.to(device)[pred_class_idx]
    unknown_mask_pred = d > tau
    pred_class_idx[unknown_mask_pred] = cfg.UNKNOWN_IDX
    return pred_class_idx


def _accumulate_embeddings(emb, obj_mask, true_class_idx, emb_acc):
    """Accumulate embeddings for each class.

    Args:
        emb (torch.Tensor): Embedding features.
        obj_mask (torch.Tensor): Object mask indicating presence of objects.
        true_class_idx (torch.Tensor): Ground truth class indices.
        emb_acc (list): List of tensors containing embeddings for each class.
    """

    with torch.no_grad():
        embs_this = emb[obj_mask].cpu()
        labels_this = true_class_idx[obj_mask].cpu()
        for c in range(cfg.NUM_CLASSES):
            idx = labels_this == c
            if idx.any():
                emb_acc[c].append(embs_this[idx])


def _gather_feature_vectors(
    emb, obj_mask, labels, snrs, store_feat, store_labels, snr_range
):
    """Collect embedding vectors for later plotting."""

    snr_range_start = snr_range[0]
    snr_range_end = snr_range[1]

    # Ensure obj_mask is on CPU for indexing
    obj_mask = obj_mask.cpu()

    with torch.no_grad():
        vecs = emb[obj_mask]
        labs = labels[obj_mask]

        if snrs is not None:
            snr_vals = snrs[:, None, None].expand_as(obj_mask)[obj_mask]
            valid = (snr_vals >= snr_range_start) & (snr_vals <= snr_range_end)
            vecs = vecs[valid]
            labs = labs[valid]

        vecs = vecs.cpu().numpy()
        labs = labs.cpu().numpy()
        for v, l in zip(vecs, labs):
            if len(store_feat) >= cfg.NUMBER_OF_FEATURES_PLOTTED:
                break
            store_feat.append(v)
            store_labels.append(int(l))


def _plot_sample_data(time_data, snr_val, gt_lines, pred_lines):
    """Generate PSD plot for a single sample.

    Args:
        time_data (torch.Tensor): Time-domain data for the sample.
        snr_val (float): Signal-to-noise ratio for the sample.
        gt_lines (list): Ground truth lines for the sample.
        pred_lines (list): Predicted lines for the sample.
    """
    i_data, q_data = time_data.cpu().numpy()
    x = np.fft.fft(i_data + 1j * q_data)
    x = np.fft.fftshift(x)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(i_data), d=1 / cfg.SAMPLING_FREQUENCY))
    power_spectral_density = 10 * np.log10(np.abs(x) ** 2 + 1e-12)
    freqs_ext = np.concatenate((freqs, freqs + cfg.SAMPLING_FREQUENCY))
    psd_ext = np.concatenate((power_spectral_density, power_spectral_density))
    idx_pos = freqs_ext >= 0
    freqs_fin = freqs_ext[idx_pos]
    psd_fin = psd_ext[idx_pos]
    psd_fin = psd_fin[freqs_fin <= cfg.SAMPLING_FREQUENCY]
    freqs_fin = freqs_fin[freqs_fin <= cfg.SAMPLING_FREQUENCY]
    if freqs_fin.size and psd_fin.size:
        _draw_test_sample(freqs_fin, psd_fin, snr_val, gt_lines, pred_lines)


def _collect_plot_lines(idx, batch_info, write):
    """Collect plot lines for a given sample.

    Args:
        idx (int): Index of the sample.
        batch_info (dict): Dictionary containing batch information.
        write (bool): Whether to write results to disk.

    Returns:
        tuple: A tuple containing ground truth lines and predicted lines.
    """

    preds = batch_info["preds"]
    tgts = batch_info["tgts"]
    pred_class_idx = batch_info["pred_class_idx"]
    true_class_idx = batch_info["true_class_idx"]

    output = {"pred_list": [], "gt_list": [], "gt_lines": [], "pred_lines": []}
    for s_idx in range(cfg.S):
        for b_idx in range(cfg.B):
            if tgts[idx, s_idx, b_idx, 1] > 0:
                freq = (s_idx + tgts[idx, s_idx, b_idx, 0].item()) * cfg.CELL_WIDTH
                desc = _convert_to_readable(
                    freq, int(true_class_idx[idx, s_idx, b_idx])
                )
                if write:
                    output["gt_list"].append(desc)
                output["gt_lines"].append((freq, desc[1]))
            if preds[idx, s_idx, b_idx, 1] > cfg.CONFIDENCE_THRESHOLD:
                freq_p = (s_idx + preds[idx, s_idx, b_idx, 0].item()) * cfg.CELL_WIDTH
                desc = _convert_to_readable(
                    freq_p, int(pred_class_idx[idx, s_idx, b_idx])
                )
                if write:
                    output["pred_list"].append(
                        (desc[0], desc[1], preds[idx, s_idx, b_idx, 1].item())
                    )
                output["pred_lines"].append(
                    (
                        freq_p,
                        desc[1],
                        preds[idx, s_idx, b_idx, 2].item() * cfg.CELL_WIDTH,
                    )
                )
    return (
        output["pred_list"],
        output["gt_list"],
        output["gt_lines"],
        output["pred_lines"],
    )


def _process_test_batch(snrs, batch_info, phase):
    """Process a single test batch.

    Args:
        snrs (list): List of SNR values for the batch.
        batch_info (dict): Dictionary containing batch information.
        phase (dict): Dictionary containing phase information.
    """

    time_data = batch_info["time_data"]
    obj_mask = batch_info["obj_mask"]
    freq_err = batch_info["freq_err"]
    correct_cls_mask = batch_info["correct_cls_mask"]
    preds = batch_info["preds"]
    tgts = batch_info["tgts"]

    for i in range(batch_info["bsize"]):
        snr_val = snrs[i]
        if snr_val not in phase["snr_classes"]:
            phase["snr_classes"][snr_val] = {"true": [], "pred": []}
        phase["snr_classes"][snr_val]["true"].extend(
            batch_info["true_class_idx"][i][obj_mask[i]].cpu().numpy().tolist()
        )
        phase["snr_classes"][snr_val]["pred"].extend(
            batch_info["pred_class_idx"][i][obj_mask[i]].cpu().numpy().tolist()
        )
        if snr_val not in phase["snr_dicts"]["tp"]:
            phase["snr_dicts"]["tp"][snr_val] = 0
            phase["snr_dicts"]["fp"][snr_val] = 0
            phase["snr_dicts"]["fn"][snr_val] = 0
            phase["snr_dicts"]["matched"][snr_val] = 0
        if snr_val not in phase["snr_dicts"]["obj_count"]:
            phase["snr_dicts"]["obj_count"][snr_val] = 0
            phase["snr_dicts"]["correct_cls"][snr_val] = 0
            phase["snr_dicts"]["freq_err"][snr_val] = 0.0

        if obj_mask[i].sum() > 0:
            phase["snr_dicts"]["obj_count"][snr_val] += obj_mask[i].sum().item()
            phase["snr_dicts"]["correct_cls"][snr_val] += (
                correct_cls_mask[i][obj_mask[i]].sum().item()
            )
            phase["snr_dicts"]["freq_err"][snr_val] += (
                freq_err[i][obj_mask[i]].sum().item()
            )

        pred_boxes, gt_boxes = _collect_lists(preds, tgts, i)
        tpi, fpi, fni, iou_sum, match = _tp_fp_fn(pred_boxes, gt_boxes)
        phase["snr_dicts"]["tp"][snr_val] += tpi
        phase["snr_dicts"]["fp"][snr_val] += fpi
        phase["snr_dicts"]["fn"][snr_val] += fni
        phase["snr_dicts"]["matched"][snr_val] += match
        if snr_val not in phase["snr_dicts"].get("iou", {}):
            phase["snr_dicts"].setdefault("iou", {})[snr_val] = 0.0
        phase["snr_dicts"]["iou"][snr_val] += iou_sum

        if phase["write"] or cfg.PLOT_TEST_SAMPLES > 0:
            pred_list, gt_list, gt_lines, pred_lines = _collect_plot_lines(
                i, batch_info, phase["write"]
            )
            if cfg.PLOT_TEST_SAMPLES > 0:
                _plot_sample_data(time_data[i], snr_val, gt_lines, pred_lines)
                cfg.PLOT_TEST_SAMPLES -= 1
            if phase["write"]:
                phase["entries"].append((snr_val, len(gt_list), pred_list, gt_list))


def _compute_epoch_stats(total_loss, loader_len, stats):
    """Compute summary statistics for a phase.

    Args:
        total_loss (float): Total loss for the epoch.
        loader_len (int): Number of batches in the DataLoader.
        stats (dict): Dictionary containing accumulated statistics.

    Returns:
        tuple: A tuple containing average loss, mean frequency error, classification accuracy,
               precision, recall, and F1 score for the phase.
    """

    obj_count = stats["obj_count"]
    mean_freq_err = stats["sum_freq_err"] / obj_count if obj_count else 0.0
    cls_accuracy = 100.0 * stats["correct_cls"] / obj_count if obj_count else 0.0
    precision = (
        stats["tp"] / (stats["tp"] + stats["fp"])
        if (stats["tp"] + stats["fp"])
        else 0.0
    )
    recall = (
        stats["matched"] / (stats["matched"] + stats["fn"])
        if (stats["matched"] + stats["fn"])
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    avg_loss = total_loss / loader_len
    return avg_loss, mean_freq_err, cls_accuracy, precision, recall, f1


def _update_class_means(emb_acc):
    """Update class mean embeddings and covariance matrices.

    Args:
        emb_acc (list): List of tensors containing embeddings for each class.
    """

    with torch.no_grad():
        model.class_means = torch.zeros(cfg.NUM_CLASSES, cfg.EMBED_DIM)
        model.inv_cov = (
            torch.eye(cfg.EMBED_DIM).unsqueeze(0).repeat(cfg.NUM_CLASSES, 1, 1)
        )
        tau_vec = torch.zeros(cfg.NUM_CLASSES)

        for c, lst in enumerate(emb_acc):
            if not lst:
                continue
            m = torch.cat(lst, 0)
            mean_c = m.mean(0)
            model.class_means[c] = mean_c
            diff = m - mean_c

            if diff.size(0) > 1:
                cov = diff.t().mm(diff) / (diff.size(0) - 1)
            else:
                cov = torch.eye(cfg.EMBED_DIM)
            cov += torch.eye(cfg.EMBED_DIM) * 1e-6
            model.inv_cov[c] = torch.inverse(cov)

            dist_sq = torch.einsum("ni,ij,nj->n", diff, model.inv_cov[c], diff)
            dist_c = torch.sqrt(dist_sq + 1e-6)
            tau_vec[c] = torch.quantile(dist_c, cfg.OPENSET_COVERAGE).item()

        cfg.OPENSET_THRESHOLD = tau_vec
        model.class_means = model.class_means.to(device)
        model.inv_cov = model.inv_cov.to(device)


def _report_test_results(metrics, phase):
    """Print and write test set statistics.

    Args:
        metrics (dict): Dictionary containing computed metrics.
        phase (dict): Dictionary containing phase information.
    """

    overall_freq_err = _convert_to_readable(metrics["mean_freq_err"], 0)[0]

    print("\n=== TEST SET RESULTS ===")
    print("\nOverall Performance:")
    print(f"\tClassification Accuracy: {metrics['cls_accuracy']:.2f}%")
    print(f"\tMean Frequency Error: {overall_freq_err}")
    print(f"\tPrecision: {metrics['precision']:.3f}")
    print(f"\tRecall: {metrics['recall']:.3f}")
    print(f"\tF1 score: {metrics['f1']:.3f}\n")

    # Aggregate metrics for signal-dominated scenarios (SNR > 0 dB)
    pos_snr_keys = [snr for snr in phase["snr_dicts"]["obj_count"] if snr > 0]
    if pos_snr_keys:
        obj_total = sum(phase["snr_dicts"]["obj_count"][k] for k in pos_snr_keys)
        correct_total = sum(phase["snr_dicts"]["correct_cls"][k] for k in pos_snr_keys)
        freq_err_total = sum(phase["snr_dicts"]["freq_err"][k] for k in pos_snr_keys)
        tp_total = sum(phase["snr_dicts"]["tp"].get(k, 0) for k in pos_snr_keys)
        fp_total = sum(phase["snr_dicts"]["fp"].get(k, 0) for k in pos_snr_keys)
        fn_total = sum(phase["snr_dicts"]["fn"].get(k, 0) for k in pos_snr_keys)
        matched_total = sum(
            phase["snr_dicts"]["matched"].get(k, 0) for k in pos_snr_keys
        )

        sigdom_acc = 100.0 * correct_total / obj_total if obj_total else 0.0
        sigdom_freq_err = freq_err_total / obj_total if obj_total else 0.0
        sigdom_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
        sigdom_rec = (
            matched_total / (matched_total + fn_total)
            if (matched_total + fn_total)
            else 0.0
        )
        sigdom_f1 = (
            2 * sigdom_prec * sigdom_rec / (sigdom_prec + sigdom_rec)
            if (sigdom_prec + sigdom_rec)
            else 0.0
        )

        sigdom_freq_err = _convert_to_readable(sigdom_freq_err, 0)[0]
        print("Signal Dominated Performance (SNR > 0dB):")
        print(f"\tClassification Accuracy: {sigdom_acc:.2f}%")
        print(f"\tMean Frequency Error: {sigdom_freq_err}")
        print(f"\tPrecision: {sigdom_prec:.3f}")
        print(f"\tRecall: {sigdom_rec:.3f}")
        print(f"\tF1 score: {sigdom_f1:.3f}\n")

    print("Per-SNR Performance:")

    snr_vals_sorted = sorted(phase["snr_dicts"]["obj_count"].keys())
    acc_list = []
    f1_list = []
    prec_list = []
    rec_list = []
    iou_list = []
    for snr_val in snr_vals_sorted:
        cls_acc_snr = (
            100.0
            * phase["snr_dicts"]["correct_cls"][snr_val]
            / phase["snr_dicts"]["obj_count"][snr_val]
        )
        freq_err_snr = (
            phase["snr_dicts"]["freq_err"][snr_val]
            / phase["snr_dicts"]["obj_count"][snr_val]
        )
        freq_err_snr = _convert_to_readable(freq_err_snr, 0)[0]

        for key in ("tp", "fp", "fn", "matched", "iou"):
            if snr_val not in phase["snr_dicts"][key]:
                phase["snr_dicts"][key][snr_val] = 0

        prec = (
            phase["snr_dicts"]["tp"][snr_val]
            / (phase["snr_dicts"]["tp"][snr_val] + phase["snr_dicts"]["fp"][snr_val])
            if (phase["snr_dicts"]["tp"][snr_val] + phase["snr_dicts"]["fp"][snr_val])
            else 0.0
        )
        rec = (
            phase["snr_dicts"]["matched"][snr_val]
            / (
                phase["snr_dicts"]["matched"][snr_val]
                + phase["snr_dicts"]["fn"][snr_val]
            )
            if (
                phase["snr_dicts"]["matched"][snr_val]
                + phase["snr_dicts"]["fn"][snr_val]
            )
            else 0.0
        )
        f1_snr = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou_snr = (
            phase["snr_dicts"]["iou"][snr_val] / phase["snr_dicts"]["tp"][snr_val]
            if phase["snr_dicts"]["tp"][snr_val]
            else 0.0
        )

        print(
            f"\tSNR {snr_val:.1f}:  "
            f"Accuracy={cls_acc_snr:.2f}%,  FreqErr={freq_err_snr},  "
            f"P={prec:.3f}, R={rec:.3f}, F1={f1_snr:.3f}, IoU={iou_snr:.3f}"
        )
        acc_list.append(cls_acc_snr / 100.0)
        f1_list.append(f1_snr)
        prec_list.append(prec)
        rec_list.append(rec)
        iou_list.append(iou_snr)

    _plot_snr_metrics(
        snr_vals_sorted,
        acc_list,
        f1_list,
        prec_list,
        rec_list,
        iou_list,
        out_dir,
    )

    if cfg.GENERATE_CONFUSION_MATRIX:
        _draw_confusion_matrix(
            phase["overall_true_classes"], phase["overall_pred_classes"], out_dir
        )

        # Signal Dominated Confusion Matrix (>0dB)
        true_class = [
            cls
            for snr in phase["snr_classes"]
            if snr > 0
            for cls in phase["snr_classes"][snr]["true"]
        ]
        pred_class = [
            cls
            for snr in phase["snr_classes"]
            if snr > 0
            for cls in phase["snr_classes"][snr]["pred"]
        ]
        _draw_confusion_matrix(true_class, pred_class, out_dir, signal_dominated=True)

    if phase["write"] and phase["entries"] is not None:
        phase["entries"].sort(key=lambda x: x[0], reverse=True)
        with open(
            os.path.join(out_dir, "test_results.txt"), "w", encoding="utf-8"
        ) as f:
            for snr_val, ntx, pred_list, gt_list in phase["entries"]:
                f.write(f"SNR {snr_val:.1f}; Center_freqs = {ntx}\n")
                f.write(f"  Predicted => {pred_list}\n")
                f.write(f"  GroundTruth=> {gt_list}\n\n")


def _run_phase(loader, mode, epoch=None, plot=0, write=False):
    """Run a training or testing phase.

    Args:
        loader (DataLoader): DataLoader for the current phase.
        mode (str): "train" or "test".
        epoch (int, optional): Current epoch number. Defaults to None.
        plot (int, optional): Number of samples to plot results. Defaults to 0.
        write (bool, optional): Whether to write results to disk. Defaults to False.

    Returns:
        tuple: A tuple containing average loss, mean frequency error, classification accuracy,
               precision, recall, and F1 score for the phase.
    """

    phase = _setup_phase(mode, plot, write)
    phase["stats"] = {
        "obj_count": 0,
        "sum_freq_err": 0.0,
        "correct_cls": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "matched": 0,
        "loss": 0.0,
    }

    desc = (
        f"{mode.capitalize()} epoch {epoch+1}/{cfg.EPOCHS}"
        if epoch is not None
        else mode.capitalize()
    )

    for batch in tqdm(loader, desc=desc):
        data = _prepare_data(batch)

        outputs = _forward_pass(
            data[0],  # Time
            data[1],  # Label
            phase["current_centers"],
            phase["require_grad"],
        )
        phase["stats"]["loss"] += outputs[2].item()

        batch_info = _analyze_batch(outputs[0], data[1], outputs[1], phase["stats"])

        if phase["is_train"] and cfg.OPENSET_ENABLE:
            _accumulate_embeddings(
                outputs[1],
                batch_info["obj_mask"],
                batch_info["true_class_idx"],
                phase["emb_acc"],
            )
        if cfg.PLOT_FEATURE_DISTRIBUTION and (
            (phase["is_test"]) or (epoch == cfg.EPOCHS - 1)
        ):
            # Only gather feature vectors for the last epoch or during testing.
            for snr_range, store in feature_store.items():
                store_feat = (
                    store["train_feat"] if phase["is_train"] else store["test_feat"]
                )
                store_labels = (
                    store["train_labels"] if phase["is_train"] else store["test_labels"]
                )
                if len(store_feat) >= cfg.NUMBER_OF_FEATURES_PLOTTED:
                    continue
                _gather_feature_vectors(
                    outputs[1],
                    batch_info["obj_mask"],
                    batch_info["true_class_idx"],
                    data[2],
                    store_feat,
                    store_labels,
                    snr_range,
                )

        if phase["is_test"]:
            phase["overall_true_classes"].extend(
                batch_info["true_class_idx"][batch_info["obj_mask"]]
                .cpu()
                .numpy()
                .tolist()
            )
            phase["overall_pred_classes"].extend(
                batch_info["pred_class_idx"][batch_info["obj_mask"]]
                .cpu()
                .numpy()
                .tolist()
            )
            batch_info["time_data"] = data[0]

            _process_test_batch(data[2].numpy(), batch_info, phase)

        # Explicitly release tensor references before the next iteration to
        # avoid holding onto GPU memory across batches. This helps prevent
        # residual allocations from accumulating between epochs.
        del data, outputs, batch_info
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    metrics = _compute_epoch_stats(phase["stats"]["loss"], len(loader), phase["stats"])

    if phase["is_train"] and cfg.OPENSET_ENABLE:
        _update_class_means(phase["emb_acc"])

    if phase["is_train"] and cfg.DETAILED_LOSS_PRINT:
        criterion.print_epoch_stats()
        criterion.reset_epoch_stats()

    if phase["is_test"]:
        _report_test_results(
            {
                "cls_accuracy": metrics[2],
                "mean_freq_err": metrics[1],
                "precision": metrics[3],
                "recall": metrics[4],
                "f1": metrics[5],
            },
            phase,
        )

    return metrics


def _mahalanobis_dist(x, mean, inv_cov):
    """Calculate the Mahalanobis distance."""
    diff = x - mean
    dist_sq = torch.einsum("bi,bij,bj->b", diff, inv_cov, diff)
    return torch.sqrt(dist_sq + 1e-6)


def _convert_to_readable(frequency, modclass):
    """Convert frequency and modulation class to human-readable format.

    Args:
        frequency (float): Frequency in Hz.
        modclass (int): Modulation class index.

    Returns:
        tuple: Human-readable frequency and modulation class.
    """

    if frequency > 1000:
        size_map = {
            1: "Hz",
            1000: "kHz",
            1000000: "MHz",
            1000000000: "GHz",
            1000000000000: "THz",
        }
        for size in size_map:
            if frequency < size:
                frequency /= size / 1000
                break
        frequency = round(frequency, 4)
        frequency_string = f"{frequency} {size_map[int(size/1000)]}"
    else:
        frequency_string = f"{frequency} Hz"

    if modclass < len(cfg.MODULATION_CLASSES):
        modclass_str = cfg.MODULATION_CLASSES[modclass]
    else:
        modclass_str = cfg.UNKNOWN_CLASS_NAME

    return frequency_string, modclass_str


def _collect_lists(pred_r, tgt_r, index):
    """Collect predicted and target frequency and bandwidth lists.

    Args:
        pred_r (torch.Tensor): Prediction tensor.
        tgt_r (torch.Tensor): Target tensor.
        index (int): Frame index.

    Returns:
        tuple: (pred, gt) lists for the specified frame.
    """
    x_pred = pred_r[..., 0][index].cpu()
    conf_pred = pred_r[..., 1][index].cpu()
    bw_pred = pred_r[..., 2][index].cpu()

    x_tgt = tgt_r[..., 0][index].cpu()
    conf_tgt = tgt_r[..., 1][index].cpu()
    bw_tgt = tgt_r[..., 2][index].cpu()

    pred, gt = [], []
    for s in range(cfg.S):
        for b in range(cfg.B):
            if conf_pred[s, b] > cfg.CONFIDENCE_THRESHOLD:
                f = (x_pred[s, b] + s) * cfg.CELL_WIDTH
                bw = bw_pred[s, b] * cfg.CELL_WIDTH
                pred.append((f, bw))

            if conf_tgt[s, b] > 0:
                f = (x_tgt[s, b] + s) * cfg.CELL_WIDTH
                bw = bw_tgt[s, b] * cfg.CELL_WIDTH
                gt.append((f, bw))
    return pred, gt


def _tp_fp_fn(pred, gt):
    """Calculate true/false positives/negatives and IoU sum.

    Args:
        pred (list): List of predicted boxes ``[(freq, bw), ...]``.
        gt (list): List of ground truth boxes ``[(freq, bw), ...]``.

    Returns:
        tuple: ``(tp_pred, fp, fn, iou_sum, matched_gt)`` where ``tp_pred`` is
        the number of predictions overlapping any ground truth, ``fn`` is the
        number of ground-truth signals with no corresponding prediction, and
        ``matched_gt`` is the number of ground-truth signals that were detected
        at least once. ``iou_sum`` is the accumulated intersection-over-union of
        all matched predictions.
    """

    counts = {"tp": 0, "fp": 0, "fn": 0}
    iou_sum = 0.0
    matched = [False] * len(gt)

    for f_pred, bw_pred in pred:
        p_low = f_pred - bw_pred / 2.0
        p_high = f_pred + bw_pred / 2.0
        best_idx = None
        best_iou = 0.0

        for i, (f_gt, bw_gt) in enumerate(gt):
            g_low = f_gt - bw_gt / 2.0
            g_high = f_gt + bw_gt / 2.0

            inter = min(p_high, g_high) - max(p_low, g_low)
            if inter > 0:
                union = max(p_high, g_high) - min(p_low, g_low)
                iou = inter / max(union, 1e-6)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

        if best_idx is not None:
            counts["tp"] += 1
            iou_sum += best_iou
            matched[best_idx] = True
        else:
            counts["fp"] += 1

    counts["fn"] = matched.count(False)
    matched_gt = len(gt) - counts["fn"]
    return counts["tp"], counts["fp"], counts["fn"], iou_sum, matched_gt


def _draw_confusion_matrix(
    overall_true_classes, overall_pred_classes, base_dir=".", signal_dominated=False
):
    """Draws a confusion matrix based on the true and predicted classes.

    Args:
        overall_true_classes (list): List of true class labels.
        overall_pred_classes (list): List of predicted class labels.
    """

    if cfg.OPENSET_ENABLE:
        class_list = cfg.MODULATION_CLASSES + [cfg.UNKNOWN_CLASS_NAME]
    else:
        class_list = cfg.MODULATION_CLASSES

    cm = confusion_matrix(
        overall_true_classes,
        overall_pred_classes,
        labels=range(len(class_list)),
    )
    cm_percent = cm.astype(float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percent[i] = (cm[i] / row_sum) * 100.0

    plt.figure(dpi=1000, figsize=(8, 6))
    heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_list,
        yticklabels=class_list,
    )
    title = (
        "Full Test Set Confusion Matrix (%)"
        if not signal_dominated
        else "Signal Dominated Confusion Matrix (%)"
    )
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    if not signal_dominated:
        fname = os.path.join(base_dir, "confusion_matrix.pdf")
    else:
        fname = os.path.join(base_dir, "confusion_matrix_SIGDOM.pdf")

    plt.savefig(fname)
    plt.close()


def _plot_snr_metrics(snrs, accs, f1s, precs, recs, ious, base_dir="."):
    """Plot performance metrics versus SNR."""
    plt.figure(dpi=1000, figsize=(8, 6))
    plt.plot(snrs, accs, marker="o", label="Accuracy")
    plt.plot(snrs, precs, marker="s", label="Precision")
    plt.plot(snrs, recs, marker="^", label="Recall")
    plt.plot(snrs, f1s, marker="x", label="F1 Score")
    plt.plot(snrs, ious, marker="d", label="IoU")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Metric")
    plt.title("Model Performance vs SNR")
    plt.grid(True)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    fname = os.path.join(base_dir, "snr_performance.pdf")
    plt.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches="tight")
    plt.close()


def _plot_feature_distribution(
    train_f, train_l, test_f, test_l, base_dir=".", snr_range=None
):
    """Plot 2-D feature embeddings for train and test sets."""

    label_suffix = _snr_range_label(snr_range) if snr_range is not None else "All SNR"
    file_suffix = (
        f"_{_snr_range_file_suffix(snr_range)}" if snr_range is not None else ""
    )

    if len(train_f) == 0 or len(test_f) == 0:
        print(f"Skipping t-SNE plot for {label_suffix}: insufficient features.")
        return

    all_f = np.vstack([train_f, test_f])
    perplexity = min(cfg.TSNE_PERPLEXITY, max(1, all_f.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        random_state=0,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
    )

    proj = tsne.fit_transform(all_f)
    train_p = proj[: len(train_f)]
    test_p = proj[len(train_f) :]

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(dpi=1000, figsize=(8, 6))
    for c in range(cfg.NUM_CLASSES):
        idx = np.array(train_l) == c
        if not np.any(idx):
            continue
        pts = train_p[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=cfg.MODULATION_CLASSES[c],
            color=colors[c % len(colors)],
            marker="o",
            alpha=0.7,
        )
    ax.set_title(f"Training Feature Distribution ({label_suffix})")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    lgd = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0.0)
    plt.savefig(
        os.path.join(base_dir, f"train_feature_distribution{file_suffix}.pdf"),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(dpi=1000, figsize=(8, 6))
    for c in range(cfg.NUM_CLASSES):
        idx = np.array(test_l) == c
        if np.any(idx):
            ax.scatter(
                test_p[idx, 0],
                test_p[idx, 1],
                label=cfg.MODULATION_CLASSES[c],
                color=colors[c % len(colors)],
                marker="o",
                alpha=0.7,
            )
    idx_u = np.array(test_l) == cfg.UNKNOWN_IDX
    if np.any(idx_u):
        ax.scatter(
            test_p[idx_u, 0],
            test_p[idx_u, 1],
            label=cfg.UNKNOWN_CLASS_NAME,
            color="black",
            marker="x",
            alpha=0.7,
        )
    ax.set_title(f"Test Feature Distribution ({label_suffix})")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.savefig(
        os.path.join(base_dir, f"test_feature_distribution{file_suffix}.pdf"),
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    plt.close(fig)


def _draw_test_sample(freqs, psd, snr, gt_lines, pred_lines):
    """Draws frequency spectrum with ground truth and predicted lines.

    Args:
        freqs (np.ndarray): Frequency bins.
        psd (np.ndarray): Power spectral density values.
        snr (float): Signal-to-noise ratio.
        gt_lines (list): Ground truth lines.
        pred_lines (list): Predicted lines.
    """

    plt.figure()
    plt.plot(freqs, psd)
    plt.title(f"SNR = {snr:.1f}; No. Transmitters = {len(gt_lines)}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("power_spectral_density [dB]")

    texts = []
    for fg, cls_g in gt_lines:
        plt.axvline(fg, linestyle="--", color="black", alpha=1.0)
        texts.append(plt.text(fg, psd.min(), f"GT:{cls_g}", va="top", ha="center"))

    for fp, cls_p, bandwidth in pred_lines:
        ax = plt.gca()
        ax.axvline(fp, linestyle="-", color="red", alpha=1.0)
        ax.axvspan(
            fp - bandwidth / 2, fp + bandwidth / 2, color="red", alpha=0.30, zorder=5
        )
        texts.append(plt.text(fp, psd.min(), f"P:{cls_p}", va="bottom", ha="center"))

    adjust_text(texts)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "frequency_domain_representations", f"{str(uuid4())}.pdf")
    )
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train and evaluate the CNN-Model for modulation classification."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Path to the directory containing the training dataset.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to the directory containing the testing dataset.",
    )
    args = parser.parse_args()

    main(train_dir=args.train_dir, test_dir=args.test_dir)
    sys.exit(0)
