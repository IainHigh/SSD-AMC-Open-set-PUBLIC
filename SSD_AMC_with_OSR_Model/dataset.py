# pylint: disable=import-error
"""
dataset.py:
Dataset class for the SSD AMC OSR model.
This class reads .sigmf-data files from a directory,
applies necessary transformations, and prepares the data for training.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from config import (
    S,  # number of grid cells
    B,  # boxes per cell
    NUM_CLASSES,
    SAMPLING_FREQUENCY,
)


class Dataset(TorchDataset):
    """
    Reads .sigmf-data files from a directory.
    For each file, it:
    - Loads the IQ data and computes its Fourier transform.
    - Constructs a label tensor with modulation classes and bandwidth.
    - Returns the time-domain IQ data, frequency-domain representation, label tensor,
      and SNR value.
    """

    def __init__(self, directory, transform=None, class_list=None):
        """Initialize the dataset.

        Args:
            directory (str): Path to the dataset directory.
            transform (callable, optional): A function/transform to apply to the samples.
            class_list (list, optional): List of modulation classes. Defaults to None.

        Raises:
            RuntimeError: If no .sigmf-data files are found.
        """
        super().__init__()
        self.directory = directory
        self.transform = transform

        # Gather all .sigmf-data files in this directory.
        self.files = [
            fname.replace(".sigmf-data", "")
            for fname in os.listdir(directory)
            if fname.endswith(".sigmf-data")
        ]
        self.files.sort()
        if len(self.files) == 0:
            raise RuntimeError(f"No .sigmf-data files found in {directory}!")

        # Build a label -> index mapping for classes.
        self.class_list = class_list or self._discover_mod_classes()
        self.class_to_idx = {c: i for i, c in enumerate(self.class_list)}

        # Determine num_samples from the first file.
        self.num_samples = self._find_num_samples(self.files[0])

    def _discover_mod_classes(self):
        """Discover modulation classes from the dataset.

        Returns:
            list: A list of unique modulation classes.
        """
        all_mods = set()
        for base in self.files:
            meta_path = os.path.join(self.directory, base + ".sigmf-meta")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ann = meta["annotations"][0]
            mod_list = ann["rfml_labels"]["modclass"]
            if isinstance(mod_list, str):
                mod_list = [mod_list]
            all_mods.update(mod_list)
            if len(all_mods) >= NUM_CLASSES:
                break

        return sorted(all_mods)

    def _find_num_samples(self, base):
        """Find the number of samples in a .sigmf-data file.

        Args:
            base (str): Base filename (without extension).

        Returns:
            int: The number of samples.
        """
        # Check the first file to see how many samples (2*N interleaved)
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        iq_data = np.fromfile(data_path, dtype=np.float32)
        return len(iq_data) // 2

    def get_num_samples(self):
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self.num_samples

    def __len__(self):
        """Get the number of files in the dataset.

        Returns:
            int: The number of files.
        """
        return len(self.files)

    def _get_time_data(self, data_path):
        """Extract time domain data from a .sigmf-data file.

        Args:
            data_path (str): Path to the .sigmf-data file.

        Returns:
            tuple: Time domain data.
        """
        # Load IQ data (time domain) in a temporary memory map.
        iq_map = np.memmap(
            data_path,
            dtype=np.float32,
            mode="r",
            shape=(2 * self.num_samples,),
        )
        iq_data = np.array(iq_map)
        del iq_map
        i_data = iq_data[0::2]
        q_data = iq_data[1::2]
        x_complex = i_data + 1j * q_data

        # Convert to real time-domain IQ: shape (2, N)
        x_real = x_complex.real.astype(np.float32)
        x_imag = x_complex.imag.astype(np.float32)
        x_wide = np.stack([x_real, x_imag], axis=0)  # shape (2, N)

        if self.transform:
            x_wide = self.transform(x_wide)

        return x_wide

    def _get_metadata(self, meta_path):
        """Extract metadata from a .sigmf-meta file.

        Args:
            meta_path (str): Path to the .sigmf-meta file.

        Returns:
            dict: A dictionary containing the extracted metadata.
        """
        # Load metadata.
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        snr_value = meta["annotations"][1]["channel"]["snr"]
        center_freqs = meta["annotations"][0]["center_frequencies"]
        mod_list = meta["annotations"][0]["rfml_labels"]["modclass"]

        sampling_rate = meta["annotations"][0]["sampling_rate"]  # Fs in Hz
        sps_list = meta["annotations"][1]["filter"]["sps"]
        beta = meta["annotations"][1]["filter"]["rolloff"]

        if isinstance(mod_list, str):
            mod_list = [mod_list]
        if isinstance(center_freqs, (float, int)):
            center_freqs = [center_freqs]
        if isinstance(sps_list, (int, float)):
            sps_list = [sps_list] * len(center_freqs)

        chan_bw = [(sampling_rate / sps) * (1.0 + beta) for sps in sps_list]

        # Normalise bandwidth to the width of one grid‑cell (“bin”)
        bin_width = (SAMPLING_FREQUENCY / 2) / S
        bw_norm = [(bw / bin_width) for bw in chan_bw]

        return snr_value, center_freqs, mod_list, bw_norm

    def _build_label(self, center_freqs, mod_list, bw_norm):
        """Build label tensor from metadata.

        Args:
            center_freqs (list): List of center frequencies.
            mod_list (list): List of modulation classes.
            bw_norm (list): List of normalized bandwidths.

        Returns:
            np.ndarray: label tensor.
        """

        # Build label: shape [S, B, 1+1+1+NUM_CLASSES]
        label_tensor = np.zeros((S, B, 1 + 1 + 1 + NUM_CLASSES), dtype=np.float32)

        # Obtain anchor values using linspace.
        anchor_values = np.linspace(1 / (B + 1), B / (B + 1), B, dtype=np.float32)

        for c_freq, m_str, bw_n in zip(center_freqs, mod_list, bw_norm):
            # Normalize frequency.
            freq_norm = c_freq / (SAMPLING_FREQUENCY / 2)  # in [0, 1]
            cell_idx = int(freq_norm * S)
            if cell_idx >= S:
                cell_idx = S - 1
            x_offset = (freq_norm * S) - cell_idx
            x_offset = np.clip(x_offset, 0.0, 1.0)

            # Find the anchor index that is closest to the computed offset.
            anchor_idx = int(np.argmin(np.abs(anchor_values - x_offset)))

            # If this anchor is already used, try to find a free one
            if label_tensor[cell_idx, anchor_idx, 1] == 1.0:
                free = np.where(label_tensor[cell_idx, :, 1] == 0)[0]
                if len(free) > 0:
                    # There are free anchors, use the first one.
                    anchor_idx = free[0]

            label_tensor[cell_idx, anchor_idx, 0] = x_offset
            label_tensor[cell_idx, anchor_idx, 1] = 1.0
            label_tensor[cell_idx, anchor_idx, 2] = bw_n  # bandwidth
            class_idx = self.class_to_idx.get(m_str, None)
            if class_idx is not None:
                label_tensor[cell_idx, anchor_idx, 3 + class_idx] = 1.0
        return label_tensor

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (time-domain IQ data, label tensor, SNR value)
        """
        base = self.files[idx]
        data_path = os.path.join(self.directory, base + ".sigmf-data")
        meta_path = os.path.join(self.directory, base + ".sigmf-meta")

        x_wide = self._get_time_data(data_path)
        snr_value, center_freqs, mod_list, bw_norm = self._get_metadata(meta_path)

        label_tensor = self._build_label(center_freqs, mod_list, bw_norm)

        # Return time-domain IQ, frequency-domain representation, label, and SNR.
        return (
            torch.from_numpy(x_wide),
            torch.from_numpy(label_tensor),
            torch.tensor(snr_value, dtype=torch.float32),
        )
