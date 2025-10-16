#!/usr/bin/python3
# pylint: disable=import-error
# pylint: disable=invalid-name
# pylint: disable=global-statement
"""
generator.py:
This script generates synthetic signals based on the provided configuration.
It supports various modulation schemes, channel types, and parameters.
The generated signals are saved in SigMF format for further analysis.
"""

# Standard library imports
import os
import argparse
import json
import ctypes

# Third-party imports
import numpy as np
from tqdm import tqdm

# Local application imports
from utils.sigmf_utils import save_sigmf, archive_sigmf
from utils.config_utils import map_config

BUF = 64
HALFBUF = BUF // 2

config = None
rng_seed = None
clinear = None
ctx = None


def _prepare_capture_params(cfg):
    """Prepare signal and channel parameters for each capture.

    Args:
        cfg (dict): Configuration dictionary containing parameters.

    Returns:
        tuple: A tuple containing the prepared signal parameters, channel parameters,
               channel type, maximum beta, and minimum samples per symbol.
    """
    sig_params_all = [
        (_sps, _beta, _delay, _dt)
        for _sps in cfg["symbol_rate"]
        for _beta in cfg["rrc_filter"]["beta"]
        for _delay in cfg["rrc_filter"]["delay"]
        for _dt in cfg["rrc_filter"]["dt"]
    ]
    idx = np.random.choice(len(sig_params_all), cfg["n_captures"])
    sig_params = [sig_params_all[_idx] for _idx in idx]

    idx = np.random.choice(len(cfg["channel_params"]), cfg["n_captures"])
    channel_params = [cfg["channel_params"][_idx] for _idx in idx]

    max_beta = max(sig_params, key=lambda x: x[1])[1]
    min_sps = min(sig_params, key=lambda x: x[0])[0]

    return sig_params, channel_params, max_beta, min_sps


def _select_center_frequencies(cfg, sampling_rate, min_sps, max_beta):
    """Select center frequencies for the capture.
    Either randomly or from a predefined list.

    Args:
        cfg (dict): Configuration dictionary containing parameters.
        sampling_rate (float): The sampling rate of the signal.
        min_sps (int): Minimum samples per symbol.
        max_beta (float): Maximum roll-off factor.

    Returns:
        list: A list of selected center frequencies.
    """

    if cfg["center_frequencies_random"]:
        lower_bound, upper_bound, n_max, prevent_overlap = cfg["center_frequencies"]
        margin = (
            (2 * (sampling_rate / min_sps) * (1 + max_beta))
            if (prevent_overlap != 0)
            else 0
        )
        n = np.random.randint(1, n_max + 1)

        attempts, max_attempts = 0, 1000
        center_frequencies = []
        while len(center_frequencies) < n and attempts < max_attempts:
            candidate = np.random.uniform(lower_bound, upper_bound)
            if all(abs(candidate - p) >= margin for p in center_frequencies):
                center_frequencies.append(candidate)
            attempts += 1
    else:
        center_frequencies = cfg["center_frequencies"]
    return center_frequencies


def _calculate_n_sym(n_samps, sps_value):
    """Calculate the number of symbols based on samples per symbol.

    Args:
        n_samps (int): The total number of samples.
        sps_value (int): The samples per symbol.

    Returns:
        int: The calculated number of symbols.
    """
    raw = int(np.ceil(n_samps / sps_value))
    if raw * sps_value > n_samps:
        n_sym = n_samps // sps_value
    else:
        n_sym = raw
    return max(n_sym, 1)


def _allocate_signal_buffers(n_sym, n_samps):
    """Allocate zero-filled ctypes buffers used for modulation.

    Args:
        n_sym (int): The number of symbols.
        n_samps (int): The total number of samples.

    Returns:
        tuple: A tuple containing the allocated signal buffers.
    """
    s = (ctypes.c_uint * n_sym)(*np.zeros(n_sym, dtype=int))
    smi = (ctypes.c_float * n_sym)(*np.zeros(n_sym))
    smq = (ctypes.c_float * n_sym)(*np.zeros(n_sym))
    xi = (ctypes.c_float * n_samps)(*np.zeros(n_samps))
    xq = (ctypes.c_float * n_samps)(*np.zeros(n_samps))
    yi = (ctypes.c_float * n_samps)(*np.zeros(n_samps))
    yq = (ctypes.c_float * n_samps)(*np.zeros(n_samps))
    return s, smi, smq, xi, xq, yi, yq


def _prepare_transmission(mods, sig_params, n_samps, verbose):
    """Prepare modulation and filtering parameters and generate baseband.

    Args:
        mods (list): List of modulation schemes.
        sig_params (list): List of signal parameters.
        n_samps (int): Total number of samples.
        verbose (bool): Verbosity flag.

    Returns:
        dict: A dictionary containing the prepared transmission parameters.
    """
    global rng_seed

    return_dict = {
        "modname": None,
        "order": None,
        "sps": None,
        "beta": None,
        "delay": None,
        "dt": None,
        "n_sym": None,
        "seed": None,
        "xi": None,
        "xq": None,
        "yi": None,
        "yq": None,
        "sps_val": None,
    }

    rng_seed += 1
    return_dict["seed"] = ctypes.c_int(rng_seed)

    mod = mods[0] if len(mods) == 1 else mods[np.random.randint(0, len(mods))]
    modtype = ctypes.c_int(mod[0])
    return_dict["order"] = ctypes.c_int(mod[1])
    return_dict["modname"] = mod[-1]

    rand_index = np.random.randint(0, len(sig_params))
    return_dict["sps_val"], beta_val, delay_val, dt_val = sig_params[rand_index]
    return_dict["sps"] = ctypes.c_int(return_dict["sps_val"])
    return_dict["beta"] = ctypes.c_float(beta_val)
    return_dict["delay"] = ctypes.c_uint(int(delay_val))
    return_dict["dt"] = ctypes.c_float(dt_val)
    return_dict["n_sym"] = _calculate_n_sym(n_samps, return_dict["sps"].value)

    (
        s,
        smi,
        smq,
        return_dict["xi"],
        return_dict["xq"],
        return_dict["yi"],
        return_dict["yq"],
    ) = _allocate_signal_buffers(return_dict["n_sym"], n_samps)

    clinear.linear_modulate(
        modtype,
        return_dict["order"],
        ctypes.c_int(return_dict["n_sym"]),
        s,
        smi,
        smq,
        verbose,
        return_dict["seed"],
    )
    ctx.rrc_tx(
        ctypes.c_int(return_dict["n_sym"]),
        return_dict["sps"],
        return_dict["delay"],
        return_dict["beta"],
        return_dict["dt"],
        smi,
        smq,
        return_dict["xi"],
        return_dict["xq"],
        verbose,
    )
    return return_dict


def _mix_signal(i_sig, q_sig, center_freq, t):
    """Mix a complex signal with a local oscillator.

    Args:
        i_sig (np.ndarray): In-phase signal component.
        q_sig (np.ndarray): Quadrature signal component.
        center_freq (float): Center frequency for mixing.
        t (np.ndarray): Time vector.

    Returns:
        tuple: Shifted in-phase and quadrature components.
    """
    freq_shift = np.exp(1j * 2 * np.pi * center_freq * t)
    i_shifted = i_sig * np.real(freq_shift) - q_sig * np.imag(freq_shift)
    q_shifted = i_sig * np.imag(freq_shift) + q_sig * np.real(freq_shift)
    return i_shifted, q_shifted


def _add_awgn_noise(i_total, q_total, snr):
    """Add AWGN noise to the aggregated signal.

    Args:
        i_total (np.ndarray): In-phase component of the signal.
        q_total (np.ndarray): Quadrature component of the signal.
        snr (float): Signal-to-noise ratio.

    Returns:
        tuple: Noisy in-phase and quadrature components.
    """
    nstd = np.power(10.0, -snr.value / 20.0)
    noise_i = np.random.normal(scale=nstd / np.sqrt(2), size=i_total.shape)
    noise_q = np.random.normal(scale=nstd / np.sqrt(2), size=q_total.shape)
    i_total = (i_total + noise_i).astype(np.float32)
    q_total = (q_total + noise_q).astype(np.float32)
    return i_total, q_total


def _create_metadata(meta_params):
    """Create metadata dictionary from the provided parameters.

    Args:
        meta_params (dict): Dictionary containing metadata parameters.

    Returns:
        dict: A dictionary containing the created metadata.
    """
    return {
        "modname": meta_params["mod_list"],
        "order": meta_params["order"].value,
        "n_samps": meta_params["n_samps"] - BUF,
        "sampling_rate": meta_params["sampling_rate"],
        "center_frequencies": meta_params["center_frequencies"],
        "channel_type": meta_params["channel_type"],
        "snr": meta_params["snr"].value,
        "filter_type": "rrc",
        "sps": meta_params["sps_list"],
        "delay": meta_params["delay"].value,
        "beta": meta_params["beta"].value,
        "dt": meta_params["dt"].value,
        "savepath": meta_params["savepath"],
        "savename": meta_params["savename"],
    }


def generate_linear():
    """Generate linear signals based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing signal parameters.
        rng_seed (int): Random seed for reproducibility.

    Raises:
        ValueError: If the configuration is invalid.
    """
    verbose = ctypes.c_int(config["verbose"])
    n_samps = config["n_samps"] + BUF
    sampling_rate = config["sampling_rate"]

    # Generate time vector for mixing
    t = np.arange(n_samps - BUF) / sampling_rate

    (
        sig_params,
        channel_params,
        max_beta,
        min_sps,
    ) = _prepare_capture_params(config)

    for i in tqdm(range(0, config["n_captures"]), desc="Generating Data"):
        center_frequencies = _select_center_frequencies(
            config, sampling_rate, min_sps, max_beta
        )

        mod_list, sps_list = [], []

        i_total = np.zeros(n_samps - BUF, dtype=np.float32)
        q_total = np.zeros(n_samps - BUF, dtype=np.float32)

        snr = ctypes.c_float(channel_params[i])
        for center_freq in center_frequencies:
            tx_data = _prepare_transmission(
                config["modulation"], sig_params, n_samps, verbose
            )
            modname = tx_data["modname"]
            order = tx_data["order"]
            beta = tx_data["beta"]
            delay = tx_data["delay"]
            dt = tx_data["dt"]
            xi = tx_data["xi"]
            xq = tx_data["xq"]
            sps_val = tx_data["sps_val"]

            mod_list.append(modname)
            sps_list.append(sps_val)

            i_sig = np.array(xi)[HALFBUF:-HALFBUF]
            q_sig = np.array(xq)[HALFBUF:-HALFBUF]

            # Draw and apply a unique phase offset per transmitter in baseband.
            phase_offset = np.random.uniform(-np.pi, np.pi)

            phase_rot = np.exp(1j * phase_offset)

            i_sig = i_sig * np.real(phase_rot) - q_sig * np.imag(phase_rot)
            q_sig = i_sig * np.imag(phase_rot) + q_sig * np.real(phase_rot)

            # Frequency-shift the phase-offset baseband to its carrier.
            i_shifted, q_shifted = _mix_signal(i_sig, q_sig, center_freq, t)

            i_total += i_shifted
            q_total += q_shifted

        # Add channel effects of AWGN.
        i_total, q_total = _add_awgn_noise(i_total, q_total, snr)

        metadata = _create_metadata(
            {
                "mod_list": mod_list,
                "order": order,
                "n_samps": n_samps,
                "sampling_rate": sampling_rate,
                "center_frequencies": center_frequencies,
                "channel_type": config["channel_type"],
                "snr": snr,
                "sps_list": sps_list,
                "delay": delay,
                "beta": beta,
                "dt": dt,
                "savepath": config["savepath"],
                "savename": config["savename"],
            }
        )

        save_sigmf(i_total, q_total, metadata, i)


if __name__ == "__main__":
    ## load c modules
    clinear = ctypes.CDLL(os.path.abspath("./cmodules/linear_modulate"))
    ctx = ctypes.CDLL(os.path.abspath("./cmodules/rrc_tx"))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to configuration file to use for data generation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory in which to save the generated dataset.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        nargs="?",
        help="Random seed for data generation.",
    )
    args = parser.parse_args()

    with open(args.config_file, encoding="utf-8") as f:
        config = json.load(f)

    # Print a copy of the loaded configuration for archiving of results.
    print(f"Contents of loaded configuration file ({args.config_file}):")
    print(json.dumps(config, indent=4))
    print("\n")

    # If a rng seed is provided, use it.
    if args.rng_seed is not None:
        rng_seed = args.rng_seed

    else:
        rng_seed = np.random.randint(0, 10000)
    np.random.seed(rng_seed)

    config = map_config(config, args.save_dir)

    ## Generate the data
    generate_linear()

    if config["archive"]:
        archive_sigmf(config["savepath"])
