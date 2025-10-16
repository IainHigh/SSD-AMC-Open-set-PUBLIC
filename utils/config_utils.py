# pylint: disable-all

## python imports
import numpy as np
import os
from datetime import datetime

from .maps import mod_str2int


def check_range(x, positive=True):
    start, stop, step = x
    if positive:
        assert all(i >= 0.0 for i in x)
    return (stop >= start) and (step <= (stop - start))


def map_config(config, dataset_dir):
    mapped = {}

    ## num samples
    assert isinstance(config["n_samps"], int), "n_samps must be an integer."
    assert config["n_samps"] > 0, "n_samps must be greater than zero."
    mapped["n_samps"] = config["n_samps"]

    ## num captures
    assert isinstance(config["n_captures"], int), "n_capures must be an integer."
    assert config["n_captures"] > 0, "n_captures must be greater than zero."
    mapped["n_captures"] = config["n_captures"]

    ## mods
    try:
        mapped["modulation"] = []
        for i in config["modulation"]:
            mapped["modulation"].append(mod_str2int[i])
    except ValueError:
        print("Invalid modulation scheme found.")

    ## symbol rate
    if isinstance(config["symbol_rate"], list):
        for i in config["symbol_rate"]:
            assert isinstance(i, int), "symbol_rate must be an integer."
            assert i > 0, "symbol_rate must be greater than zero."
        mapped["symbol_rate"] = config["symbol_rate"]
    elif isinstance(config["symbol_rate"], int):
        assert config["symbol_rate"] > 0, "symbol_rate must be greater than zero."
        mapped["symbol_rate"] = [config["symbol_rate"]]
    else:
        raise ValueError("Invalid symbol rate type.")

    ## Sampling rate
    assert isinstance(
        config["sampling_rate"], (int, float)
    ), "sampling_rate must be a number."
    assert config["sampling_rate"] > 0, "sampling_rate must be positive."
    mapped["sampling_rate"] = config["sampling_rate"]

    # Randomly generated center frequencies
    if "randomly_generated_center_frequencies" in config.keys():
        # Assert "center_frequencies" is not also provided
        assert (
            "center_frequencies" not in config.keys()
        ), "center_frequencies and randomly_generated_center_frequencies cannot both be provided."
        assert isinstance(
            config["randomly_generated_center_frequencies"], list
        ), "randomly_generated_center_frequencies must be a list."
        mapped["center_frequencies"] = config["randomly_generated_center_frequencies"]
        mapped["center_frequencies_random"] = True
    ## Center frequencies
    elif "center_frequencies" in config.keys():
        assert isinstance(
            config["center_frequencies"], list
        ), "center_frequencies must be a list."
        assert all(
            isinstance(f, (int, float)) and f > 0 for f in config["center_frequencies"]
        ), "All center frequencies must be positive numbers."
        mapped["center_frequencies_random"] = False
        mapped["center_frequencies"] = config["center_frequencies"]

    ## If center frequencies list is empty, default to a single frequency at half Nyquist
    if not mapped["center_frequencies"]:
        mapped["center_frequencies"] = [mapped["sampling_rate"] / 2]

    ## filters
    for i, f in enumerate(config["filter"]):
        if f["type"] in ["rrc", "gaussian"]:
            filter_type = f["type"] + "_filter"
            mapped[filter_type] = {}

            d = config["filter"]
            tmp = [i["type"] == f["type"] for i in d]
            d_i = np.where(tmp)[0][0]

            if "beta" in f.keys():
                tmp = f["beta"]

            if isinstance(tmp, list):
                assert check_range(tmp)
                mapped[filter_type]["beta"] = np.arange(
                    tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
                )
            elif isinstance(tmp, float) and tmp > 0.0:
                mapped[filter_type]["beta"] = [tmp]
            else:
                raise ValueError("Invalid filter beta.")

            if "dt" in f.keys():
                tmp = f["dt"]
            else:
                print("No filter dt provided. Using defaults.")
                tmp = d[d_i]["dt"]
            if isinstance(tmp, list):
                assert check_range(tmp)
                mapped[filter_type]["dt"] = np.arange(
                    tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
                )
            elif isinstance(tmp, float) and tmp >= 0.0:
                mapped[filter_type]["dt"] = [tmp]
            else:
                raise ValueError("Invalid filter dt.")

            if "delay" in f.keys():
                tmp = f["delay"]
            else:
                print("No filter delay provided. Using defaults")
                tmp = d[d_i]["delay"]
            if isinstance(tmp, list):
                assert check_range(tmp)
                mapped[filter_type]["delay"] = np.arange(
                    tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2]
                )
            elif isinstance(tmp, int) and tmp >= 0:
                mapped[filter_type]["delay"] = [tmp]
            else:
                raise ValueError("Invalid filter delay.")
        else:
            raise ValueError("Invalid filter type.")

    ## channel
    if "channel" in config.keys():
        if config["channel"]["type"] == "awgn":
            mapped["channel_type"] = "awgn"

            tmp = config["channel"]["snr"]

            if isinstance(tmp, list):
                assert check_range(tmp, positive=False)
                snr_list = np.arange(tmp[0], tmp[1] + (0.5 * tmp[2]), tmp[2])
            elif isinstance(tmp, int):
                snr_list = [tmp]
            else:
                raise ValueError("Invalid SNR range.")

            mapped["channel_params"] = [snr for snr in snr_list]
        else:
            raise ValueError("Invalid channel type.")

    ## savename
    tmp = dataset_dir + "/" + config["savepath"]

    if os.path.exists(tmp):
        ## modify pathname with date (DD-MM-YY) and time (H-M-S)
        t = datetime.today()
        mapped["savepath"] = tmp + "_" + t.strftime("%d-%m-%y-%H-%M-%S")
    else:
        mapped["savepath"] = tmp
    os.makedirs(mapped["savepath"])
    mapped["savename"] = mapped["savepath"].split("/")[-1]

    ## verbosity
    if config["verbose"] in [0, 1]:
        mapped["verbose"] = config["verbose"]
    else:
        raise ValueError("Verbosity may only be 0 or 1")

    ## archive
    assert isinstance(config["archive"], bool), "archive must be a boolean."
    mapped["archive"] = config["archive"]

    return mapped
