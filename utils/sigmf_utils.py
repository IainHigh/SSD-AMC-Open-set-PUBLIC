# pylint: disable-all
## python imports
import numpy as np
from datetime import datetime
import os
import tarfile
import sigmf


def archive_sigmf(savepath):
    savepath_ext = savepath + ".sigmf"
    files = os.listdir(savepath)

    with tarfile.open(savepath_ext, mode="w") as out:
        for f in files:
            ## check file extensions
            assert f[-5:] in ["-data", "-meta"], (
                "Invalid file type found in " + savepath
            )
            out.add(savepath + "/" + f, arcname=f)


def save_sigmf(i, q, config, idx):
    record_name = config["savepath"] + "/" + config["savename"] + "-" + str(idx)

    # Create the savepath directory if it doesn't exist
    if not os.path.exists(config["savepath"]):
        os.makedirs(config["savepath"])

    data_name = record_name + ".sigmf-data"
    meta_name = record_name + ".sigmf-meta"

    ## interleave IQ, save in .sigmf-data file
    iq = np.zeros(2 * len(i), dtype=i.dtype)
    iq[0::2] = i
    iq[1::2] = q
    # Write raw interleaved IQ samples to comply with the SigMF spec
    with open(data_name, "wb") as f:
        iq.tofile(f)

    _global = {
        "core:datatype": "cf32_le",
        "core:version": "v0.0.1",
        "core:recorder": "liquid-dsp",
        "core:extensions": {
            "modulation": "v0.0.2",
            "channel": "v0.0.1",
            "filter": "v0.0.1",
        },
    }
    f = sigmf.SigMFFile(data_file=data_name, global_info=_global)
    f.add_capture(
        0,
        metadata={
            "core:length": config["n_samps"],
            "core:time": datetime.today().isoformat(),
        },
    )
    f.add_annotation(
        0,
        config["n_samps"],
        metadata={
            "rfml_labels": {"modclass": config["modname"]},
            "sampling_rate": config["sampling_rate"],
            "center_frequencies": config["center_frequencies"],
        },
    )

    f.add_annotation(
        0,
        config["n_samps"],
        metadata={
            "channel": {
                "type": config["channel_type"],
                "snr": config["snr"],
            },
            "filter": {
                "type": config["filter_type"],
                "sps": config["sps"],
                "delay": config["delay"],
                "rolloff": config["beta"],
                "dt": config["dt"],
            },
        },
    )

    with open(meta_name, "w") as mf:
        f.dump(mf, pretty=True)

    return record_name
