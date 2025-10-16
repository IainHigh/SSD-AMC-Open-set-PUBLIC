# Single-shot Detector for Joint Signal Detection and Modulation Classification with Open-set Recognition

This repository contains all the code associated with the academic paper:
"Single-shot Detector for Joint Signal Detection and Modulation Classification with Open-set Recognition"

This work was completed by Iain High (University of Edinburgh), Wasiu O. Popoola (University of Edinburgh), and David Sadler (Roke Manor Research Ltd.).

For correspondance, please contact Iain High:
Academic email: i.high@sms.ed.ac.uk
Personal email: iain.high@sky.com

## Abstract

Increased crowding on the radio frequency spectrum has resulted in a greater risk of radar interference, creating the demand for cognitive radars that can dynamically adapt to avoid interference. A technique that benefits cognitive radar performance is automatic modulation classification, which is the task of identifying the modulation scheme used to encode received digital communications without prior knowledge. Current approaches fail to address the challenges of wideband radio frequency environments that simultaneously contain multiple transmitters and previously unseen modulation schemes. To address the issue of numerous transmitters, this research proposes a novel singleshot detector architecture for detecting and classifying radio frequency communications in a single forward pass through the model. The second issue of previously unseen modulation schemes is also addressed through the incorporation of open-set recognition. The results demonstrate that the proposed model achieves high accuracy in detection, classification, and open-set recognition. This research helps adapt automatic modulation classification to more realistic scenarios by addressing the joint detection and classification in wideband operation with previously unseen modulation schemes. In turn, this framework can improve the performance of downstream tasks such as spectrum sensing for cognitive radar

## Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council and Ministry of Defence Centre for Doctoral Training in Sensing, Processing and AI for Defence and Security, [EP/Y013859/1]. This work has made use of the resources provided by the Edinburgh Compute and Data Facility (ECDF) (http://www.ecdf.ed.ac.uk/).

## License

This repository is licensed under the MIT License.

# Setup Instructions

In addition to the python packages listed in requirements.sh, the code in this repo is dependent upon [liquid-dsp](https://github.com/jgaeddert/liquid-dsp).
To install liquid-dsp, clone the repo linked, and follow the installation instructions in the README.
Ensure that you rebind your dynamic libraries using `sudo ldconfig`.

1. Clone or download the repository.
2. Install liquid-dsp as described above.
3. Install the required Python packages.
   ```
   pip install -r requirements.txt
   ```
4. Compile the C modules.
   ```
   cd ./cmodules && make && cd ../
   ```
5. Adjust the paths in "SSD_AMC_with_OSR_Model/config.py" and "generator.py" to your working directory and desired dataset directory.
6. Generate datasets using the `generator.py` script with a configuration file from the `configs` directory.
   ```
   python generator.py ./configs/training_set.json
   python generator.py ./configs/testing_set.json
   ```
7. Train and test the model using the `main.py` script in the `SSD_AMC_with_OSR_Model` directory.
   ```
   python SSD_AMC_with_OSR_Model/main.py
   ```

# generator.py

Python tool to generate synthetic radio frequency (RF) datasets.

Datasets are saved in SigMF format.
Each dataset is a _SigMF Archive_ composed of multiple _SigMF Recordings_.
Each _SigMF Recording_ contains a single capture, saved as a binary file (.sigmf-data files), with an associated metadata file (.sigmf-meta) containing the parameters used to generate that capture.
See the [SigMF specification](https://github.com/gnuradio/SigMF/blob/master/sigmf-spec.md) to read more.

# SSD_AMC_with_OSR_Model

This directory contains the PyTorch implementation of the single-shot detector architecture for the joint detection and automatic modulation classification of signals with open-set recognition.

## Files

- `config.py` – configuration parameters controlling dataset paths, model size and training options.
- `dataset.py` – for loading SigMF recordings and producing labels.
- `model.py` – network architecture used for frequency localisation. During
  inference the model outputs only embedding vectors which are classified via
  distance to class centres.
- `loss.py` – custom loss function combining IoU, confidence, classification and
  open‑set embedding terms.
- `main.py` – training and testing pipeline.

## Usage

1. Ensure datasets `training` and `testing` are generated in the directory defined in `config.py`.
2. Adjust hyperparameters in `config.py` if required.
3. Launch training and testing with:
   ```
   python main.py
   ```
   The script prints training progress and writes optional plots and results when enabled in the configuration file.

# Documentation of Config JSON Files

Configuration files should contain the following parameters:

- `n_captures`: the number of captures to generate per modulation scheme. e.g. 10 will create 10 different files for each modulation type.
- `n_samps`: the number of raw IQ samples per capture. This will be the length the IQ list after taking samples.
- `modulations`: the modulation schemes to include in the dataset (may include "bpsk", "qpsk", "8psk", "16psk", "4dpsk", "16qam", "32qam", "64qam", "16apsk", "32apsk", "fsk5k", "fsk75k", "gfsk5k", "gfsk75k", "msk", "gmsk", "fmnb", "fmwb", "dsb", "dsbsc", "lsb", "usb", and "awgn")
- `symbol_rate`: number of symbols per frame, list of desired symbol rates accepted. Lower value means less samples per signal = more signals over total sample space. AKA Samples per Symbol.
- `sampling_rate`: The sampling rate of the data in Hz.
- `center_frequencies`: List of center frequencies for the carriers in Hz. This is a list of floats, e.g. [100e6, 200e6, 300e6]. If this is provided, `randomly_generated_center_frequencies` should not be provided.
- `randomly_generated_center_frequencies`: Ramdomly generate center frequency carriers: [lower_bound, upper_bound, number_of_carriers, prevent_overlap]. Should not be provided when `center_frequencies` is provided. prevent_overlap is a boolean value 1 for true, 0 for false. If true the generated center frequencies will not overlap with each other. If false the generated center frequencies may overlap with each other.
- `filter`: default transmit filter parameters, including the type of filter (Gaussian or root-raised cosine (RRC)), the excess bandwidth or `beta`, the symbol overlap or `delay`, and the fractional sample delay or `dt` (all gfsk/gmsk signals use Gaussian filters, all remaining fsk/msk signals use square filters, all psk/qam signals use RRC filters)
- `channel`: {
  `type`: Currently the only supported channel is "awgn"
  `snr`: Signal to Noise Ratio (dB)
  }
- `savepath`: the dataset location
- `verbose`: 0 for minimal verbosity, 1 for debugging
- `archive`: create a SigMF archive of dataset when complete
