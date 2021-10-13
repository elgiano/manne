# manne

Adapted ANNe effect from [JITColonel/manne](https://github.com/JTColonel/manne).

Reference paper:
[J. T. Colonel, S. Keene - Conditioning Autoencoder Latent Spaces for Real-Time Timbre Interpolation and Synthesis (2020)](
https://deepai.org/publication/conditioning-autoencoder-latent-spaces-for-real-time-timbre-interpolation-and-synthesis)

Tested with Python 3.9

### Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Preparing dataset
Datasets are stored as single wave files in the `waves` folder. The first step is to preprocess a dataset and store STFT frames in a .npy files in the `frames` folder.
```
python manne_dataset.py waves/your-filename.wav
```

```
usage: manne_dataset.py [-h] [-N FFT_SIZE] [-H FFT_HOP] [-n] [-a {stft,cqt}] [-o CQT_BINS_PER_OCTAVE] [-+ AUGMENT] filename

positional arguments:
  filename

optional arguments:
  -h, --help            show this help message and exit
  -N FFT_SIZE, --fft_size FFT_SIZE
                        FFT window size in samples (default: 4096)
  -H FFT_HOP, --fft_hop FFT_HOP
                        FFT hop in samples (default: 1024)
  -n, --normalize       normalize frames
  -a {stft,cqt}, --anal {stft,cqt}
                        STFT or CQT analysis
  -o CQT_BINS_PER_OCTAVE, --cqt_bins_per_octave CQT_BINS_PER_OCTAVE
                        CQT bins per octave
  -+ AUGMENT, --augment AUGMENT
                        chroma, octave or chroma+octave (default: chroma)
```

Note: CQT analysis is not implemented yet

### Training
To train the network on a .npy dataset run:
```
python manne_train.py frames/your-filename.npy -e 30
```
This trains a VAE for 30 epochs on the dataset in `frames/your-filename.npy`. Models (encoder and decoder) are automatically saved in the `models` folder during training.
After training, models are evaluated on testing and eventual validation set, and samples are printed to PDFs in the `eval` folder.

```
usage: manne_train.py [-h] [--net_type {ae,vae}] [--skip] [-e N_EPOCHS] [--latent_size LATENT_SIZE]
                      [--batch_size BATCH_SIZE] [--train_size TRAIN_SIZE] [--test_size TEST_SIZE] [--distribute]
                      [--save_history] [--save_latents] [--mode {train,plot,save_latents}]
                      dataset_path

positional arguments:
  dataset_path          dataset to train on, or model to load for modes other than train

optional arguments:
  -h, --help            show this help message and exit
  --net_type {ae,vae}
  --skip                enable skip connection
  -e N_EPOCHS, --n_epochs N_EPOCHS
  --latent_size LATENT_SIZE
  --batch_size BATCH_SIZE
  --train_size TRAIN_SIZE
                        fraction of the dataset to use for training. If 1, doesn't perform validation (default: 0.8).
  --test_size TEST_SIZE
                        fraction of the dataset to use for testing. If None, the fraction of dataset not used for
                        training is split equally between validation and testing (default: None).
  --distribute
  --save_history
  --save_latents
  --mode {train,plot,save_latents}
                        train: trains a new model; plot: loads an existing model, evaluates it and saves PDFs;
                        save_latents: loads an existing model and save latent embeddings of each training example
                        (default: train)
```

### Rendering interpolations
The idea about interpolating is different from the original. Instead of a free manipulation of latent space coordinates, we will interpolate between two embeddings. The encoder will then encode for two points, from data from two different files, and produce a single interpolated result. The generated magnitude spectra is then coupled with phases from the first embedding and written to a file.


The system works with couples of soundfiles to be interpolated.
```
python manne_render.py models/your_model_filename source_a source_b result_name [interp=0.5]
```
Where:
- `source_a` and `source_b` can be both either files or folders containing wave files.
- `interp` is an optional float number controlling the amount of interpolation between sources.
