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

### Training
Datasets are stored as single wave files in the `waves` folder. The first step is to preprocess a dataset and store STFT frames in a .npy files in the `frames` folder.
```
python wav2frames.py your_filename.wav
```
Note that the program looks for `your_filename.wav` in the `waves` folder, so it doesn't need to be a path. It will save frames in `frames/your_filename.npy`


To train the network on a .npy dataset run:
```
python manne_train.py your_filename --n-epoch 30
```
This trains a VAE for 30 epochs on the dataset in `frames/your_filename.npy` (note there is no .npy in the argument passed to the program above). Models (encoder and decoder) are automatically saved in the `models` folder after training.


Training evaluation is performed as described in the referenced paper by the original author. Datasets splits for training, validation and testing can be configured by changing global variables `TRAINING_SIZE` and `VALIDATION_SIZE` in `manne_train.py`.

### Rendering interpolations
The idea about interpolating is different from the original. Instead of a free manipulation of latent space coordinates, we will interpolate between two embeddings. The predictor will then encode for two points, from data from two different files, and produce a single interpolated result. The generated magnitude spectra is then coupled with phases from the first embedding and written to a file.


The system works with couples of soundfiles to be interpolated.
```
python manne_render.py models/your_model_filename source_a source_b result_name [interp=0.5]
```
Where:
- `source_a` and `source_b` can be both either files or folders containing wave files.
- `interp` is an optional float number controlling the amount of interpolation between sources.
