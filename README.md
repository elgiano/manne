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
This trains a VAE for 30 epochs on the dataset in `frames/your_filename.npy` (note there is no .npy in the argument passed to the program above).
Models (encoder and decoder) are automatically saved in the `models` folder after training.

### Rendering predictions
```
python manne_render.py your_model_filename
```
