from os.path import isdir, basename, splitext, join, dirname
import numpy as np
import os
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import tensorflow as tf
from scipy import signal
from rtpghi import PGHI
from manne_dataset import append_augmentations, get_augmentations_from_filename, get_skip_from_filename

RATE = int(44100)
CHUNK = int(1024)


class ManneRender():

    def __init__(self, model_name, verbose=True):
        self.model_name = model_name
        self.scales = np.ones(8)
        self.verbose = verbose
        self.load_model()

    def log(self, *data):
        if self.verbose:
            print(data)

    def sampling(self, args):
        self.z_mean, self.z_log_var = args
        batch = tf.keras.backend.shape(self.z_mean)[0]
        dim = tf.keras.backend.int_shape(self.z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return self.z_mean + tf.keras.backend.exp(0.5 * self.z_log_var) * epsilon

    def load_model(self, print_summary=False):
        print(f'[Model] loading {self.model_name}')
        data_path_enc = os.path.join(
            os.getcwd(), self.model_name + '_trained_encoder.h5')
        self.encoder = load_model(data_path_enc, compile=False)

        data_path_dec = os.path.join(
            os.getcwd(), self.model_name + '_trained_decoder.h5')
        self.decoder = load_model(data_path_dec, compile=False)
        print(f'[Model] loaded {self.model_name}')
        self.model_augmentations, self.model_num_aug = get_augmentations_from_filename(
            self.model_name)
        print(f'[Model] augmentations: {self.model_augmentations}')
        self.model_has_skip = get_skip_from_filename(self.model_name)
        print(f'[Model] skip connections: {self.model_has_skip}')

        if print_summary:
            print('\n encoder summary \n')
            self.encoder.summary()
            print('\n decoder summary \n')
            self.decoder.summary()

    def process_track(self, track_name, augmentations=None):
        print(f'[Track] processing {track_name}')
        data_path = os.path.join(os.getcwd(), track_name)
        y, sr = librosa.load(data_path, sr=44100, mono=True)
        return self.process_audio(y, augmentations=None)

    def process_audio(self, y, augmentations=None):
        len_window = 4096  # Specified length of analysis window
        hop_length = 1024  # Specified percentage hop length between windows
        D = librosa.stft(y, n_fft=len_window,
                         hop_length=hop_length, window='hann')
        mag = D[:-1, :]
        mag = np.abs(mag)  # Magnitude response of the STFT
        # Used for normalizing STFT frames (with addition to avoid division by zero)
        remember = mag.max(axis=0) + 0.000000001
        mag = (mag / remember).T  # Normalizing
        phase = np.angle(D)  # Phase response of STFT
        remember = remember
        if augmentations is None:
            augmentations = self.model_augmentations
        mag = append_augmentations(
            augmentations, mag, y, len_window, hop_length)

        return mag, phase, remember

    def generate(self, mag, phase, remember, scales=None):
        scales = scales or self.scales
        print(mag)
        enc_mag = self.encoder.predict(mag)
        enc_mag = enc_mag * scales  # NEED TO ADD SCALE HERE
        if self.model_has_skip:
            out_mag = self.decoder.predict(
                np.hstack((enc_mag, mag[:, -self.model_num_aug:])))
        else:
            out_mag = self.decoder.predict(enc_mag)
        # reappend nyquist bin after prediction
        out_mag = np.hstack((out_mag, np.zeros((len(out_mag), 1))))
        out_mag = out_mag.T * remember
        E = out_mag * np.exp(1j * phase)
        out = np.float32(librosa.istft(E))
        return out

    def render(self, out_file_name, track_name):
        global CHUNK
        mag, phase, remember = self.process_track(track_name)
        print('[Rendering] Start')
        out = self.generate(mag, phase, remember)
        out = 0.8 * out[3 * CHUNK:]
        print(f'[Rendering] Writing {out_file_name}')
        sf.write(out_file_name, out, 44100, subtype='PCM_16')
        print('[Rendering] Done')

    def rtpghi_start(self):
        self.rtpghi = PGHI(4096, 4, verbose=False)

    def get_phase(self, mag, rtpghi=False):
        if rtpghi:
            if not self.rtpghi:
                self.rtpghi_start()
            return self.rtpghi.estimate(mag.T).T
        else:
            return np.random.random(mag.shape) * np.pi * 2


class ManneInterpolator(ManneRender):

    def wrap_extend(self, array, new_len):
        if len(array) < new_len:
            tiles = int(np.ceil(new_len / len(array)))
            return np.tile(array, (tiles, 1))[:new_len]
        elif len(array) > new_len:
            return array[:new_len]
        else:
            return array

    def interpolate(self, mag_a, phase_a, remember_a, mag_b, phase_b, remember_b, interp=0.5, rtpghi=None):

        self.log('[Interpolate] encoding file 1')
        enc_mag = self.encoder.predict(mag_a)
        self.log('[Interpolate] encoding file 2')
        enc_mag2 = self.encoder.predict(mag_b)
        enc_mag2 = self.wrap_extend(enc_mag2, len(enc_mag))
        enc_mag_interp = (1 - interp) * enc_mag + interp * enc_mag2
        self.log('[Interpolate] decoding')
        if self.model_has_skip:
            out_mag = self.decoder.predict(
                np.hstack((enc_mag_interp, mag_a[:, -self.model_num_aug:])))
        else:
            out_mag = self.decoder.predict(enc_mag_interp)
        out_mag = np.hstack((out_mag, np.zeros((len(out_mag), 1))))
        out_mag = out_mag.T * remember_a
        if rtpghi is None:
            phase = phase_a
        else:
            phase = self.get_phase(out_mag, rtpghi)
        E = out_mag * np.exp(1j * phase)
        out = np.float32(librosa.istft(E))
        return out

    def render(self, out_file_name, track_name_a, track_name_b, interp=0.5, rtpghi=None):
        global CHUNK
        mag_a, phase_a, remember_a = self.process_track(track_name_a)
        mag_b, phase_b, remember_b = self.process_track(track_name_b)
        print('[Rendering] Start')
        if rtpghi:
            self.rtpghi_start()
        out = self.interpolate(mag_a, phase_a, remember_a,
                               mag_b, phase_b, remember_b, interp, rtpghi)
        out = 0.8 * out[3 * CHUNK:]
        print(f'[Rendering] Writing {out_file_name}')
        sf.write(out_file_name, out, 44100, subtype='PCM_16')
        print('[Rendering] Done')


class ManneSynth(ManneRender):

    def decode(self, latent, sr, fft_size, fft_hop, rtpghi=False):
        out_mag = self.decoder.predict(latent)
        out_mag = np.hstack((out_mag, np.zeros((len(out_mag), 1))))
        out_phase = self.get_phase(out_mag.T, rtpghi).T
        E = out_mag * np.exp(1j * out_phase)
        out = np.float32(librosa.istft(
            E.T, hop_length=fft_hop, win_length=fft_size))
        # out = np.float32(signal.istft(
        #     E.T, fs=sr, nfft=fft_size, noverlap=fft_hop))
        return out

    def chroma(self, chroma, latent, sr, fft_size, fft_hop, rtpghi=False):
        chroma_vec = np.zeros((len(latent), 12))
        chroma_vec[:, chroma] = 1
        latent = np.hstack((latent, chroma_vec))
        return self.decode(latent, sr, fft_size, fft_hop, rtpghi)

    def note(self, chroma, octave, latent, sr, fft_size, fft_hop, rtpghi=False):
        chroma_vec = np.zeros((len(latent), 12))
        chroma_vec[:, chroma] = 1
        octave_vec = np.zeros((len(latent), 8))
        octave_vec[:, octave] = 1
        latent = np.hstack((latent, chroma_vec, octave_vec))
        return self.decode(latent, sr, fft_size, fft_hop, rtpghi)

    def render_note(self, out_file_name, chroma, octave, latent, sr, fft_size, fft_hop, rtpghi=False):
        self.rtpghi_start()
        out = self.note(chroma, octave, latent, sr, fft_size, fft_hop, rtpghi)
        out = 2 * out[3 * CHUNK:]
        print(f'[Rendering] Writing {out_file_name}')
        sf.write(out_file_name, out, sr, subtype='PCM_16')
        print('[Rendering] Done')

    def render_chromatic_line(self, out_file_name, sr, fft_size, fft_hop, rtpghi=False, dur=60):
        self.rtpghi_start()
        note_samples = int(sr * dur / (12 * 8))
        chromatic = ['c', 'c#', 'd', 'd#', 'e',
                     'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
        latent = None
        for octave in range(8):
            octave_vec = np.zeros((note_samples, 8))
            octave_vec[:, octave] = 1
            for (chroma, name) in enumerate(chromatic):
                chroma_vec = np.zeros((note_samples, 12))
                chroma_vec[:, chroma] = 1
                note_noise = np.linspace(np.random.rand(
                    8), np.random.rand(8), note_samples)
                note_latent = np.hstack((note_noise, chroma_vec, octave_vec))
                if latent is None:
                    latent = note_latent
                else:
                    latent = np.vstack((latent, note_latent))

        out = self.decode(latent, sr, fft_size, fft_hop, rtpghi)
        out = 2 * out[3 * CHUNK:]
        print(f'[Rendering] Writing {out_file_name}')
        sf.write(out_file_name, out, sr, subtype='PCM_16')
        print('[Rendering] Done')
