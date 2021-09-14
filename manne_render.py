import numpy as np
import os
import librosa
import soundfile as sf
import argparse
import pyaudio
import numpy as np
from tensorflow.keras.models import Model, load_model
import tensorflow as tf

RATE     = int(44100)
CHUNK    = int(1024)

class ManneRender():

    def __init__(self, model_name):
        self.model_name = model_name
        self.scales = np.ones(8)
        self.load_model()

    def sampling(self, args):
        self.z_mean, self.z_log_var = args
        batch = tf.keras.backend.shape(self.z_mean)[0]
        dim = tf.keras.backend.int_shape(self.z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch,dim))
        return self.z_mean + tf.keras.backend.exp(0.5*self.z_log_var)*epsilon

    def load_model(self, print_summary=False):
        print(f'[Model] loading {self.model_name}')
        data_path_enc = os.path.join(os.getcwd(), self.model_name + '_trained_encoder.h5')
        self.encoder = load_model(data_path_enc, compile=False, custom_objects={"sampling":lambda args:self.sampling(args)})

        data_path_dec = os.path.join(os.getcwd(), self.model_name +'_trained_decoder.h5')
        self.decoder = load_model(data_path_dec, compile=False)

        if print_summary:
            print('\n encoder summary \n')
            self.encoder.summary()
            print('\n decoder summary \n')
            self.decoder.summary()

    def process_track(self, track_name, augment=True):
        print(f'[Track] processing {track_name}')
        len_window = 4096 #Specified length of analysis window
        hop_length_ = 1024 #Specified percentage hop length between windows

        data_path = os.path.join(os.getcwd(), track_name)
        y, sr = librosa.load(data_path, sr=44100, mono=True)

        D = librosa.stft(y,n_fft=len_window, window='hann')
        mag = D
        mag = np.abs(mag) #Magnitude response of the STFT
        remember = mag.max(axis=0)+0.000000001 #Used for normalizing STFT frames (with addition to avoid division by zero)
        mag = (mag / remember).T #Normalizing
        phase = np.angle(D) #Phase response of STFT
        remember = remember
        if augment:
            chroma = librosa.feature.chroma_stft(S=np.transpose(mag), sr=44100)
            chroma = (chroma == chroma.max(axis=1)[:,None]).astype(int)
            chroma = np.transpose(chroma)
            augmentations = chroma
            mag = np.hstack((mag, augmentations))

        return mag, phase, remember

    def generate(self, mag, phase, remember, scales=None):
        scales = scales or self.scales
        print(scales)

        enc_mag = self.encoder.predict(mag)
        enc_mag = enc_mag[2] * scales #NEED TO ADD SCALE HERE
        out_mag = self.decoder.predict(enc_mag)

        out_mag = out_mag.T * remember
        E = out_mag * np.exp(1j * phase)
        out = np.float32(librosa.istft(E))
        return out

    def render(self, out_file_name, track_name):
        global CHUNK
        mag, phase, remember = self.process_track(track_name)
        print('[Rendering] Start')
        out = self.generate(mag, phase, remember)
        out = 0.8*out[3*CHUNK:]
        print(f'[Rendering] Writing {out_file_name}')
        sf.write(out_file_name, out, 44100, subtype='PCM_16')
        print('[Rendering] Done')


class ManneInterpolator(ManneRender):

    def interpolate(self, mag_a, phase_a, remember_a, mag_b, phase_b, remember_b, interp=0.5):

        print('[Interpolate] encoding file 1')
        enc_mag = self.encoder.predict(mag_a)[2]
        print('[Interpolate] encoding file 2')
        enc_mag2 = self.encoder.predict(mag_b)[2]
        enc_mag_interp = (enc_mag + enc_mag2) * interp
        print('[Interpolate] decoding')
        out_mag = self.decoder.predict(enc_mag_interp)

        out_mag = out_mag.T * remember_a
        E = out_mag * np.exp(1j * phase_a)
        out = np.float32(librosa.istft(E))
        return out

    def render(self, out_file_name, track_name_a, track_name_b, interp = 0.5):
        global CHUNK
        mag_a, phase_a, remember_a = self.process_track(track_name_a)
        mag_b, phase_b, remember_b = self.process_track(track_name_b)
        print('[Rendering] Start')
        out = self.interpolate(mag_a, phase_a, remember_a, mag_b, phase_b, remember_b, interp)
        out = 0.8*out[3*CHUNK:]
        print(f'[Rendering] Writing {out_file_name}')
        sf.write(out_file_name, out, 44100, subtype='PCM_16')
        print('[Rendering] Done')


model_name   = "models/vae_soren_magpickup23short"
track_name   = "waves/soren_magpickup23shorterA.wav"
track_name_b = "waves/soren_magpickup23shorterB.wav"

# ManneRender(model_name).render('rendered.wav', track_name)
ManneInterpolator(model_name).render('interpolated.wav', track_name, track_name_b, 0.5)
