from manne_dataset import get_augmentations_from_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model, save_model
from os.path import join, basename
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers


class ManneModel():
    def __init__(self, filenameOrOptions, verbose=True):
        if filenameOrOptions.__class__ is str:
            self.load_saved_model(filenameOrOptions)
        elif filenameOrOptions.__class__ is dict:
            self.init_new_model(filenameOrOptions, verbose)

    def load_saved_model(self, filename):
        self.network = load_model(filename)
        self.encoder = self.network.get_layer('encoder')
        self.decoder = self.network.get_layer('decoder')

        self.input_size = self.network.input.shape[1]
        self.latent_size = self.encoder.output.shape[1]
        self.output_size = self.network.output.shape[1]
        self.augmentation_size = self.output_size - self.input_size
        self.augmentations = get_augmentations_from_filename(self.name)

        self.name = basename(filename)

        self.history = np.load(
            filename + '.training_history.npy', allow_pickle=True)

    def save_model(self, save_history=False):
        self.network.save(join('models', self.name))
        if save_history:
            self.save_history()

    def save_latents(self, inputs, name=''):
        latents = self.encode(inputs)
        if name:
            name += '_'
        np.save(join('models', self.name, f'{name}latents.npy'), latents)

    def save_history(self, extra={}):
        h = self.network.history.history
        h.update(extra)
        np.save(join('models', self.name, 'history.npy'), h)

    def init_new_model(self, options, print_summaries=True):
        dataset_name = options.get('dataset_name')
        self.input_size = options.get('input_size', 2048)
        self.latent_size = options.get('latent_size', 8)
        self.output_size = options.get('output_size', 2048)
        self.augmentation_size = self.output_size - self.input_size

        self.net_type = options.get('net_type', 'ae')
        self.skip = options.get('skip', 'false')

        self.define_net(print_summaries)

        skip = ['noskip', 'skip'][self.skip]
        self.name = f'{self.net_type}_{skip}_l{self.latent_size}_{dataset_name}'

    def define_net(self, print_summaries=True):

        if self.net_type == 'vae':
            l2_penalty = 0
        else:
            l2_penalty = 1e-7

        first_enc = int(np.log2(self.input_size)) - 1
        last_enc = int(np.log2(self.latent_size))
        self.encoder_widths = [
            2 ** n for n in range(first_enc, last_enc, -1)] + [self.latent_size]
        self.decoder_widths = [
            2 ** (n + 1) for n in range(last_enc, first_enc)]

        # Lighter weight model
        # self.encoder_widths = [512,256,128,64,8]
        # self.decoder_widths = [64,128,256,512]

        alpha_val = 0.1

        # ENCODER

        input_spec = Input(shape=(self.input_size,))
        encoded = Dense(units=self.encoder_widths[0],
                        activation=None,
                        kernel_regularizer=l2(l2_penalty))(input_spec)
        encoded = LeakyReLU(alpha=alpha_val)(encoded)
        for width in self.encoder_widths[1:-1]:
            encoded = Dense(units=width,
                            activation=None,
                            kernel_regularizer=l2(l2_penalty))(encoded)
            encoded = LeakyReLU(alpha=alpha_val)(encoded)
        encoded = Dense(
            units=self.encoder_widths[-1], activation='sigmoid', kernel_regularizer=l2(l2_penalty))(encoded)

        if self.net_type == 'vae':
            dim_z = self.encoder_widths[-1]
            prior = tfd.Independent(tfd.Normal(loc=tf.zeros(
                dim_z), scale=1.), reinterpreted_batch_ndims=1)
            encoded = Dense(tfpl.IndependentNormal.params_size(
                dim_z), activation=None, name='z_params')(encoded)
            encoded = tfpl.IndependentNormal(event_shape=dim_z,
                                             convert_to_tensor_fn=tfd.Distribution.sample,
                                             activity_regularizer=tfpl.KLDivergenceRegularizer(
                                                 prior, weight=self.beta),
                                             name='z_layer')(encoded)

        self.encoder = Model(input_spec, encoded, name='encoder')

        # DECODER

        if self.skip is True:
            input_latent = Input(
                shape=(self.encoder_widths[-1] + self.augmentation_length,))
        else:
            input_latent = Input(shape=(self.encoder_widths[-1],))

        decoded = Dense(units=self.decoder_widths[0],
                        activation=None,
                        kernel_regularizer=l2(l2_penalty))(input_latent)
        decoded = LeakyReLU(alpha=alpha_val)(decoded)
        for width in self.decoder_widths[1:]:
            decoded = Dense(units=width,
                            activation=None,
                            kernel_regularizer=l2(l2_penalty))(decoded)
            decoded = LeakyReLU(alpha=alpha_val)(decoded)
        decoded = Dense(units=self.output_size,
                        activation='relu',
                        kernel_regularizer=l2(l2_penalty))(decoded)
        if self.net_type == 'vae':
            decoded = Dense(
                units=tfpl.IndependentNormal.params_size(self.output_size))(decoded)
            decoded = tfpl.IndependentNormal(
                self.output_size, name='x_layer', )(decoded)
        self.decoder = Model(input_latent, decoded, name='decoder')

        # NEWORK

        auto_input = Input(shape=(self.input_size, ), name='net_input')
        encoded = self.encoder(auto_input)

        if self.skip is True:
            onehot_input = Input(shape=(self.augmentation_size,))
            new_latents = Concatenate()([encoded, onehot_input])
            decoded = self.decoder(new_latents)
            self.network = Model(
                inputs=[auto_input, onehot_input], outputs=decoded, name='autoencoder')
        else:
            decoded = self.decoder(encoded)
            self.network = Model(
                inputs=[auto_input], outputs=decoded, name='autoencoder')

        if print_summaries:
            self.print_summary()

    def print_summary(self):
        print('\n net summary \n')
        self.network.summary()
        print('\n encoder summary \n')
        self.encoder.summary()
        print('\n decoder summary \n')
        self.decoder.summary()

    def predict(self, inputs):
        if self.net_type == 'vae':
            return self.network.network(inputs).sample()
        else:
            return self.network.predict(inputs, verbose=1)

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, latents):
        return self.decoder.predict(latents)
        return self.decoder.predict(latents)
