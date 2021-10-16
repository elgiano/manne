import timeit
from manne_dataset import get_augmentations_from_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
from os import makedirs
from os.path import join, basename, isdir, splitext
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers


def load_saved_model(keras_or_tflite_path):
    if keras_or_tflite_path.endswith('tflite'):
        return ManneModelLite(keras_or_tflite_path)
    else:
        return ManneModel(keras_or_tflite_path)


class ManneModel():
    def __init__(self, filenameOrOptions, verbose=True):
        if filenameOrOptions.__class__ is str:
            self.load_saved_model(filenameOrOptions)
        elif filenameOrOptions.__class__ is dict:
            self.init_new_model(filenameOrOptions, verbose)

    def load_saved_model(self, filename):
        if filename.endswith('/'):
            filename = filename[:-1]
        self.model_path = filename
        self.name = splitext(basename(filename))[0]
        print(f"[Model] loading {self.name}")

        self.network = load_model(filename)
        self.encoder = self.network.get_layer('encoder')
        self.decoder = self.network.get_layer('decoder')
        self.net_type, self.skip = self.name.split('_')[0:2]
        self.skip = self.skip == 'skip'

        if self.skip:
            self.input_size = self.network.input[0].shape[1]
        else:
            self.input_size = self.network.input.shape[1]

        self.latent_size = self.encoder.output.shape[1]
        self.output_size = self.network.output.shape[1]
        self.augmentation_size = self.input_size - self.output_size
        self.augmentations = get_augmentations_from_filename(self.name)[0]

        self.history = self.load_history()

    def get_checkpoints_dir(self):
        # return join('models', self.name, 'checkpoints')
        return join('models', self.name)

    def save_model(self, save_history=False):
        self.network.save(join('models', self.name))
        if save_history:
            self.save_history()

    def save_latents(self, inputs, name=''):
        latents = self.encode(inputs)
        if name:
            name += '_'
        np.save(join('models', self.name, f'{name}latents.npy'), latents)

    def load_latents(self, name=''):
        if name:
            name += '_'
        return np.load(join(self.model_path, f'{name}latents.npy'))

    def save_history(self, extra={}):
        h = self.network.history.history
        h.update(extra)
        np.save(join('models', self.name, 'history.npy'), h)

    def load_history(self):
        try:
            h = np.load(join('models', self.name, 'history.npy'),
                        allow_pickle=True)
            if h.__class__ == np.ndarray and h.dtype == np.object:
                h = h.item()
            return h
        except:
            print(f'Model training history not found')

    def init_new_model(self, options, print_summaries=True):
        dataset_path = options.get('dataset_path')
        self.input_size = options.get('input_size', 2048)
        self.latent_size = options.get('latent_size', 8)
        self.output_size = options.get('output_size', 2048)
        self.augmentation_size = self.input_size - self.output_size

        self.net_type = options.get('net_type', 'ae')
        self.skip = options.get('skip', 'false')

        self.define_net(print_summaries)

        skip = ['noskip', 'skip'][self.skip]
        self.name = f'{self.net_type}_{skip}_l{self.latent_size}_{splitext(basename(dataset_path))[0]}'
        print(f"[Model] creating {self.name}")

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
                shape=(self.encoder_widths[-1] + self.augmentation_size,))
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

    def reconstruct(self, inputs, verbose=0):
        if self.net_type == 'vae':
            return self.network.network(inputs).sample()
        else:
            return self.network.predict(inputs)

    def encode(self, inputs):
        return self.encoder.predict(inputs)

    def decode(self, latents):
        return self.decoder.predict(latents)

    def benchmark(self, verbose=True):
        results = []

        def random_input(len):
            if self.skip:
                return [np.random.rand(len, self.input_size), np.random.rand(len, self.augmentation_size)]
            else:
                return np.random.rand(len, self.input_size)

        def random_latent(len):
            if self.skip:
                return np.random.rand(len, self.latent_size + self.augmentation_size)
            else:
                return np.random.rand(len, self.latent_size)

        self.encode(np.random.rand(1, self.input_size))

        for n in [1, 10, 100]:
            res = timeit.repeat(lambda: self.encode(
                np.random.rand(n, self.input_size)), number=10, repeat=5)
            w = max(res) / 10
            v = (max(res) - min(res)) / 10
            if verbose:
                print(f"Encoder: timing {n} frames")
                print(f"Slowest: {w}")
            results += [(w, v)]

            res = timeit.repeat(lambda: self.decode(
                random_latent(n)), number=10, repeat=5)
            w = max(res) / 10
            v = (max(res) - min(res)) / 10
            if verbose:
                print(f"Decoder: timing {n} frames")
                print(f"Slowest: {w}")
            results += [(w, v)]

            res = timeit.repeat(lambda: self.reconstruct(
                random_input(n)), number=10, repeat=5)
            w = max(res) / 10
            v = (max(res) - min(res)) / 10
            if verbose:
                print(f"Autoencoding: timing {n} frames")
                print(f"Slowest: {w}")
            results += [(w, v)]

        return results


class ManneModelLite:
    def __init__(self, model_path):

        if model_path.endswith('tflite'):
            self.load_model(model_path)
        else:
            self.load_keras_model(model_path)

        self.input_size = self.encoder['input_shape'][1]
        self.latent_size = self.encoder['output_shape'][1]
        self.decoder_input_size = self.decoder['input_shape'][1]
        self.output_size = self.decoder['output_shape'][1]
        self.augmentation_size = self.input_size - self.output_size

    def load_keras_model(self, path):
        self.name = basename(path)
        self.path = path
        self.model = ManneModel(path)
        print('Converting to tflite')
        self.autoencoder, self.encoder, self.decoder = [
            self.convert_model(net) for net in (self.model.network, self.model.encoder, self.model.decoder)]
        self.net_type, self.skip = self.model.net_type, self.model.skip
        self.augmentations = self.model.augmentations

    def convert_model(self, model):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        return self.get_model_data_from_interpreter(tflite_model, interpreter)

    def get_model_data_from_interpreter(self, model, interpreter):
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        output_shape = output_details[0]['shape']
        input_index = input_details[0]['index']
        output_index = output_details[0]['index']
        return {
            'tflite_model': model,
            'interpreter': interpreter, 'input_shape': input_shape, 'output_shape': output_shape,
            'input_index': input_index, 'output_index': output_index
        }

    def save_model(self):
        path = self.path + "_tflite"
        if not isdir(path):
            makedirs(path)
        with open(join(path, self.name + '_autoencoder.tflite'), 'wb') as f:
            f.write(self.autoencoder['tflite_model'])
        with open(join(path, self.name + '_encoder.tflite'), 'wb') as f:
            f.write(self.encoder['tflite_model'])
        with open(join(path, self.name + '_decoder.tflite'), 'wb') as f:
            f.write(self.decoder['tflite_model'])

    def load_model(self, model_path):
        self.path = model_path
        self.name = ("_").join(basename(model_path).split("_")[:-1])
        self.autoencoder, self.encoder, self.decoder = [
            self.load_submodel(join(self.path, self.name + p + '.tflite')) for p in ('_autoencoder', '_encoder', '_decoder')]
        self.net_type, self.skip = self.name.split('_')[0:2]
        self.skip = self.skip == 'skip'
        self.augmentations = get_augmentations_from_filename(self.name)[0]

    def load_submodel(self, path):
        interpreter = tf.lite.Interpreter(path)
        return self.get_model_data_from_interpreter(None, interpreter)

    def run_model_once(self, model, input):
        model['interpreter'].set_tensor(model['input_index'], input)
        model['interpreter'].invoke()
        output = model['interpreter'].get_tensor(model['output_index'])[0]
        return output

    def run_model(self, model, input):
        # print('pre', input)
        input = np.array(input, dtype=np.float32)
        # print('post', input)
        return np.vstack([self.run_model_once(model, np.array([x])) for x in input])

    def reconstruct(self, input):
        return self.run_model(self.autoencoder, input)

    def decode(self, input):
        return self.run_model(self.decoder, input)

    def encode(self, input):
        return self.run_model(self.encoder, input)

    def benchmark(self, verbose=True):
        results = []

        def random_encode(len):
            self.encode(np.random.rand(
                    len, self.input_size).astype(np.float32))

        def random_reconstruct(len):
            self.reconstruct(np.random.rand(
                    len, self.input_size).astype(np.float32))

        def random_decode(len):
            self.decode(np.random.rand(
                    len, self.decoder_input_size).astype(np.float32))

        self.encode(np.random.rand(1, self.input_size).astype(np.float32))

        for n in [1, 10, 100]:
            res = timeit.repeat(lambda: random_encode(n), number=10, repeat=5)
            w = max(res) / 10
            v = (max(res) - min(res)) / 10
            if verbose:
                print(f"Encoder: timing {n} frames")
                print(f"Slowest: {w}")
            results += [(w, v)]

            res = timeit.repeat(lambda: random_decode(n), number=10, repeat=5)
            w = max(res) / 10
            v = (max(res) - min(res)) / 10
            if verbose:
                print(f"Decoder: timing {n} frames")
                print(f"Slowest: {w}")
            results += [(w, v)]

            res = timeit.repeat(
                lambda: random_reconstruct(n), number=10, repeat=5)
            w = max(res) / 10
            v = (max(res) - min(res)) / 10
            if verbose:
                print(f"Autoencoding: timing {n} frames")
                print(f"Slowest: {w}")
            results += [(w, v)]

        return results
