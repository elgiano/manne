from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import os
from os.path import splitext

from manne_dataset import ManneDatasetReader
from manne_checkpoint import ManneCheckpoint

global TRAINING_SIZE
global VALIDATION_SIZE
TRAINING_SIZE = 0.6  # 0.85
VALIDATION_SIZE = 0.2  # 0.075

global alpha
global beta
beta = tf.Variable(3e-7)
alpha = tf.Variable(0.3)


# on_epoch_end callback for training
def change_params(epoch, logs):
    if epoch <= 5 and epoch % 1 == 0:
        beta.assign_add(2e-5)
    if epoch == 30:
        alpha.assign(0)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--net_type', type=str, default='vae')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--distribute', type=bool, default=False)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--skip', type=bool, default=False)
    return parser.parse_args()


class ManneTrain:
    def __init__(self, args):
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.encoder = []
        self.decoder = []
        self.network = []
        self.encoder_widths = []
        self.decoder_widths = []

        self.z_mean = K.placeholder(shape=(8,))
        self.z_log_var = K.placeholder(shape=(8,))
        self.beta_changer = []

        self.n_epochs = args.n_epochs
        self.net_type = args.net_type
        self.skip = args.skip
        self.filename = args.filename
        if args.distribute:
            self.distribution_strategy = tf.distribute.MirroredStrategy()
        else:
            self.distribution_strategy = tf.distribute.get_strategy()

    def do_everything(self):
        self.load_dataset()
        with self.distribution_strategy.scope():
            self.define_net()
            self.make_net()
            self.train_net()
        self.evaluate_net()
        # self.save_latents()

    # DATASET

    def load_dataset(self):
        global TRAINING_SIZE
        global VALIDATION_SIZE

        skipstr = "skip"
        if not self.skip:
            skipstr = "noskip"
        self.model_name = f"{self.net_type}_{skipstr}_{splitext(self.filename)[0]}"
        filename = 'frames/' + self.filename
        filepath = os.path.join(os.getcwd(), filename)
        print(f"Loading dataset: {filename}")
        dataset_reader = ManneDatasetReader(filepath, self.skip)
        self.feature_length = dataset_reader.feature_size
        self.augmentation_length = dataset_reader.augmentation_size
        self.num_bins = self.feature_length - self.augmentation_length
        self.fft_size = (self.num_bins - 1) * 2
        (train, val, test) = dataset_reader.get_splits(
            TRAINING_SIZE, VALIDATION_SIZE, 64)
        self.train_data = train
        self.val_data = val
        self.test_data = test
        print(f"fftSize: {self.fft_size}")
        print(
            f"frames: {dataset_reader.dataset_size} (size: {dataset_reader.feature_size})")
        print(
            f"augmentations: {dataset_reader.augmentations} (size: {dataset_reader.augmentation_size})")

    # ARCHITECTURE

    def define_net(self):
        if self.net_type == 'vae':
            l2_penalty = 0
        else:
            l2_penalty = 1e-7

        # 8 Neuron Model from the paper
        self.encoder_widths = [1024, 512, 256, 128, 64, 32, 16, 8]
        self.decoder_widths = [16, 32, 64, 128, 256, 512, 1024]

        # Lighter weight model
        # self.encoder_widths = [512,256,128,64,8]
        # self.decoder_widths = [64,128,256,512]

        alpha_val = 0.1

        input_spec = Input(shape=(self.feature_length,))
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
            z_mean = Dense(
                self.encoder_widths[-1], input_shape=(self.encoder_widths[-1],), name='z_mean')(encoded)
            z_log_var = Dense(self.encoder_widths[-1], input_shape=(
                self.encoder_widths[-1],), name='z_log_var')(encoded)
            z = Lambda(self.sampling, output_shape=(
                self.encoder_widths[-1],), name='z')([z_mean, z_log_var])
            self.encoder = Model(input_spec, [z_mean, z_log_var, z])
            self.z_mean = z_mean
            self.z_log_var = z_log_var
        else:
            self.encoder = Model(input_spec, encoded)

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
        decoded = Dense(units=2049,
                        activation='relu',
                        kernel_regularizer=l2(l2_penalty))(decoded)
        self.decoder = Model(input_latent, decoded)

    def make_net(self):
        auto_input = Input(shape=(self.feature_length, ))
        encoded = self.encoder(auto_input)

        if self.net_type == 'vae':
            latents = encoded[2]
        else:
            latents = encoded

        if self.skip is True:
            onehot_input = Input(shape=(self.augmentation_length,))
            new_latents = Concatenate()([latents, onehot_input])
            decoded = self.decoder(new_latents)
            self.network = Model(
                inputs=[auto_input, onehot_input], outputs=decoded)
        else:
            decoded = self.decoder(latents)
            self.network = Model(inputs=[auto_input], outputs=decoded)

        print('\n net summary \n')
        self.network.summary()
        print('\n encoder summary \n')
        self.encoder.summary()
        print('\n decoder summary \n')
        self.decoder.summary()

    # Reparametrization trick for vae encoder
    def sampling(self, args):
        self.z_mean, self.z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(self.z_mean))
        return self.z_mean + tf.math.exp(0.5 * self.z_log_var) * epsilon

    # TRAINING
    def get_loss(self, inputs, outputs):
        global beta
        reconstruction_loss = mse(inputs[:, :self.num_bins], outputs)
        kl_loss = 1 + self.z_log_var - \
            tf.math.square(self.z_mean) - tf.math.exp(self.z_log_var)
        kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5 * beta
        vae_loss = tf.math.reduce_sum(reconstruction_loss + kl_loss)
        return vae_loss

    def my_mse(self, inputs, outputs):
        return mse(inputs[:, :self.num_bins], outputs)

    def my_kl(self, inputs, outputs):
        kl_loss = 1 + self.z_log_var - \
            tf.math.square(self.z_mean) - tf.math.exp(self.z_log_var)
        kl_loss = tf.math.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss

    def train_net(self):

        model_path = os.path.join('models', self.model_name)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        checkpoint_filepath = os.path.join(
            model_path, self.model_name + "_epoch{epoch}.h5")

        checkpoint_cb = ManneCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            save_best_only=True,
        )

        checkpoint_cb.set_submodels(self.encoder, self.decoder)

        adam_rate = 5e-4

        if self.net_type == 'vae':
            beta_changer = LambdaCallback(on_epoch_end=change_params)
            self.network.compile(optimizer=Adam(
                learning_rate=adam_rate), loss=self.get_loss, metrics=[self.my_mse, self.my_kl])
            self.network.fit(self.train_data,
                             epochs=self.n_epochs,
                             validation_data=self.val_data,
                             callbacks=[beta_changer, checkpoint_cb]
                             )
        else:
            alpha_changer = LambdaCallback(on_epoch_end=change_params)
            self.network.compile(optimizer=Adam(
                learning_rate=adam_rate), loss=self.my_mse, metrics=[self.my_mse])
            self.network.fit(self.train_data,
                             epochs=self.n_epochs,
                             validation_data=self.val_data,
                             callbacks=[alpha_changer, checkpoint_cb]
                             )
        self.encoder.save(f'models/{self.model_name}_trained_encoder.h5')
        self.decoder.save(f'models/{self.model_name}_trained_decoder.h5')

    # EVALUATION

    def evaluate_net(self):
        print('\n')
        print('Evaluating performance on validation and test sets')
        a = self.network.evaluate(self.val_data, verbose=1)
        b = self.network.evaluate(self.test_data, verbose=1)
        print('\n')
        val_metrics = "Validation\n\n"
        for idx in range(len(self.network.metrics_names)):
            print('Validation ' + self.network.metrics_names[idx])
            print(a[idx])
            val_metrics += f"{self.network.metrics_names[idx]}: {a[idx]}\n"
        print('\n')
        test_metrics = "Testing\n\n"
        for idx in range(len(self.network.metrics_names)):
            print('Testing ' + self.network.metrics_names[idx])
            print(b[idx])
            test_metrics += f"{self.network.metrics_names[idx]}: {b[idx]}\n"
        print('\n')
        print('Plotting network reconstructions')

        num_plots = 10
        valset_eval_in = self.get_samples(self.val_data, num_plots)
        testset_eval_in = self.get_samples(self.test_data, num_plots)

        valset_eval = self.network.predict(valset_eval_in, verbose=1)
        testset_eval = self.network.predict(testset_eval_in, verbose=1)

        print('Printing PDFs')
        self.plot_pdf('val', valset_eval_in, valset_eval, val_metrics)
        self.plot_pdf('test', testset_eval_in, testset_eval, test_metrics)

    def get_samples(self, dataset, n):
        dataset = dataset.unbatch()
        if self.skip is True:
            # separate inputs
            samples_x = list(dataset.map(
                lambda i, o: i[0]).as_numpy_iterator())
            samples_a = list(dataset.map(
                lambda i, o: i[1]).as_numpy_iterator())
            idx = np.round(np.linspace(0, len(samples_x) - 1, n)).astype(int)
            samples_x = [s for (i, s) in enumerate(samples_x) if i in idx]
            samples_a = [s for (i, s) in enumerate(samples_a) if i in idx]
            return [np.array(samples_x), np.array(samples_a)]
        else:
            samples = list(dataset.map(lambda i, o: i).as_numpy_iterator())
            idx = np.round(np.linspace(0, len(samples) - 1, n)).astype(int)
            samples = [s for (i, s) in enumerate(samples) if i in idx]
            return np.array(samples)

    def plot_pdf(self, pdf_name, original, predicted, note):
        if self.skip is True:
            original = original[0]
        figs_per_page = 5
        x = np.arange(self.num_bins) * (44100 / self.fft_size)
        filename = os.path.join('eval', f'{self.model_name}_{pdf_name}.pdf')
        fig = None
        with PdfPages(filename) as pdf:
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.text(0.5, 0.5, f"{self.model_name}\n\n{note}",
                     ha='center', va='center')
            pdf.savefig()
            for (fig_n, (in_data, out_data)) in enumerate(zip(original, predicted)):
                in_data = in_data[:self.num_bins]
                if fig_n % figs_per_page == 0:
                    if fig_n > 0:
                        fig.legend()
                        pdf.savefig()
                    fig = plt.figure(figsize=(8.5, 11))

                plt.subplot(figs_per_page, 1, fig_n % figs_per_page + 1)
                plt.ylabel('Spectral Magnitude')
                plt.xlabel('Frequency (Hz)')
                plt.ylim([0, 1.2])
                plt.xscale('log')
                line, = plt.plot(x, in_data)
                if fig_n % figs_per_page == 0:
                    line.set_label("Input Spectrum")
                line, = plt.plot(x, out_data, color='r')
                if fig_n % figs_per_page == 0:
                    line.set_label("Output Spectrum")
            fig.legend()
            pdf.savefig()
            plt.close()

    def just_plot(self):
        self.load_dataset()
        self.load_net()
        self.make_net()
        adam_rate = 5e-4
        self.network.compile(optimizer=Adam(
            learning_rate=adam_rate), loss=self.my_mse, metrics=[self.my_mse])
        self.evaluate_net()
        # self.save_latents()

    def load_net(self):
        enc_filename = os.path.join(
            os.getcwd(), 'models', self.model_name + '_trained_encoder.h5')
        self.encoder = load_model(enc_filename, custom_objects={
                                  'sampling': self.sampling}, compile=False)
        dec_filename = os.path.join(
            os.getcwd(), 'models', self.model_name + '_trained_decoder.h5')
        self.decoder = load_model(dec_filename, custom_objects={
                                  'sampling': self.sampling}, compile=False)


if __name__ == '__main__':
    args = get_arguments()
    my_manne = ManneTrain(args)
    if args.mode == 'plot':
        my_manne.just_plot()
    else:
        my_manne.do_everything()
