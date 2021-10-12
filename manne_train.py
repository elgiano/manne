from manne_dataset import ManneDataset
import argparse
from os.path import join
from os import getcwd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LambdaCallback
from manne_model import ManneModel
from time import time


def mse(inputs, outputs):
    return tf.keras.losses.mse(inputs[:, :outputs.shape[1]], outputs)


def sse(inputs, outputs):
    diff = (inputs[:, :outputs.shape[1]] - outputs) ** 2
    return diff.sum(axis=-1)


class ManneTrain:
    def __init__(self, args, print_summaries=True):
        self.train_data = []
        self.val_data = []
        self.test_data = []

        self.n_epochs = args['n_epochs']
        self.net_type = args['net_type']
        self.batch_size = args['batch_size']
        self.train_size = args['train_size']
        self.val_size = args['val_size']
        self.dataset_name = args['dataset_name']
        self.skip = args['skip']

        self.save_history = args['save_history']

        if args['distribute']:
            self.distribution_strategy = tf.distribute.MirroredStrategy()
        else:
            self.distribution_strategy = tf.distribute.get_strategy()

        self.load_dataset(self.dataset_name, self.skip)
        args['input_size'] = self.input_size
        args['output_size'] = self.output_size

        with self.distribution_strategy.scope():
            self.model = ManneModel(args, print_summaries)

    def train_and_evaluate(self):
        with self.distribution_strategy.scope():
            self.train_net()
            self.evaluate_net()

    # DATASET

    def load_dataset(self, filename, skip_connection):
        # filepath = join(getcwd(), 'frames', filename)
        print(f"[ManneTrain] Loading dataset: {filename}")
        dataset = ManneDataset(filename)
        self.input_size = dataset.feature_size
        self.augmentation_length = dataset.augmentations_size
        self.output_size = self.input_size - self.augmentation_length
        self.fft_size = (self.output_size - 1) * 2
        print(f"fftSize: {self.fft_size}")
        print(
            f"frames: {dataset.dataset_size} (size: {dataset.feature_size})")
        print(
            f"augmentations: {dataset.augmentations} (size: {dataset.augmentations_size})")

        (train_data, val_data, test_data) = dataset.get_splits(
            self.train_size, self.val_size, self.batch_size)
        self.train_data = self.data_to_dataset(train_data)
        self.val_data = self.data_to_dataset(val_data)
        self.test_data = self.data_to_dataset(test_data)

    def data_to_dataset(self, data):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = tf.data.Dataset.zip((dataset, dataset))
        return dataset.batch(self.batch_size)
    # TRAINING

    def compile_model(self):
        adam_rate = 5e-4

        if self.net_type == 'vae':
            def neg_log_likelihood(x, rv_x):
                return -(rv_x).log_prob(x[:, :self.output_size])

            self.model.network.compile(optimizer=Adam(
                learning_rate=adam_rate), loss=neg_log_likelihood, metrics=[mse])
        else:
            self.model.network.compile(optimizer=Adam(
                learning_rate=adam_rate), loss=mse, metrics=[mse])

    def train_net(self):

        # model_path = join('models', self.model_name)
        # if not isdir(model_path):
        #     mkdir(model_path)
        # checkpoint_filepath = join(
        #     model_path, self.model_name + "_epoch{epoch}.h5")

        # checkpoint_cb = ManneCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=False,
        #     save_best_only=True,
        # )
        # checkpoint_cb.set_submodels(self.encoder, self.decoder)

        # checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=False,
        #     save_best_only=True,
        # )

        wsse_weights = np.arange(
            2, self.output_size + 1) / np.arange(1, self.output_size) - 1
        wsse_weights = np.hstack((1, wsse_weights))

        def wsse(inputs, outputs):
            diff = (inputs[:, :outputs.shape[1]] - outputs) ** 2
            return np.dot(diff, wsse_weights)

        # on_epoch_end callback for training
        alpha = tf.Variable(0.3)
        beta = tf.Variable(3e-7)

        def change_params(epoch, logs):
            if epoch <= 5 and epoch % 1 == 0:
                beta.assign_add(2e-5)
            if epoch == 30:
                alpha.assign(0)

        cp_cb = LambdaCallback(on_epoch_end=change_params)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        callbacks = [cp_cb, early_stop]

        self.compile_model()
        train_start_time = time()
        h = self.model.network.fit(self.train_data,
                                   epochs=self.n_epochs,
                                   validation_data=self.val_data,
                                   callbacks=callbacks
                                   )
        train_dur = time() - train_start_time
        self.model.save_model()
        if self.save_history:
            self.model.save_history(extra={'train_dur': train_dur})

    # EVALUATION

    def evaluate_net(self):
        print('\n')
        print('Evaluating performance on validation and test sets')
        a = self.model.network.evaluate(self.val_data, verbose=1)
        b = self.model.network.evaluate(self.test_data, verbose=1)
        print('\n')
        val_metrics = "Validation\n\n"
        for idx in range(len(self.model.network.metrics_names)):
            print('Validation ' + self.model.network.metrics_names[idx])
            print(a[idx])
            val_metrics += f"{self.model.network.metrics_names[idx]}: {a[idx]}\n"
        print('\n')
        test_metrics = "Testing\n\n"
        for idx in range(len(self.model.network.metrics_names)):
            print('Testing ' + self.model.network.metrics_names[idx])
            print(b[idx])
            test_metrics += f"{self.model.network.metrics_names[idx]}: {b[idx]}\n"
        print('\n')
        print('Plotting network reconstructions')

        num_plots = 10
        valset_eval_in = self.get_samples(self.val_data, num_plots)
        testset_eval_in = self.get_samples(self.test_data, num_plots)

        valset_eval = self.model.reconstruct(valset_eval_in)
        testset_eval = self.model.reconstruct(testset_eval_in)

        print('Printing PDFs')
        self.plot_pdf('val', valset_eval_in, valset_eval, val_metrics)
        self.plot_pdf('test', testset_eval_in, testset_eval, test_metrics)

    def get_samples(self, dataset, n):
        dataset = list(dataset.unbatch())
        np.random.shuffle(dataset)
        samples = np.array(dataset[:n])[:, 0]
        if self.skip is True:
            # separate inputs
            return [samples[:, 0], samples[:, 1]]
        else:
            return samples

    def plot_pdf(self, pdf_name, original, predicted, note):
        if self.skip is True:
            original = original[0]
        figs_per_page = 5
        num_bins = predicted.shape[1]
        x = np.arange(num_bins) * (44100 / self.fft_size)
        filename = join('eval', f'{self.model.name}_{pdf_name}.pdf')
        fig = None
        with PdfPages(filename) as pdf:
            plt.figure(figsize=(8.5, 11))
            plt.axis('off')
            plt.text(0.5, 0.5, f"{self.model.name}\n\n{note}",
                     ha='center', va='center')
            pdf.savefig()
            for (fig_n, (in_data, out_data)) in enumerate(zip(original, predicted)):
                in_data = in_data[:num_bins]
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
        self.compile_model()
        self.evaluate_net()

    def save_latents(self):
        self.model.save_latents(self.train_data, name='train')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--net_type', type=str, default='ae')
    parser.add_argument('--skip', action="store_true")
    parser.add_argument('--latent_size', type=int, default=8)
    parser.add_argument('-e', '--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--distribute', action="store_true")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--save_history', action="store_true")
    parser.add_argument('--save_latents', action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    t = ManneTrain(vars(args))
    if args.mode == 'plot':
        t.just_plot()
    elif args.mode == 'save_latents':
        t.compile_model()
        t.save_latents()
    else:
        t.train_and_evaluate()
        if args.save_latents:
            t.save_latents()
