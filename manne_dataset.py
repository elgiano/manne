import numpy as np
import librosa
import argparse
import os
from os.path import splitext, basename, join, isfile


class ManneDataset():

    default_options = {
        'anal': 'stft',
        'num_bins': 2048,
        'hop': 2048,
        "augmentations": [],
        "normalize": True,
        "cqt_bins_per_octave": 24
    }

    default_cqt_options = {
        'anal': 'cqt',
        'num_bins': 240,
        'hop': 2048,
        "augmentations": [],
        "normalize": True,
        "cqt_bins_per_octave": 24
    }

    def __init__(self, wave_or_dataset_path, options={}):
        if wave_or_dataset_path.endswith(".wav"):
            self.make_npy_dataset(wave_or_dataset_path, options)
        elif wave_or_dataset_path.endswith(".npy"):
            self.load_npy_dataset(wave_or_dataset_path)

    def make_npy_dataset(self, wave_path, options={}):
        defaults = self.default_options
        if options.get('anal', None) == 'cqt':
            defaults = self.default_cqt_options
        self.options = {k: options.get(k, v)
                        for (k, v) in self.default_options.items()}
        self.filename = splitext(basename(wave_path))[0]

        print(f"Loading audiofile: {wave_path}")
        y, sr = librosa.load(wave_path, sr=44100)

        if self.options['anal'] == 'stft':
            fft_size = self.options['num_bins'] * 2
            frames = self.get_stft_frames(
                y, sr, fft_size, self.options['hop'], self.options['normalize'])
            frames = self.append_stft_augmentations(
                self.options['augmentations'], frames, y, fft_size, self.options['hop'])
        elif self.options['anal'] == 'cqt':
            f_min = 20
            interval = 2**(1 / self.options['cqt_bins_per_octave'])
            frames = self.get_cqt_frames(y, sr,
                                         f_min, self.options['num_bins'], self.options['cqt_bins_per_octave'],
                                         self.options['hop'], self.options['normalize'])
            frames = self.append_cqt_augmentations(
                self.options['augmentations'], frames, interval, f_min)

        np.random.shuffle(frames)
        print(frames.shape)
        self.frames = frames
        self.save_npy()

    def get_stft_frames(self, y, sr, fft_size, fft_hop, normalize):
        print("Calculating STFT frames...")
        D = librosa.stft(y, n_fft=fft_size, hop_length=fft_hop, window='hann')
        temp = D[:-1, :]
        print(f"numBins: {temp.shape[0]}, numFrames: {temp.shape[1]}")
        # phase = np.angle(temp)
        temp = np.abs(temp)
        if normalize:
            print("Normalizing magnitudes...")
            temp = temp / (temp.max(axis=0) + 0.000000001)
        else:
            temp = temp / (temp.max() + 0.000000001)

        temp = np.transpose(temp)
        # phase = np.transpose(phase)
        print("Filtering out empty frames", end="... ")
        output = temp[~np.all(temp == 0, axis=1)]
        # out_phase = phase[~np.all(temp == 0, axis=1)]
        print(f"found {output.shape[0] - temp.shape[0]} empty frames")
        return output

    def append_stft_augmentations(self, augmentations, mags, y, fft_size, fft_hop):
        output = mags
        if 'chroma' in augmentations:
            print("Computing chroma augmentation")
            chroma = librosa.feature.chroma_stft(S=np.transpose(mags))
            chroma = (chroma == chroma.max(axis=1)[:, None]).astype(int)
            chroma = np.transpose(chroma)
            output = np.hstack((mags, chroma))

        if 'octave' in augmentations:
            print("Computing octave augmentation")
            pitch = librosa.yin(y, 27.5, 4187,
                                frame_length=fft_size, hop_length=fft_hop)
            octave = librosa.hz_to_octs(pitch).astype(int)
            octave[np.where(octave > 7)] = -1
            octave = np.eye(8)[octave]
            output = np.hstack((output, octave))

        return output

    def get_cqt_frames(self, y, sr, f_min, num_bins, bins_per_octave, fft_hop, normalize):
        print("Calculating CQT frames...")
        D = librosa.cqt(y, sr, hop_length=fft_hop, fmin=f_min,
                        n_bins=num_bins, bins_per_octave=bins_per_octave)
        temp = D
        print(f"numBins: {temp.shape[0]}, numFrames: {temp.shape[1]}")
        # phase = np.angle(temp)
        temp = np.abs(temp)
        if normalize:
            print("Normalizing magnitudes...")
            temp = temp / (temp.max(axis=0) + 0.000000001)
        else:
            temp = temp / (temp.max() + 0.000000001)
        temp = np.transpose(temp)
        # phase = np.transpose(phase)
        print("Filtering out empty frames", end="... ")
        output = temp[~np.all(temp == 0, axis=1)]
        # out_phase = phase[~np.all(temp == 0, axis=1)]
        print(f"found {output.shape[0] - temp.shape[0]} empty frames")
        return output

    def append_cqt_augmentations(self, augmentations, cqt, interval, fmin):
        output = cqt
        pitch = cqt.argmax(axis=1) * interval + fmin
        if 'chroma' in augmentations:
            print("Computing chroma augmentation")
            chroma = np.eye(12)[pitch % 12]
            output = np.hstack((cqt, chroma))

        if 'octave' in augmentations:
            print("Computing octave augmentation")
            octave = librosa.hz_to_octs(pitch).astype(int)
            octave[np.where(octave > 7)] = -1
            octave = np.eye(8)[octave]
            output = np.hstack((output, octave))

        return output

    def save_npy(self):
        filename_out = self.get_filename_from_options()
        print(f"Saving {filename_out}")
        np.save(join('frames', filename_out), self.frames)
        # np.save(filename_out+'_phase.npy',out_phase)

    def load_npy_dataset(self, filename):
        print(f"[Dataset] Loading {filename}")
        if not isfile(filename):
            print(f"[Dataset] Error: {filename} not found")
            return
        self.filename = splitext(basename(filename))[0]
        self.frames = np.load(filename)
        self.get_options_from_filename(filename)

    def get_options_from_filename(self, filename):
        (anal, num_bins, bins_per_octave) = self.get_anal_from_filename(filename)
        (augs, aug_size) = self.get_augmentations_from_filename(filename)
        self.options = {
            'num_bins': self.frames.shape[1] - aug_size,
            'anal': anal,
            "augmentations": augs,
            "cqt_bins_per_octave": bins_per_octave
        }

    def get_filename_from_options(self):
        augname = "+".join(self.options['augmentations'])
        analname = f"{self.options['anal']}{self.options['num_bins']}"
        if self.options['anal'] == 'cqt':
            analname += f"-{self.options['cqt_bins_per_octave']}"
        normname = 'nonorm'
        if self.options['normalize']:
            normname = 'norm'

        return f"{self.filename.replace('_','-')}_{analname}_{normname}_{augname}.npy"

    def get_anal_from_filename(self, filename):
        filename = basename(filename)
        anal_info = filename.split("_")[1]
        if anal_info.startswith('stft'):
            num_bins = int(anal_info[len('stft'):])
            return ('stft', num_bins, 0)
        elif anal_info.startswith('cqt'):
            numbers = anal_info[len('cqt'):].split('-')
            num_bins = int(numbers[0])
            bins_per_octave = int(numbers[1])
            return ('cqt', num_bins, bins_per_octave)
        else:
            print(f"[Dataset] unrecognized anal '{anal_info}'")
            return None

    def get_augmentations_from_filename(self, filename):
        filename = basename(filename)
        augs = splitext(filename)[0].split("_")[-1].split('+')
        n = 0
        if 'chroma' in augs:
            n += 12
        if 'octave' in augs:
            n += 8
        return (augs, n)

    def get_feature_size(self):
        return self.frames.shape[1]

    def get_augmentations_size(self):
        return self.feature_size - self.options['num_bins']

    def get_dataset_size(self):
        return len(self.frames)

    def get_augmentations(self):
        return self.options['augmentations']

    augmentations = property(get_augmentations)
    augmentations_size = property(get_augmentations_size)
    feature_size = property(get_feature_size)
    dataset_size = property(get_dataset_size)

    def get_splits(self, train_ratio, test_ratio=None, batch_size=200, shuffle=True):
        data = self.frames
        if shuffle:
            np.random.shuffle(data)

        train_size = int(train_ratio * self.dataset_size)
        train_data = data[:train_size]

        if test_ratio is None:
            if train_ratio == 1:
                test_ratio = 0.1
            else:
                test_ratio = (1 - train_ratio) / 2

        test_size = int(test_ratio * self.dataset_size)

        if train_ratio == 1:
            test_data = data[-test_size:]
            val_data = []
        else:
            val_point = int((train_ratio + test_ratio) * self.dataset_size)
            test_data = data[train_size:val_point]
            val_data = data[val_point:]

        print(self.dataset_size, train_ratio,
              test_ratio, train_size, test_size)
        return train_data, val_data, test_data


def get_augmentations_from_filename(filename):
    augs = splitext(filename)[0].split("_")[-1].split('+')
    n = 0
    if 'chroma' in augs:
        n += 12
    if 'octave' in augs:
        n += 8
    return (augs, n)


def get_skip_from_filename(filename):
    return splitext(filename)[0].split("_")[1] == 'skip'


def get_latent_dim_from_filename(filename):
    try:
        return int(splitext(filename)[0].split("_")[2][1:])
    except ValueError:
        return 8


def wavToFrames(filename_in, fft_size, fft_hop, augmentations):
    print(f"Loading audiofile: {filename_in}")
    y, sr = librosa.load(filename_in, sr=44100)
    # y = y[:44100 * 10]
    print("Calculating spectral frames...")
    D = librosa.stft(y, n_fft=fft_size, hop_length=fft_hop, window='hann')
    temp = D[:-1, :]
    print(f"numBins: {temp.shape[0]}, numFrames: {temp.shape[1]}")
    # phase = np.angle(temp)
    print("Normalizing magnitudes...")
    temp = np.abs(temp)
    temp = temp / (temp.max(axis=0) + 0.000000001)

    temp = np.transpose(temp)
    # phase = np.transpose(phase)
    print("Filtering out empty frames", end="... ")
    output = temp[~np.all(temp == 0, axis=1)]
    # out_phase = phase[~np.all(temp == 0, axis=1)]
    print(f"found {output.shape[0] - temp.shape[0]} empty frames")

    output = append_augmentations(augmentations, output, y, fft_size, fft_hop)

    np.random.shuffle(output)
    print(output.shape)
    return output


def append_augmentations(augmentations, mags, y, fft_size, fft_hop):
    output = mags
    if 'chroma' in augmentations:
        print("Computing chroma augmentation")
        chroma = librosa.feature.chroma_stft(S=np.transpose(mags))
        chroma = (chroma == chroma.max(axis=1)[:, None]).astype(int)
        chroma = np.transpose(chroma)
        output = np.hstack((mags, chroma))

    if 'octave' in augmentations:
        print("Computing octave augmentation")
        pitch = librosa.yin(y, 27.5, 4187,
                            frame_length=fft_size, hop_length=fft_hop)
        octave = librosa.hz_to_octs(pitch).astype(int)
        octave[np.where(octave > 7)] = -1
        octave = np.eye(8)[octave]
        output = np.hstack((output, octave))

    return output


def save_npy(data, filename_out):
    print(f"Saving {filename_out}")
    np.save(filename_out, data)
    # np.save(filename_out+'_phase.npy',out_phase)


def make_dataset(filename, fft_size, fft_hop, augmentations, format):
    filename_in = os.path.join(os.getcwd(), 'waves', filename + '.wav')
    frames = wavToFrames(filename_in, fft_size, fft_hop, augmentations)
    augname = "+".join(augmentations)
    filename_out = f"{filename}_{augname}"
    if format == "npy" or format == "both":
        filename_out = os.path.join(
            os.getcwd(), 'frames', filename_out + ".npy")
        save_npy(frames, filename_out)
    # if format == "tfrecords" or format == "both":
    #     filename_out = os.path.join(
    #         os.getcwd(), 'frames', filename_out + ".tfrecords")
    #     save_tfrecords(frames, filename_out)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('-N', '--fft_size', type=int, default=4096,
                        help="FFT window size in samples (default: 4096)")
    parser.add_argument('-H', '--fft_hop', type=int, default=1024,
                        help="FFT hop in samples (default: 1024)")
    parser.add_argument('-n', '--normalize', action="store_true",
                        help="normalize each frame (default: normalizes whole dataset)")
    parser.add_argument('-a', '--anal', type=str, default="stft", choices=['stft', 'cqt'],
                        help="STFT or CQT analysis")
    parser.add_argument('-o', '--cqt_bins_per_octave', type=int, default=24,
                        help="CQT bins per octave")
    parser.add_argument('-+', '--augment', type=str, default="chroma",
                        help="chroma, octave or chroma+octave (default: chroma)")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    augmentations = []
    if args.augment == 'chroma' or args.augment == 'chroma+octave':
        augmentations = augmentations + ['chroma']
    if args.augment == 'octave' or args.augment == 'chroma+octave':
        augmentations = augmentations + ['octave']

    ManneDataset(args.filename, {
        'anal': args.anal,
        'num_bins': args.fft_size // 2,
        'hop': args.fft_hop,
        "augmentations": augmentations,
        "cqt_bins_per_octave": args.cqt_bins_per_octave,
        "normalize": args.normalize
    })

# import tensorflow as tf
# import pickle
# class ManneDatasetReader():
#
#     def __init__(self, filename, skip_connection=True):
#         ext = splitext(filename)[-1]
#         if ext == '.npy':
#             self.load_npy(filename)
#         # elif ext == '.tfrecords':
#         #     self.load_tfrecords(filename)
#         else:
#             print(
#                 f"[ManneDataset] Error: unknown extension '{ext}' for file '{filename}'")
#             return
#
#         self.augmentations, self.augmentation_size = get_augmentations_from_filename(
#             filename)
#
#         # Handling case where Keras expects two inputs
#         if skip_connection is True:
#             augs_only = self.dataset.map(
#                 lambda x: x[-self.augmentation_size:])
#             self.dataset = tf.data.Dataset.zip(
#                 ((self.dataset, augs_only), self.dataset))
#         else:
#             self.dataset = tf.data.Dataset.zip((self.dataset, self.dataset))
#
#     def load_npy(self, filename):
#         frames = np.load(filename)
#         self.dataset = tf.data.Dataset.from_tensor_slices(frames)
#         self.dataset_size = len(frames)
#         self.feature_size = frames.shape[1]
#
#     def load_tfrecords(self, filename):
#         info = self.load_metadata(filename)
#         self.dataset = tf.data.TFRecordDataset([filename]).map(
#             lambda x: self.decode(x, info['features']))
#         self.dataset_size = info['num_records']
#         self.feature_size = info['features']['stft'].shape
#
#     def decode(self, example, schema, key=None):
#         # print(example)
#         if key is None:
#             key = list(schema.keys())[0]
#         return tf.io.parse_single_example(example, schema)[key]
#
#     def load_metadata(self, filename):
#         with open(filename + '.meta', 'rb') as r:
#             meta = pickle.load(r)
#         return meta
#
#     def get_splits(self, train_ratio, val_ratio, batch_size=200, shuffle=True):
#         train_size = int(train_ratio * self.dataset_size)
#         test_size = int((1 - train_ratio - val_ratio) * self.dataset_size)
#
#         data = self.dataset
#         if shuffle:
#             data = data.shuffle(self.dataset_size)
#         train_data = data.take(train_size)
#         test_data = data.skip(train_size)
#         val_data = test_data.skip(test_size)
#         test_data = test_data.take(test_size)
#
#         train_data = train_data.batch(batch_size, drop_remainder=True)
#         val_data = val_data.batch(batch_size, drop_remainder=True)
#         test_data = test_data.batch(batch_size, drop_remainder=True)
#         return train_data, val_data, test_data

# class ManneDatasetWriter():
#     num_records = 0
#
#     def __init__(self, path, feature_info, options=None):
#         self.filename = path
#         self.feature_info = feature_info
#         self.writer = tf.io.TFRecordWriter(path, options)
#
#     def __enter__(self):
#         self.writer.__enter__()
#         return self
#
#     def __exit__(self, type, value, traceback):
#         self.writer.__exit__(type, value, traceback)
#         self.close()
#
#     def write(self, record):
#         self.writer.write(record)
#         self.num_records += 1
#
#     def close(self):
#         self.writer.close()
#         self.writeMetadata()
#
#     def getMetadata(self):
#         return {
#             'num_records': self.num_records,
#             'features': self.feature_info
#         }
#
#     def writeMetadata(self):
#         metadataFile = self.filename + '.meta'
#         print(f'[ManneDatasetWriter] writing meta to: {metadataFile}')
#         data = self.getMetadata()
#         with open(metadataFile, 'wb') as outfile:
#             pickle.dump(data, outfile)
#         print('[ManneDatasetWriter] done')

# def save_tfrecords(frames, filename_out):
#     print(f"Saving {filename_out}")
#     feature_length = frames.shape[1]
#
#     feature_info = {"stft": tf.io.FixedLenFeature(
#         (feature_length), tf.float32)}
#     # write records to a tfrecords file
#     with ManneDatasetWriter(filename_out, feature_info) as writer:
#         for frame in frames:
#             # Construct the Example proto object
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'stft': tf.train.Feature(float_list=tf.train.FloatList(value=frame))
#             }))
#             # Serialize the example to a string
#             serialized = example.SerializeToString()
#             # write the serialized objec to the disk
#             writer.write(serialized)
