import numpy as np
import librosa
import argparse
import os
from os.path import splitext
import tensorflow as tf
import pickle


class ManneDatasetReader():

    def __init__(self, filename, skip_connection=True):
        ext = splitext(filename)[-1]
        if ext == '.npy':
            self.load_npy(filename)
        elif ext == '.tfrecords':
            self.load_tfrecords(filename)
        else:
            print(
                f"[ManneDataset] Error: unknown extension '{ext}' for file '{filename}'")
            return

        self.augmentations = splitext(filename)[0].split("_")[-1].split('+')
        self.augmentation_size = 0
        if 'chroma' in self.augmentations:
            self.augmentation_size += 12
        if 'octave' in self.augmentations:
            self.augmentation_size += 8

        # Handling case where Keras expects two inputs
        if skip_connection is True:
            augs_only = self.dataset.map(
                lambda x: x[-self.augmentation_size:])
            self.dataset = tf.data.Dataset.zip(
                ((self.dataset, augs_only), self.dataset))
        else:
            self.dataset = tf.data.Dataset.zip((self.dataset, self.dataset))

    def load_npy(self, filename):
        frames = np.load(filename)
        self.dataset = tf.data.Dataset.from_tensor_slices(frames)
        self.dataset_size = len(frames)
        self.feature_size = frames.shape[1]

    def load_tfrecords(self, filename):
        info = self.load_metadata(filename)
        self.dataset = tf.data.TFRecordDataset([filename]).map(
            lambda x: self.decode(x, info['features']))
        self.dataset_size = info['num_records']
        self.feature_size = info['features']['stft'].shape

    def decode(self, example, schema, key=None):
        # print(example)
        if key is None:
            key = list(schema.keys())[0]
        return tf.io.parse_single_example(example, schema)[key]

    def load_metadata(self, filename):
        with open(filename + '.meta', 'rb') as r:
            meta = pickle.load(r)
        return meta

    def get_splits(self, train_ratio, val_ratio, batch_size=200, shuffle=True):
        train_size = int(train_ratio * self.dataset_size)
        test_size = int((1 - train_ratio - val_ratio) * self.dataset_size)

        data = self.dataset
        if shuffle:
            data = data.shuffle(self.dataset_size)
        train_data = data.take(train_size)
        test_data = data.skip(train_size)
        val_data = test_data.skip(test_size)
        test_data = test_data.take(test_size)

        print(self.dataset_size, train_size, test_size)
        print(len(list(test_data)))

        train_data = train_data.batch(batch_size, drop_remainder=True)
        val_data = val_data.batch(batch_size, drop_remainder=True)
        test_data = test_data.batch(batch_size, drop_remainder=True)
        return train_data, val_data, test_data


class ManneDatasetWriter():
    num_records = 0

    def __init__(self, path, feature_info, options=None):
        self.filename = path
        self.feature_info = feature_info
        self.writer = tf.io.TFRecordWriter(path, options)

    def __enter__(self):
        self.writer.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.writer.__exit__(type, value, traceback)
        self.close()

    def write(self, record):
        self.writer.write(record)
        self.num_records += 1

    def close(self):
        self.writer.close()
        self.writeMetadata()

    def getMetadata(self):
        return {
            'num_records': self.num_records,
            'features': self.feature_info
        }

    def writeMetadata(self):
        metadataFile = self.filename + '.meta'
        print(f'[ManneDatasetWriter] writing meta to: {metadataFile}')
        data = self.getMetadata()
        with open(metadataFile, 'wb') as outfile:
            pickle.dump(data, outfile)
        print('[ManneDatasetWriter] done')


def wavToFrames(filename_in, fft_size, fft_hop, augmentations):
    print(f"Loading audiofile: {filename_in}")
    y, sr = librosa.load(filename_in, sr=44100)
    # y = y[:44100 * 10]
    print("Calculating spectral frames...")
    D = librosa.stft(y, n_fft=fft_size, hop_length=fft_hop, window='hann')
    print(f"numBins: {D.shape[0]}, numFrames: {D.shape[1]}")
    temp = D[:, :]
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

    if 'chroma' in augmentations:
        print("Computing chroma augmentation")
        chroma = librosa.feature.chroma_stft(S=np.transpose(output))
        chroma = (chroma == chroma.max(axis=1)[:, None]).astype(int)
        chroma = np.transpose(chroma)
        output = np.hstack((output, chroma))

    if 'octave' in augmentations:
        print("Computing octave augmentation")
        pitch = librosa.yin(y, 27.5, 4187,
                            frame_length=fft_size, hop_length=fft_hop)
        octave = librosa.hz_to_octs(pitch).astype(int)
        octave[np.where(octave > 7)] = -1
        octave = np.eye(8)[octave]
        output = np.hstack((output, octave))

    np.random.shuffle(output)
    print(output.shape)
    return output


def save_npy(data, filename_out):
    print(f"Saving {filename_out}")
    np.save(filename_out, data)
    # np.save(filename_out+'_phase.npy',out_phase)


def save_tfrecords(frames, filename_out):
    print(f"Saving {filename_out}")
    feature_length = frames.shape[1]

    feature_info = {"stft": tf.io.FixedLenFeature(
        (feature_length), tf.float32)}
    # write records to a tfrecords file
    with ManneDatasetWriter(filename_out, feature_info) as writer:
        for frame in frames:
            # Construct the Example proto object
            example = tf.train.Example(features=tf.train.Features(feature={
                'stft': tf.train.Feature(float_list=tf.train.FloatList(value=frame))
            }))
            # Serialize the example to a string
            serialized = example.SerializeToString()
            # write the serialized objec to the disk
            writer.write(serialized)


def make_dataset(filename, fft_size, fft_hop, augmentations, format):
    filename_in = os.path.join(os.getcwd(), 'waves', filename + '.wav')
    frames = wavToFrames(filename_in, fft_size, fft_hop, augmentations)
    augname = "+".join(augmentations)
    filename_out = f"{filename}_{augname}"
    if format == "npy" or format == "both":
        filename_out = os.path.join(
            os.getcwd(), 'frames', filename_out + ".npy")
        save_npy(frames, filename_out)
    if format == "tfrecords" or format == "both":
        filename_out = os.path.join(
            os.getcwd(), 'frames', filename_out + ".tfrecords")
        save_tfrecords(frames, filename_out)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)
    parser.add_argument('--fft_size', type=int, default=4096,
                        help="FFT window size in samples (default: 4096)")
    parser.add_argument('--fft_hop', type=int, default=1024,
                        help="FFT hop in samples (default: 1024)")
    parser.add_argument('--format', type=str, default="npy",
                        help="npy, tfrecords or both (default: npy)")
    parser.add_argument('--augment', type=str, default="chroma",
                        help="chroma, octave or chroma+octave (default: chroma)")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    augmentations = []
    if args.augment == 'chroma' or args.augment == 'chroma+octave':
        augmentations = augmentations + ['chroma']
    if args.augment == 'octave' or args.augment == 'chroma+octave':
        augmentations = augmentations + ['octave']
    make_dataset(args.filename, args.fft_size,
                 args.fft_hop, augmentations, args.format)
