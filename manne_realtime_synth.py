import pyaudio
import argparse
from manne_realtime import ManneRealtime
import numpy as np
from os import listdir
from os.path import join
# from time import time
#
#
# class Timer():
#     def __init__(self):
#         self.start = time()
#         self.partials = []
#         self.partial_names = []
#
#     def partial(self, name=None):
#         self.partials.append(time())
#         self.partial_names.append(name)
#
#     def end(self):
#         self.end = time()
#         print(f'Total: {self.end - self.start}')
#         durs = np.diff([self.start] + self.partials)
#         for (n, dur) in enumerate(durs):
#             print(f"{self.partial_names[n]}:\t{dur}")


class ManneRealtimeSynth(ManneRealtime):

    def __init__(self, model_name, rate=44100, num_channels=1, stereo=True, block_size=2048, fft_size=4096, output_device=None):
        super().__init__(model_name, rate, num_channels, stereo,
                         fft_size, block_size, False, output_device)

    def audio_callback(self, in_frames, frame_count, time_info, status_flags):
        # timer = Timer()
        # latents, note, amp, models, self.stereo = self.ctrl.get_all_params()
        latents, note, amp, models, self.stereo = self.ctrl.get_changed_params()
        # timer.partial('get_params')
        self.update_gen_params(models, latents, note)
        # timer.partial('update_params')
        if amp is not None:
            self.set_amp(amp)
        amp = self._get_updated_amp(self.block_size)
        # timer.partial('update_amp')
        out_channels = np.array([gen.get_audio_block() * a
                                 for (gen, a) in zip(self.gen, amp)])
        # timer.partial('get_blocks')
        # out_channels = np.array([self.gen[ch].get_audio_block() * amp[ch]
        #                          for ch in range(self.num_channels)])
        if self.stereo:
            out = out_channels.sum(axis=0)
            out_frames = np.repeat(out, 2).tobytes()
        else:
            out_frames = out_channels.reshape(-1, order="F")  # interleave
        # timer.partial('get_frames')
        # timer.end()

        return (out_frames, pyaudio.paContinue)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir',
                        action="store_true", default=False)
    parser.add_argument('-o', '--output_device',
                        type=str, default=None)
    parser.add_argument('-c', '--channels',
                        type=int, default=1)
    parser.add_argument('-s', '--separate',
                        action="store_true", default=False,
                        help="output each channel separately (default: mix to stereo)")
    parser.add_argument('model_name', type=str)
    args = vars(parser.parse_args())

    model_name = args.get('model_name')
    if args['dir']:
        model_name = [join(model_name, p) for p in listdir(model_name)]
        print("Loading multiple models", model_name)

    m = ManneRealtimeSynth(
        model_name, num_channels=args['channels'], stereo=not args['separate'],
        output_device=args['output_device'])
    print('[ManneRealtime] synth ready')
    m.run_main()
