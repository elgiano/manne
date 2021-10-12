import pyaudio
import threading
import argparse
from manne_realtime import ManneMidiThread, OutputFrameBuffer, ManneRealtime
import numpy as np
# from functools import partial

#
# def make_float32_default(fn):
#     return partial(fn, dtype=np.float32)
#
#
# np.array = make_float32_default(np.array)
# np.random.rand = make_float32_default(np.random.rand)
# np.linspace = make_float32_default(np.linspace)
#


class ManneRealtimeSynth(ManneRealtime):

    def __init__(self, model_name, rate=44100, num_channels=2, block_size=2048, fft_size=4096):
        super().__init__(model_name, rate, num_channels,
                         fft_size, block_size, wants_inputs=False)

    def start_midi(self):
        self.midi = ManneMidiThread(self.renderer.latent_size)
        self.midi.start()

    def stop(self):
        self.audio_stream.close()
        # self.gen.stop()
        self.midi.stop()
        self.midi.join()

    def start_generator(self):
        # self.gen = GeneratorThread(
        #     self.renderer, self.block_size, gen_blocks=1, max_frames=1)
        # self.gen.start()
        self.gen = Generator(
            self.renderer, self.block_size, gen_blocks=1, max_frames=1)

    def audio_callback(self, in_frames, frame_count, time_info, status_flags):
        latents, note, amp = self.midi.get_all_params()
        self.gen.set_latents(latents)
        self.gen.set_note(note)
        self.set_amp(amp)
        out = self.gen.get_audio_block()
        amp = self._get_updated_amp(len(out))
        out *= amp
        out_frames = (out * 32768).astype(np.int16, order='C')
        out_frames = np.repeat(out_frames, 2).tobytes()
        return (out_frames, pyaudio.paContinue)


class Generator():
    def __init__(self, renderer, block_size, gen_blocks=1, max_frames=1, sr=44100, fft_size=4096, fft_hop=None):
        if fft_hop is None:
            fft_hop = block_size
        self.renderer, self.sr, self.fft_size, self.fft_hop = renderer, sr, fft_size, fft_hop
        self.block_frames = block_size / fft_size * (fft_size // fft_hop)
        self.gen_frames = int(gen_blocks * self.block_frames)
        print(
            f"[Generator] making {self.gen_frames} frames per call to generate {gen_blocks} audio blocks ({self.block_frames} frames/block)")
        self.out = OutputFrameBuffer(block_size, max_frames * gen_blocks)
        self.running = False
        self.latents = np.zeros(self.renderer.latent_size)
        self.new_latents = None
        self.chroma = 6
        self.octave = 4

        self.has_skip = self.renderer.model_has_skip
        self.has_chroma = 'chroma' in self.renderer.model_augmentations
        self.has_octave = 'octave' in self.renderer.model_augmentations

        print(
            f"[Generator] model skip:{self.has_skip}, chroma:{self.has_chroma}, octave:{self.has_octave}")
        # latent transition durs
        self.trans_frames = int(self.block_frames)
        if self.trans_frames < 1:
            self.trans_frames = 1
        self.cont_frames = self.gen_frames - self.trans_frames
        if self.cont_frames < 0:
            self.cont_frames = 0

    def set_latents(self, new_latents):
        if np.any(new_latents != self.latents):
            self.new_latents = new_latents

    def set_note(self, midinote):
        self.chroma = midinote % 12
        self.octave = midinote // 12

    def _get_updated_latents(self):
        latent_shape = (-1, len(self.latents))
        if self.new_latents is not None:
            trans = np.linspace(
                self.latents, self.new_latents, self.trans_frames)
            cont = np.tile(self.new_latents, self.cont_frames).reshape(
                latent_shape)
            self.latents = self.new_latents
            self.new_latents = None
            # self.out.frames.queue.clear()
            latents = np.vstack((trans, cont))
        else:
            latents = np.tile(self.latents, self.gen_frames).reshape(
                latent_shape)
        return latents

    def get_audio_block(self):
        self.out.add_frames(self.generate_audio())
        return self.out.get_audio_block()

    def generate_audio(self):
        chroma = self.chroma
        octave = self.octave
        latents = self._get_updated_latents()
        if self.has_skip and self.has_chroma:
            if self.has_octave:
                out = self.renderer.note(chroma, octave, latents, self.sr,
                                         self.fft_size, self.fft_hop, rtpghi=False, istft=False)
            else:
                out = self.renderer.chroma(chroma, latents, self.sr,
                                           self.fft_size, self.fft_hop, rtpghi=False, istft=False)
        else:
            out = self.renderer.decode(
                latents, self.sr, self.fft_size, self.fft_hop, rtpghi=False, istft=False)
        return out


class GeneratorThread(threading.Thread, Generator):
    def __init__(self, renderer, block_size, gen_blocks=1, max_frames=1, sr=44100, fft_size=4096, fft_hop=2048):
        threading.Thread.__init__(self)
        Generator.__init__(self, renderer, block_size,
                           gen_blocks, max_frames, sr, fft_size, fft_hop)

    def get_audio_block(self):
        return self.out.get_audio_block()

    def run(self):
        print("[Generator] Starting generator thread")
        self.running = True
        while self.running:
            self.out.add_frames(self.generate_audio())
            # print(self.out.frames.qsize())

    def stop(self):
        self.running = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    args = vars(parser.parse_args())

    model_name = args.get(
        'model_name', 'models/ae_skip_tusk_single_chroma+octave')

    m = ManneRealtimeSynth(model_name)
    print('[ManneRealtime] synth ready')
    m.run_main()
