import pyaudio
import numpy as np
import threading
from time import sleep
from manne_realtime import ManneMidiThread, OutputFrameBuffer, ManneRealtime


class ManneRealtimeSynth(ManneRealtime):

    def __init__(self, model_name, rate=44100, num_channels=2, block_size=4096):
        super().__init__(model_name, rate, num_channels, block_size, wants_inputs=False)

    def start_midi(self):
        self.midi = ManneMidiThread(self.renderer.latent_size)
        self.midi.start()

    def start_generator(self):
        self.gen = GeneratorThread(
            self.renderer, self.block_size, gen_blocks=1, max_frames=1)
        self.gen.start()

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


class GeneratorThread(threading.Thread):
    def __init__(self, renderer, block_size, gen_blocks=1, max_frames=1, sr=44100, fft_size=4096, fft_hop=2048):
        super(GeneratorThread, self).__init__()
        self.renderer, self.sr, self.fft_size, self.fft_hop = renderer, sr, fft_size, fft_hop
        self.block_frames = (fft_size // fft_hop)
        self.num_gen_frames = block_size // fft_size * gen_blocks
        self.gen_blocks = self.num_gen_frames * self.block_frames
        self.out = OutputFrameBuffer(block_size, max_frames * gen_blocks)
        self.running = False
        self.latents = np.zeros(self.renderer.latent_size)
        self.new_latents = None
        self.chroma = 6
        self.octave = 4

        self.has_skip = self.renderer.model_has_skip
        if self.renderer.model_has_skip:
            self.has_chroma = 'chroma' in self.renderer.model_augmentations
            self.has_octave = 'octave' in self.renderer.model_augmentations

    def set_latents(self, new_latents):
        if np.any(new_latents != self.latents):
            self.new_latents = new_latents

    def set_note(self, midinote):
        self.chroma = midinote % 12
        self.octave = midinote // 12

    def _get_updated_latents(self):
        if self.new_latents is not None:
            trans = np.linspace(
                self.latents, self.new_latents, self.block_frames)
            cont = np.tile(self.new_latents,
                           self.gen_blocks - self.block_frames).reshape((-1, len(self.latents)))
            self.latents = self.new_latents
            self.new_latents = None
            # self.out.frames.queue.clear()
            return np.vstack((trans, cont))

        else:
            return np.tile(self.latents, self.gen_blocks).reshape((-1, len(self.latents)))

    def _gen_audio(self):
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

    def run(self):
        print("[Generator] Starting generator thread")
        self.running = True
        while self.running:
            self.out.add_frames(self._gen_audio())
            # print(self.out.frames.qsize())

    def stop(self):
        self.running = False


if __name__ == "__main__":
    m = ManneRealtimeSynth('models/ae_skip_tusk_single_chroma+octave')
    print('[ManneRealtime] synth ready')
    m.run_main()
