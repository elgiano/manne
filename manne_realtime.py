import pyaudio
import numpy as np
from pyaudio import PyAudio
from time import time, sleep
from manne_render import ManneSynth
from queue import Queue, Empty
import rtmidi
import threading
import librosa


class ManneMidiThread(threading.Thread):
    def __init__(self, num_latents, min_latent=0, max_latent=1):
        super(ManneMidiThread, self).__init__()
        self.midiin = rtmidi.MidiIn(name="manne")
        self.midiin.open_virtual_port("manne")
        self.latents = np.zeros(num_latents)
        self.note = 60
        self.amp = 10
        self._wallclock = time()
        self.queue = Queue()
        self.latent_range = max_latent - min_latent
        self.min_latent = min_latent

    def __call__(self, event, data=None):
        message, deltatime = event
        self._wallclock += deltatime
        # print("IN: @%0.6f %r", self._wallclock, message)
        self.queue.put((message, self._wallclock))

    def run(self):
        print("[MIDI] Attaching MIDI input callback handler.")
        self.midiin.set_callback(self)

        while True:
            event = self.queue.get()

            if event is None:
                self.midiin.close_port()
                break
            else:
                type, num, val = event[0]
                if type == 176:
                    if num == 1:
                        self.amp = val / 127 * 100
                    elif num in range(2, len(self.latents) + 1):
                        self.latents[num - 2] = val / 127 * \
                            (self.latent_range) + self.min_latent
                elif type == 144:
                    self.note = num
                    self.amp = val
                elif type == 128:
                    self.amp = 0
                else:
                    print('[MIDI] unknown event type', type, val)

    def get_latents(self):
        return np.copy(self.latents)

    def get_note(self):
        return self.note

    def get_amp(self):
        return self.amp

    def get_all_params(self):
        return np.copy(self.latents), self.note, self.amp

    def stop(self):
        self.queue.put(None)


class OutputFrameBuffer():
    def __init__(self, block_size, maxFrames, dtype=np.float32):
        self.block_size = block_size
        self.buffer = np.zeros(block_size)
        self.buffer_pos = 0
        self.frames = Queue(maxFrames)
        self.incomplete_block = None
        self.last_frame = None

    def add_frames(self, frames, fft_hop=2048):
        if self.last_frame is None:
            self.last_frame = frames[0]
        frames = np.vstack(([self.last_frame], frames))
        audio = np.float32(librosa.istft(
            frames.T, hop_length=fft_hop))
        self.add_audio_blocks(audio)
        self.last_frame = frames[-1]

    def add_audio_blocks(self, audio):
        blocks = np.split(audio, np.arange(
            self.block_size, len(audio), self.block_size))

        if self.incomplete_block is not None:
            self.incomplete_block = self.incomplete_block + blocks[0]
            self.frames.put(self.incomplete_block)
            self.incomplete_block = None
            blocks = blocks[1:]

        if len(audio) % self.block_size != 0:
            remainder = blocks[-1]
            self.incomplete_block = np.pad(
                remainder, (0, self.block_size - len(remainder)))
            blocks = blocks[:-1]

        for b in blocks:
            self.frames.put(b)

    def get_audio_block(self):
        try:
            return self.frames.get_nowait()
        except Empty:
            print("Warning: out buffer is empty")
            return np.zeros(self.block_size)


class ManneRealtime():
    def __init__(self, model_name, rate=44100, num_channels=2, block_size=4096, wants_inputs=False):
        self.renderer = ManneSynth(model_name, verbose=False)
        self.rate, self.num_channels, self.wants_inputs = rate, num_channels, wants_inputs
        self.window_size = 4096
        self.block_size = self.window_size
        # self.block_time = self.block_size / self.rate
        self.amp = 10
        self.new_amp = None

    def _get_updated_amp(self, block_size):
        if self.new_amp is not None:
            trans = np.linspace(
                self.amp, self.new_amp, block_size)
            self.amp = self.new_amp
            self.new_amp = None
            return trans
        else:
            return self.amp

    def set_amp(self, new_amp):
        if new_amp != self.amp:
            self.new_amp = new_amp

    def start(self):
        self.start_generator()
        self.init_audio()
        self.start_midi()

    def stop(self):
        self.gen.stop()
        self.midi.stop()
        self.midi.join()

    def init_audio(self):
        self.p = PyAudio()
        self.audio_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.num_channels,
            rate=self.rate,
            output=True, input=self.wants_inputs,
            frames_per_buffer=self.block_size,
            stream_callback=self.audio_callback
        )

    def run_main(self):
        try:
            self.start()
            while True:
                sleep(1)
        except KeyboardInterrupt:
            self.stop()
            print('')
        finally:
            print("Exit.")

    def start_generator(self):
        pass

    def start_midi(self):
        pass

    def audio_callback(self, in_frames, frame_count, time_info, status_flags):
        pass
