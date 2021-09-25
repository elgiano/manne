import pyaudio
import numpy as np
from pyaudio import PyAudio
from time import sleep, time
from manne_render import ManneInterpolator, ManneSynth
from queue import Queue
import rtmidi
import threading


class MidiDispatcher(threading.Thread):
    def __init__(self):
        super(MidiDispatcher, self).__init__()
        self.midiin = rtmidi.MidiIn(name="manne")
        self.midiin.open_virtual_port("manne")
        self.latents = np.zeros(8)
        self._wallclock = time()
        self.queue = Queue()

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
                    if num in range(1, 8):
                        self.latents[num - 1] = val / 127

    def get_latents(self):
        return np.copy(self.latents)

    def stop(self):
        self.queue.put(None)


class InputFrameBuffer():
    def __init__(self, fftSize, maxFrames, dtype=np.float32):
        self.fftSize = fftSize
        self.buffer = np.zeros(fftSize)
        self.buffer_free = fftSize
        self.frames = Queue(maxFrames)

    def put(self, data):
        if len(data) <= self.buffer_free:
            self.buffer[self.fftSize - self.buffer_free:] = data
            self.frames.put(np.copy(self.buffer))
            self.buffer_free = self.fftSize
        else:
            self.put(data[:self.buffer_free])
            self.put(data[self.buffer_free:])


class OutputFrameBuffer():
    def __init__(self, block_size, maxFrames, dtype=np.float32):
        self.block_size = block_size
        self.buffer = np.zeros(block_size)
        self.buffer_pos = 0
        self.frames = Queue(maxFrames)

    def get(self, n):
        out = np.zeros(n)
        for out_i in range(n):
            out[out_i] = self.buffer[self.buffer_pos]
            self.buffer_pos += 1
            if self.buffer_pos == self.block_size:
                self.buffer_pos = 0
                next_frame = self.frames.get()
                if next_frame:
                    self.buffer = np.copy(self.frames.get())
                else:
                    self.buffer = np.zeros(self.block_size)
        return out


class ManneRealtimeInterpolator():

    def __init__(self, model_name, rate=44100, block_size=4096):
        self.renderer = ManneInterpolator(model_name, verbose=False)
        print('[ManneRealtime] renderer ready')
        self.interp = 0.5
        self.rate = rate
        self.sample_width = 2
        self.window_size = 4096 * 8
        self.block_size = self.window_size
        self.block_time = self.block_size / self.rate
        self.num_buffered_windows = 20
        self.init_audio()

    def audio_callback(self, in_frames, frame_count, time_info, status_flags):
        start = time()
        channels = int(len(in_frames) / (self.sample_width * frame_count))
        buffer = np.frombuffer(in_frames, dtype=np.int16)
        buffer = buffer.astype(np.float32, order='C') / 32768.0
        channels = buffer.reshape(frame_count, channels).transpose()
        # print(frame_count, buffer.shape)
        fftTime = time()
        fft = [self.renderer.process_audio(y) for y in channels]
        predictTime = time()
        out = self.renderer.interpolate(*fft[0], *fft[1], self.interp)
        fftTime = predictTime - fftTime
        predictTime = time() - predictTime
        # print(out.shape)
        # out = channels[0]
        out_frames = (out * 32768).astype(np.int16, order='C')
        out_frames = np.repeat(out_frames, 2)
        # print(frame_count, out.shape)
        out_frames = out_frames.tobytes()
        end = time()
        print(self.block_time, end - start, fftTime, predictTime)
        return (out_frames, pyaudio.paContinue)

    def init_audio(self):
        self.p = PyAudio()
        self.audio_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=self.rate,
            output=True, input=True,
            frames_per_buffer=self.block_size,
            stream_callback=self.audio_callback
        )


class ManneRealtimeSynth():

    def __init__(self, model_name, rate=44100, block_size=4096):
        self.renderer = ManneSynth(model_name, verbose=False)
        print('[ManneRealtime] synth ready')
        self.latent = np.zeros(8)
        self.rate = rate
        self.sample_width = 2
        self.window_size = 4096
        self.block_size = self.window_size
        self.block_time = self.block_size / self.rate

    def start(self):
        self.init_audio()
        self.midi = MidiDispatcher()
        self.midi.start()

    def stop(self):
        self.midi.stop()
        self.midi.join()

    def audio_callback(self, in_frames, frame_count, time_info, status_flags):
        start = time()
        latents = self.midi.get_latents() * \
            np.ones((self.block_size // self.window_size * 2, 8))
        # print(latents)
        out = self.renderer.decode(latents, self.rate, self.window_size)
        predictTime = time() - start
        # print(out.shape)
        # out = channels[0]
        out_frames = (out * 32768).astype(np.int16, order='C')
        out_frames = np.repeat(out_frames, 2)
        # print(frame_count, out.shape)
        out_frames = out_frames.tobytes()
        end = time()
        #print(self.block_time, end - start, predictTime)
        return (out_frames, pyaudio.paContinue)

    def init_audio(self):
        self.p = PyAudio()
        self.audio_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=self.rate,
            output=True, input=False,
            frames_per_buffer=self.block_size,
            stream_callback=self.audio_callback
        )


# m = ManneRealtimeInterpolator('models/vae_tusk')
m = ManneRealtimeSynth('models/vae_tusk')
try:
    m.start()
    while True:
        sleep(1)
except KeyboardInterrupt:
    m.stop()
    print('')
finally:
    print("Exit.")
