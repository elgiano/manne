import pyaudio
import numpy as np
from pyaudio import PyAudio
from time import sleep, time
from manne_render import ManneInterpolator
from queue import Queue, Empty
import rtmidi
import threading
import librosa
from manne_realtime import ManneMidiThread, ManneRealtime


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


class ManneRealtimeInterpolator(ManneRealtime):

    def __init__(self, model_name, rate=44100, num_channels=2, block_size=4096):
        super().__init__(model_name, rate, num_channels, block_size, wants_inputs=True)

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


if __name__ == '__main__':
    m = ManneRealtimeInterpolator('models/ae_skip_tusk_single_chroma+octave')
    print('[ManneRealtime] renderer ready')
    try:
        m.start()
        while True:
            sleep(1)
    except KeyboardInterrupt:
        m.stop()
        print('')
    finally:
        print("Exit.")
