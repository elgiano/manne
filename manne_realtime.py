import pyaudio
import numpy as np
from pyaudio import PyAudio
from time import sleep
from manne_render import ManneSynth
from queue import Queue, Empty
import threading
import librosa
from os.path import basename

try:
    from collections.abc import Iterable
except ImportError:  # python 3.5
    from collections import Iterable

import socket

from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
from pythonosc.osc_bundle import OscBundle
from pythonosc.osc_server import OSCUDPServer
from pythonosc.dispatcher import Dispatcher

from typing import Union, Tuple


class OSCServerClient(OSCUDPServer):

    def __init__(self, server_address: Tuple[str, int], dispatcher: Dispatcher, allow_broadcast=True) -> None:
        super().__init__(server_address, dispatcher)
        if allow_broadcast:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def send(self, recv_addr, content: Union[OscMessage, OscBundle]) -> None:
        self.socket.sendto(content.dgram, recv_addr)

    def send_message(self, recv_addr, address: str, value: Union[int, float, bytes, str, bool, tuple, list]) -> None:
        builder = OscMessageBuilder(address=address)
        if value is None:
            values = []
        elif not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
            values = [value]
        else:
            values = value
        for val in values:
            builder.add_arg(val)
        msg = builder.build()
        self.send(recv_addr, msg)


class ManneOSCThread(threading.Thread):
    def __init__(self, synth, host='localhost', port=57130):
        super(ManneOSCThread, self).__init__()
        self.synth = synth
        self.latents = self.synth.get_empty_latents()
        self.loaded_model_names = self.synth.get_model_names()
        self.active_model_names = [
            self.loaded_model_names[0] for i in self.latents]
        self.note = np.tile(60.0, self.synth.num_channels).astype(np.float32)
        self.amp = np.tile(10.0, self.synth.num_channels).astype(np.float32)
        self.stereo = synth.stereo
        dispatcher = Dispatcher()
        dispatcher.map('/latents', self.set_latents)
        dispatcher.map('/latent', self.set_latent)
        dispatcher.map('/amp', self.set_amp)
        dispatcher.map('/noteOn', self.note_on)
        dispatcher.map('/noteOff', self.note_off)
        dispatcher.map('/model', self.choose_model)
        dispatcher.map('/stereoMix', self.set_stereo)
        dispatcher.map('/get_model_names', self.reply_model_names,
                       needs_reply_address=True)
        dispatcher.map('/get_active_models',
                       self.reply_active_models, needs_reply_address=True)
        print(f"[OSC] starting server at {host}:{port}")
        self.osc_server = OSCServerClient((host, port), dispatcher)

    def run(self):
        self.osc_server.serve_forever()

    def stop(self):
        self.osc_server.shutdown()

    def get_latents(self):
        return np.copy(self.latents)

    def get_note(self):
        return self.note

    def get_amp(self):
        return self.amp

    def set_amp(self, cmd, *args):
        ch = args[0]
        self.amp[ch] = np.float32(args[1])

    def set_latent(self, cmd, *args):
        # print('[OSC] set latent', args)
        ch = args[0]
        latent_num, new_val = args[1:3]
        self.latents[ch][latent_num] = new_val

    def set_latents(self, cmd, *args):
        # print('[OSC] set latents', args)
        ch = args[0]
        self.latents[ch] = np.array(args[1:])

    def note_on(self, cmd, *args):
        print('[OSC] set note', args[:2])
        ch = args[0]
        self.note[ch] = args[1]
        self.amp[ch] = args[2]

    def note_off(self, cmd, *args):
        ch = args[0]
        self.amp[ch] = 0

    def choose_model(self, cmd, *args):
        ch, model = args[:2]
        if type(model) in [int, float]:
            model = self.loaded_model_names[int(model)]
        self.active_model_names[ch] = model

    def set_stereo(self, cmd, *args):
        self.stereo = bool(args[0])

    def reply_model_names(self, client_addr, cmd, *args):
        self.osc_server.send_message(client_addr, cmd, self.loaded_model_names)

    def reply_active_models(self, client_addr, cmd, *args):
        self.osc_server.send_message(client_addr, cmd, self.active_model_names)

    def get_all_params(self):
        return self.latents, self.note, self.amp, self.active_model_names, self.stereo


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


class Generator():
    def __init__(self, renderer, block_size, gen_blocks=1, max_frames=1, sr=44100, fft_size=4096, fft_hop=None):
        if fft_hop is None:
            fft_hop = block_size
        self.sr, self.fft_size, self.fft_hop = sr, fft_size, fft_hop
        self.block_frames = block_size / fft_size * (fft_size // fft_hop)
        self.gen_frames = int(gen_blocks * self.block_frames)
        print(
            f"[Generator] making {self.gen_frames} frames per call to generate {gen_blocks} audio blocks ({self.block_frames} frames/block)")
        self.out = OutputFrameBuffer(block_size, max_frames * gen_blocks)
        self.running = False
        self.chroma = 6
        self.octave = 4

        self.set_renderer(renderer)

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

    def set_renderer(self, renderer):
        self.renderer = renderer
        self.latents = np.zeros(self.renderer.latent_size)
        self.new_latents = None

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
        frames = self.generate_audio()
        if len(frames) > 0:
            self.out.add_frames(frames)
        return self.out.get_audio_block()

    def generate_audio(self):
        chroma = self.chroma
        octave = self.octave
        latents = self._get_updated_latents()
        if self.has_skip and self.has_chroma:
            if self.has_octave:
                out = self.renderer.note(chroma, octave, latents, self.sr,
                                         self.fft_size, self.fft_hop, istft=False)
            else:
                out = self.renderer.chroma(chroma, latents, self.sr,
                                           self.fft_size, self.fft_hop, istft=False)
        else:
            out = self.renderer.decode(
                latents, self.sr, self.fft_size, self.fft_hop, istft=False)
        return out


class ManneRealtime():
    def __init__(self, model_name, rate=48000, num_channels=1, stereo=True, fft_size=4096, block_size=2048, wants_inputs=False, output_device=None):
        self.models = {}
        if type(model_name) is not list:
            model_name = [model_name]

        for path in model_name:
            name = basename(path)
            self.models[name] = ManneSynth(path, verbose=False)

        self.output_device = output_device
        self.rate, self.num_channels, self.stereo, self.wants_inputs = rate, num_channels, stereo, wants_inputs
        self.window_size = fft_size
        self.block_size = block_size
        self.amp = []
        self.new_amp = []
        self.gen = []
        first_model = self.models[self.get_model_names()[0]]
        for ch in range(self.num_channels):
            self.amp.append(10)
            self.new_amp.append(None)
            self.gen.append(Generator(
                first_model, self.block_size, gen_blocks=1, max_frames=1))

    def _get_updated_amp_ch(self, ch, block_size):
        if self.new_amp[ch] is not None:
            trans = np.linspace(
                self.amp[ch], self.new_amp[ch], block_size, dtype=np.float32)
            self.amp[ch] = trans[-1]
            self.new_amp[ch] = None
            return trans
        else:
            return self.amp[ch]

    def _get_updated_amp(self, block_size):
        return [self._get_updated_amp_ch(ch, block_size)
                for ch in range(self.num_channels)]

    def get_empty_latents(self):
        return [np.zeros(g.renderer.latent_size) for g in self.gen]

    def set_model(self, ch, model_name):
        # print(ch, self.num_channels)
        new_model = self.models[model_name]
        if self.gen[ch].renderer is not new_model:
            if new_model is not None:
                self.gen[ch].set_renderer(new_model)
            else:
                print(f'[ManneSynth] invalid model name "{model_name}"')

    def update_gen_params(self, model_names, new_latents, new_notes):
        for ch in range(self.num_channels):
            self.set_model(ch, model_names[ch])
            self.gen[ch].set_latents(new_latents[ch])
            self.gen[ch].set_note(new_notes[ch])

    def get_model_names(self):
        return list(self.models.keys())

    def set_amp(self, new_amps):
        for (ch, new_amp) in enumerate(new_amps):
            self.set_amp_ch(ch, new_amp)

    def set_amp_ch(self, ch, new_amp):
        if new_amp != self.amp[ch]:
            self.new_amp[ch] = new_amp

    def start(self):
        self.start_ctrl()
        self.init_audio()

    def stop(self):
        self.audio_stream.close()
        self.ctrl.stop()
        self.ctrl.join()

    def start_ctrl(self):
        self.ctrl = ManneOSCThread(self)
        self.ctrl.start()

    def init_audio(self):
        self.p = PyAudio()
        channels = self.num_channels
        if self.stereo:
            channels = 2
        if self.output_device:
            (self.output_device, deviceRate) = self.find_device(self.output_device)
            if self.output_device:
                self.rate = deviceRate

        print(self.rate)
        self.audio_stream = self.p.open(
            output_device_index=self.output_device,
            format=pyaudio.paFloat32,
            channels=channels,
            rate=self.rate,
            output=True, input=self.wants_inputs,
            frames_per_buffer=self.block_size,
            stream_callback=self.audio_callback
        )

    def find_device(self, device_name):
        num_devices = self.p.get_device_count()
        for i in range(num_devices):
            info = self.p.get_device_info_by_index(i)
            if info['name'] == device_name:
                print(f'[ManneRealtime] output device = {i}: {device_name}')
                print(info)
                return (i, int(info['defaultSampleRate']))
        print(f'[ManneRealtime] device {device_name} not found. Using default')
        return (None, None)

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

    def audio_callback(self, in_frames, frame_count, time_info, status_flags):
        pass
