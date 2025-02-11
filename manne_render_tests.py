from manne_render import ManneInterpolator, ManneSynth, ManneRender
from os.path import isdir, join, basename, dirname, splitext
from os import makedirs, listdir
import numpy as np
import argparse
from manne_dataset import get_augmentations_from_filename, get_skip_from_filename


def get_sources(path):
    if isdir(path):
        return [join(path, n) for n in listdir(path)]
    else:
        return [path]


def render_test_reconstructions(model_name, files, out_dir):
    print("[TestRender] Reconstructions")
    r = ManneRender(join('models', model_name))
    out_dir = join(out_dir, "reconstructions")
    if not isdir(out_dir):
        makedirs(out_dir)
    for in_file in files:
        r.render(join(out_dir, basename(in_file)), in_file)


def render_test_interpolations(model_name, a, b, out_dir, reverse_interpolations=False, rtpghi=None):
    print("[TestRender] Interpolations")
    m = ManneInterpolator(join('models', model_name))
    combinations = np.array(np.meshgrid(a, b)).T.reshape(-1, 2)
    out_dir = join(out_dir, "interpolations")
    if rtpghi is True:
        print("[TestRender] Phase: RTPGHI")
        out_dir = join(out_dir, "rtpghi")
    elif rtpghi is False:
        print("[TestRender] Phase: Noise")
        out_dir = join(out_dir, "noisephase")
    else:
        print("[TestRender] Phase: from source A")

    for files in combinations:
        names = [splitext(basename(n))[0] for n in files]
        outname = '%'.join(names) + '.wav'
        outname = join(out_dir, outname)
        if not isdir(dirname(outname)):
            makedirs(dirname(outname))
        m.render(outname, files[0], files[1], 0.5)
        if reverse_interpolations:
            names.reverse()
            outname = '%'.join(names) + '.wav'
            outname = join(out_dir, outname)
            if not isdir(dirname(outname)):
                makedirs(dirname(outname))
            m.render(outname, files[1], files[0], 0.5)


def render_test_notes(model_name, out_dir, latent='noise', note_samples=100):
    print("[TestRender] Notes")
    m = ManneSynth(join('models', model_name))
    chromatic = ['c', 'c#', 'd', 'd#', 'e',
                 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    note_dir = join(out_dir, 'note')
    if not isdir(note_dir):
        makedirs(note_dir)
    for (chroma, name) in enumerate(chromatic):
        for octave in range(8):
            outname = f'{latent}_{octave}-{name}'
            outname = join(note_dir, outname) + ".wav"
            if latent == 'noise':
                noise = np.random.rand(note_samples, m.latent_size)
            elif latent == 'line':
                noise = np.linspace(np.random.rand(
                     m.latent_size), np.random.rand(m.latent_size), note_samples)
            m.render_note(outname, chroma, octave, noise,
                          44100, 4096, 1024, rtpghi=True)


def render_test_chromas(model_name, out_dir, latent='noise', note_samples=100):
    print("[TestRender] Notes")
    m = ManneSynth(join('models', model_name))
    chromatic = ['c', 'c#', 'd', 'd#', 'e',
                 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    note_dir = join(out_dir, 'note')
    if not isdir(note_dir):
        makedirs(note_dir)
    for (chroma, name) in enumerate(chromatic):
        for octave in range(8):
            outname = f'{latent}_{octave}-{name}'
            outname = join(note_dir, outname) + ".wav"
            if latent == 'noise':
                noise = np.random.rand(note_samples, m.latent_size)
            elif latent == 'line':
                noise = np.linspace(np.random.rand(
                   m.latent_size), np.random.rand(m.latent_size), note_samples)
            m.render_chroma(outname, chroma, noise,
                            44100, 4096, 1024, rtpghi=True)


def render_test_sequences(model_name, out_dir, dur=10):
    print("[TestRender] Sequences")
    m = ManneSynth(join('models', model_name))
    outname = f'chromatic_lines{dur}'
    outname = join(out_dir, 'seq', outname) + ".wav"
    m.render_chromatic_line(outname, 44100, 4096,
                            1024, rtpghi=True, dur=dur)


def render_tests(model_name, reverse_interpolations=False):
    test_source = "renders/test"
    test_result = join(test_source, 'renders', model_name)

    a = get_sources(join(test_source, "tu"))
    b = get_sources(join(test_source, "sk"))

    render_test_reconstructions(model_name, a + b, test_result)
    render_test_interpolations(model_name, a, b, test_result, rtpghi=None)
    render_test_interpolations(model_name, a, b, test_result, rtpghi=True)
    render_test_interpolations(model_name, a, b, test_result, rtpghi=False)
    if not get_skip_from_filename(model_name):
        print("Model has no skip connection. Skipping notes renders")
        return

    (augs, n_augs) = get_augmentations_from_filename(model_name)
    if 'chroma' in augs and 'octave' in augs:
        render_notes_fn = render_test_notes
    elif 'chroma' in augs:
        render_notes_fn = render_test_chromas
    else:
        print("Model has no augmentations. Skipping notes renders")
        return

    render_notes_fn(model_name, test_result, 'noise', 100)
    render_notes_fn(model_name, test_result, 'line', 100)
    # for dur in [10, 20, 30]:
    #     render_test_sequences(model_name, test_result, dur)


def render_test_line(model_name):
    print("[TestRender] Eval line", model_name)
    m = ManneSynth(join('models', model_name))
    outdir = join('renders', 'eval-line')
    outname = join(outdir, model_name + ".wav")
    if not isdir(outdir):
        makedirs(outdir)

    sr = 44100
    fft_size = 4096
    fft_hop = 1024
    dur = 30
    frames = dur * sr // fft_hop
    latent = np.linspace(np.zeros(m.model.latent_size),
                         np.ones(m.model.latent_size), frames)
    if m.model_has_skip:
        if 'octave' in m.model_augmentations:
            print('note')
            m.render_note(outname, 7, 4, latent, sr, fft_size, fft_hop, False)
        else:
            print('chroma')
            m.render_chroma(outname, 7, latent, sr, fft_size, fft_hop, False)
    else:
        m.render(outname, latent, sr, fft_size, fft_hop, False)


def render_test_scale(model_name):
    print("[TestRender] Eval line", model_name)
    m = ManneSynth(join('models', model_name))
    outdir = join('renders', 'eval-scale')
    outname = join(outdir, model_name + ".wav")
    if not isdir(outdir):
        makedirs(outdir)

    sr = 44100
    fft_size = 4096
    fft_hop = 1024
    note_dur = 2
    note_frames = note_dur * sr // fft_hop
    if m.model_has_skip:
        if 'octave' in m.model_augmentations:
            m.render_chromatic_line(
                outname, sr, fft_size, fft_hop, False, note_frames)
        else:
            m.render_chromaline(outname, sr, fft_size,
                                fft_hop, False, note_frames)
    else:
        print('Model has no skip_connection. Skipping (haha)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str)
    args = vars(parser.parse_args())

    model_name = args.get(
        'model_name', 'models/ae_skip_tusk_single_chroma+octave')

    render_tests(model_name)
