from manne_render import ManneInterpolator, ManneSynth, ManneRender
from os.path import isdir, join, basename, dirname, splitext
from os import makedirs, listdir
import numpy as np


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
                noise = np.random.rand(note_samples, 8)
            elif latent == 'line':
                noise = np.linspace(np.random.rand(
                    8), np.random.rand(8), note_samples)
            m.render_note(outname, chroma, octave, noise,
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

    # render_test_reconstructions(model_name, a + b, test_result)
    # render_test_interpolations(model_name, a, b, out_dir, rtpghi=None)
    # render_test_interpolations(model_name, a, b, out_dir, rtpghi=True)
    # render_test_interpolations(model_name, a, b, out_dir, rtpghi=False)
    # render_test_notes(model_name, out_dir, latent='noise', 100)
    # render_test_notes(model_name, out_dir, latent='lines', 100)

    for dur in [10, 20, 30]:
        render_test_sequences(model_name, test_result, dur)

if __name__ == '__main__':
    render_tests('ae_skip_tusk_single_chroma+octave')
