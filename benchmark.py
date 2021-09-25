from manne_render import ManneInterpolator
import timeit

model_name = "models/vae_tusk"
render_path = "renders/line-straight"
source_b = render_path + "/tu/mm.wav"
source_a = render_path + "/sk/line_mid.wav"
interp = 0.5

if __name__ == '__main__':
    print(f"Loading model {model_name}")
    m = ManneInterpolator(model_name, False)
    mag_a, phase_a, remember_a = m.process_track(source_a)
    mag_b, phase_b, remember_b = m.process_track(source_b)

    def test(n=1):
        m.interpolate(mag_a[0:n], phase_a[:, 0:n], remember_a[0:n],
                      mag_b[0:n], phase_b[:, 0:n], remember_b[0:n], interp)

    print("Timing one frame interpolation")
    res = timeit.repeat(lambda: test(1), number=10, repeat=5)
    print(f"Fastest: {min(res) / 10}")
    print(f"Slowest: {max(res) / 10}")

    print("Timing ten frames interpolation")
    res = timeit.repeat(lambda: test(10), number=10, repeat=5)
    print(f"Fastest: {min(res) / 10}")
    print(f"Slowest: {max(res) / 10}")
