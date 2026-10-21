"""Bright summer-pop score for the 40s trip film — 120 BPM, C major, I–V–vi–IV.
Beat 0.5s / bar 2s, so every cut in index.html lands on the grid.
Form: intro+riser (0–4) · groove (4–20) · build+countdown (20–22) · drop (22–34) · finale (34–40).
Sidechain pump on the synths, SFX on the edit, and the two water-park clips' own sound mixed underneath."""
import numpy as np, wave
from scipy.signal import butter, sosfilt, fftconvolve

SR, DUR, BPM = 48000, 40.0, 120
BEAT = 60 / BPM; BAR = 4 * BEAT
N = int(SR * DUR)
rng = np.random.default_rng(7)
drm = np.zeros((N, 2)); syn = np.zeros((N, 2)); lead = np.zeros((N, 2)); sfx = np.zeros((N, 2)); send = np.zeros((N, 2))

def mtof(m): return 440 * 2 ** ((m - 69) / 12)
def tt(d): return np.arange(int(SR * d)) / SR
def noise(d): return rng.uniform(-1, 1, int(SR * d))
def filt(x, kind, f, order=2): return sosfilt(butter(order, f, btype=kind, fs=SR, output='sos'), x)
def env(d, dec, att=0.003): t = tt(d); return np.minimum(1, t / att) * np.exp(-t / dec)
def add(bus, t0, sig, g=1.0, pan=0.0, rev=0.0):
    i = int(round(t0 * SR))
    if i >= N or i < 0: return
    sig = sig[: N - i] * g
    l, r = np.cos((pan + 1) * np.pi / 4) * 1.414, np.sin((pan + 1) * np.pi / 4) * 1.414
    bus[i:i + len(sig), 0] += sig * l; bus[i:i + len(sig), 1] += sig * r
    if rev: send[i:i + len(sig), 0] += sig * l * rev; send[i:i + len(sig), 1] += sig * r * rev

# ---------- instruments ----------
def saw(f, t, det=(-0.006, 0, 0.006)):
    s = 0
    for d in det: ph = f * (1 + d) * t + rng.uniform(); s = s + 2 * (ph - np.floor(ph + .5))
    return s / len(det)
def stab(ms, d=0.22, cut=3200):
    t = tt(d); s = sum(saw(mtof(m), t) for m in ms) / len(ms)
    return filt(s, 'low', cut) * np.minimum(1, t / 0.004) * np.exp(-t / 0.09)
def pad(ms, d, cut=1400):
    t = tt(d); s = sum(saw(mtof(m), t) for m in ms) / len(ms)
    return filt(s, 'low', cut) * np.minimum(1, t / 0.3) * np.clip((d - t) / 0.3, 0, 1)
def bass(m, d=0.22):
    t = tt(d); f = mtof(m); ph = f * t
    s = 0.7 * np.sin(2 * np.pi * ph) + 0.4 * filt(2 * (ph - np.floor(ph + .5)), 'low', 900)
    return s * np.minimum(1, t / 0.004) * np.exp(-t / 0.18)
def marimba(m, d=0.6):
    f, t = mtof(m), tt(d)
    s = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.28) + 0.3 * np.sin(2 * np.pi * f * 4 * t) * np.exp(-t / 0.05)
    return s * np.minimum(1, t / 0.001) * 0.5
def pluck(m, d=0.35):
    t = tt(d); f = mtof(m); ph = f * t; sq = np.sign(np.sin(2 * np.pi * ph)) * 0.5 + 0.5 * np.sin(2 * np.pi * ph)
    return filt(sq, 'low', 4200) * np.minimum(1, t / 0.002) * np.exp(-t / 0.12) * 0.45
def glock(m, d=1.0):
    f, t = mtof(m), tt(d)
    return (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.45) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.12)) * 0.3
def kick():
    t = tt(0.4); f = 46 + 110 * np.exp(-t / 0.025)
    return np.tanh(1.6 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2)) * 0.95
def clap():
    s = np.zeros(int(SR * 0.3))
    for k, o in enumerate((0, 0.011, 0.022)):
        c = filt(noise(0.25), 'band', [900, 4000]) * env(0.25, 0.012 if k < 2 else 0.09, 0.0005); i = int(o * SR); s[i:i + len(c)] += c
    return s * 0.6
def hat(open_=False):
    d = 0.25 if open_ else 0.05
    return filt(noise(d), 'high', 7500) * env(d, 0.08 if open_ else 0.015, 0.0005) * 0.5
def snare():
    t = tt(0.2)
    return (filt(noise(0.2), 'band', [1500, 6000]) * 0.6 + np.sin(2 * np.pi * 190 * t) * 0.5) * env(0.2, 0.06, 0.0005)
def crash():
    return filt(noise(2.2), 'high', 4500) * env(2.2, 0.7, 0.001) * 0.45
def riser(d):
    t = tt(d); x = filt(noise(d), 'band', [500, 2500])
    out = np.zeros_like(t); n = 2048
    for s in range(0, len(t), n):  # sweep a band-pass upward
        fc = 400 * (24 ** (s / len(t))); out[s:s + n] = filt(noise(len(t[s:s + n]) / SR), 'band', [fc, min(fc * 2.2, 20000)])
    return out * (t / d) ** 2 * 0.7 + 0 * x
def whoosh(d=0.4):
    t = tt(d); e = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 2
    return filt(noise(d), 'band', [700, 4500]) * e * 0.55
def boom():
    t = tt(1.4); f = 38 + 50 * np.exp(-t / 0.08)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5)
def splashfx():
    d = 0.9; t = tt(d)
    return (filt(noise(d), 'band', [600, 6000]) * np.exp(-t / 0.18) + filt(noise(d), 'high', 3000) * np.exp(-t / 0.4) * 0.4) * 0.8
def blip(m=84):
    t = tt(0.16); f = mtof(m) * (1 + 0.6 * t / 0.16)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(0.16, 0.05, 0.002) * 0.4
def shutter():
    s = np.zeros(int(SR * 0.12))
    for o, g in [(0, 1), (0.05, 0.7)]:
        i = int(o * SR); c = filt(noise(0.03), 'band', [1800, 7000]) * env(0.03, 0.006, 0.0005); s[i:i + len(c)] += c * g
    return s

# ---------- harmony / hook ----------
CH = [([60, 64, 67, 72], 36), ([59, 62, 67, 71], 43), ([60, 64, 69, 72], 45), ([60, 65, 69, 72], 41)]  # C G Am F
HOOK = [[(0, 76), (1, 79), (2, 81), (3, 79), (5, 76), (6, 74), (7, 72)],
        [(0, 74), (2, 79), (3, 74), (4, 71), (6, 74), (7, 76)],
        [(0, 72), (1, 76), (2, 81), (3, 79), (5, 76), (6, 77), (7, 76)],
        [(0, 77), (2, 76), (3, 74), (4, 72), (6, 74), (7, 79)]]
NB = 20
kicks = []
for n in range(NB):
    t0 = n * BAR; voic, root = CH[n % 4]
    intro, groove, build, drop, fin = n < 2, 2 <= n < 10, n == 10, 11 <= n < 17, n >= 17
    if n == 19:  # final hit
        for j, m in enumerate([36, 48] + voic + [76, 79, 84]):
            add(syn, t0 + j * 0.01, pad([m], 2.0, 3000), 0.35, -0.5 + j * 0.12, rev=0.4)
        add(drm, t0, kick(), 1.0); add(drm, t0, crash(), 0.8, rev=0.3); add(lead, t0, glock(84, 1.8), 0.5, rev=0.5)
        continue
    # pads under everything (filtered in intro/build)
    add(syn, t0, pad(voic, BAR + 0.1, 900 if (intro or build) else 1800), 0.22 if not build else 0.3, 0, rev=0.3)
    if not intro and not build:
        # four-on-the-floor + clap + hats
        for b in range(4): add(drm, t0 + b * BEAT, kick(), 0.9); kicks.append(t0 + b * BEAT)
        for b in (1, 3): add(drm, t0 + b * BEAT, clap(), 0.55, 0.05, rev=0.2)
        for k in range(8): add(drm, t0 + k * BEAT / 2, hat(k % 2 == 1), 0.35 if k % 2 else 0.18, 0.3)
        if drop or fin:
            for k in range(16):
                if k % 4: add(drm, t0 + k * BEAT / 4, hat(), 0.12, -0.35)
        # bouncy octave bass on 8ths
        for k in range(8): add(syn, t0 + k * BEAT / 2, bass(root + (12 if k % 2 else 0)), 0.75)
        # off-beat chord stabs
        for k in range(1, 8, 2): add(syn, t0 + k * BEAT / 2, stab([m + 12 for m in voic], cut=4200 if drop else 3000), 0.35, 0.2 * (-1) ** k, rev=0.25)
    # hook: marimba in groove, + square pluck an octave up in the drop
    if (2 <= n < 10 and n % 8 < 6) or drop or fin:
        for k, m in HOOK[n % 4]:
            add(lead, t0 + k * BEAT / 2, marimba(m), 0.55, -0.2, rev=0.3)
            if drop: add(lead, t0 + k * BEAT / 2, pluck(m + 12), 0.4, 0.25, rev=0.35)
    if intro:
        for k in range(8):
            if n == 1 and k >= 4: add(drm, t0 + k * BEAT / 2, clap(), 0.25 + k * 0.05)
        for k, m in HOOK[n % 4]: add(lead, t0 + k * BEAT / 2, glock(m + 12), 0.3, 0.2, rev=0.5)
    if build:  # accelerating snare roll, then a gap before the drop
        for k in range(28):
            tk = t0 + BAR * (1 - (1 - k / 28) ** 1.6) * 0.86
            add(drm, tk, snare(), 0.18 + 0.4 * k / 28, 0, rev=0.2)
        for k, d in enumerate([0, 0.25, 0.5]): add(sfx, t0 + 1.0 + d, blip(79 + k * 5), 0.9)  # 3-2-1

# ---------- sidechain pump on synths ----------
duck = np.ones(N); ti = np.arange(N) / SR
for k in kicks:
    i = int(k * SR); j = min(N, i + int(0.3 * SR)); duck[i:j] = np.minimum(duck[i:j], 1 - 0.55 * np.exp(-(ti[i:j] - k) / 0.09))
syn *= duck[:, None]

# ---------- ear-candy on the edit ----------
add(sfx, 0.0, riser(4.0), 0.6, rev=0.3)
add(sfx, 4.0, boom(), 0.9); add(drm, 4.0, crash(), 0.7, rev=0.3)
add(sfx, 20.0, riser(1.75), 0.8, rev=0.3)
add(sfx, 22.0, boom(), 1.0); add(drm, 22.0, crash(), 0.8, rev=0.3)
add(drm, 30.0, crash(), 0.5, rev=0.3)
for c, pan in [(6, -0.6), (12, 0), (14, 0.4), (16, 0.6), (20, -0.6), (25, 0.6), (30, 0), (34, 0)]:
    add(sfx, c - 0.22, whoosh(), 0.6, pan, rev=0.2)
for t0, m in [(4.3, 84), (6.6, 86), (14.35, 88), (16.0, 84), (16.5, 88), (17.0, 91), (18.3, 86), (30.3, 84), (30.8, 88), (35.3, 84), (35.6, 88)]:
    add(sfx, t0, blip(m), 0.55, rng.uniform(-.4, .4))
for t0 in (11.05, 24.0, 29.1): add(sfx, t0, splashfx(), 0.7, 0.2, rev=0.3)
add(sfx, 13.0, shutter(), 0.9, 0.3)
for k in range(8): add(sfx, 32 + k * 0.25, shutter(), 0.35, -0.4 + k * 0.1)

# ---------- the clips' own sound (kids squealing, water) ----------
def wav(p):
    with wave.open(p) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, w.getnchannels()) / 32768
    return a if a.shape[1] == 2 else np.repeat(a, 2, 1)
clip = np.zeros((N, 2))
for path, t0, src0, d in [('assets/v1.wav', 22.0, 0.0, 2.75), ('assets/v2.wav', 25.0, 0.8, 5.0)]:
    try: a = wav(path)[int(src0 * SR):int((src0 + d) * SR)]
    except FileNotFoundError: continue
    f = np.minimum(1, np.minimum(np.arange(len(a)), len(a) - np.arange(len(a))) / (0.08 * SR))[:, None]
    i = int(t0 * SR); clip[i:i + len(a)] += a * f / (np.abs(a).max() + 1e-9) * 0.35

# ---------- reverb + master ----------
L = int(2.0 * SR); tl = np.arange(L) / SR
ir = np.stack([filt(rng.uniform(-1, 1, L), 'low', 6000) * np.exp(-tl / 0.45) for _ in range(2)], 1); ir[:int(0.015 * SR)] = 0
wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1) * 0.07
mix = drm * 0.9 + syn * 0.8 + lead * 0.9 + sfx * 0.8 + clip + wet
mix = np.stack([filt(mix[:, c], 'high', 28) for c in range(2)], 1)
fade = np.ones(N); fo = int(1.4 * SR); fade[-fo:] = np.linspace(1, 0, fo) ** 1.3
mix *= fade[:, None]
mix = np.tanh(mix / np.percentile(np.abs(mix), 99.9) * 1.1)
mix /= np.max(np.abs(mix)) / 0.93
with wave.open('assets/audio.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print('ok rms dB', 20 * np.log10(np.sqrt((mix ** 2).mean())))
