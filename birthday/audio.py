"""Upbeat party-pop score for the 60s birthday film — 120 BPM, A major, I–V–vi–IV.
0–27s band + SFX on every cut · 27–52s the family's own singing (music out) ·
52–60s confetti drop finale. Reads assets/clipA.wav + clipB.wav, writes audio.wav."""
import numpy as np, wave
from scipy.signal import butter, sosfilt, fftconvolve

SR, DUR, BPM = 48000, 60.0, 120
BEAT = 60 / BPM; BAR = 4 * BEAT
CLIP0, CLIPA, FIN = 27.0, 21.6, 52.0
N = int(SR * DUR)
rng = np.random.default_rng(7)
mus = np.zeros((N, 2)); sfx = np.zeros((N, 2)); send = np.zeros((N, 2))

def mtof(m): return 440 * 2 ** ((m - 69) / 12)
def tt(d): return np.arange(int(SR * d)) / SR
def noise(d): return rng.uniform(-1, 1, int(SR * d))
def filt(x, kind, f, order=2): return sosfilt(butter(order, f, btype=kind, fs=SR, output='sos'), x)
def env(d, dec, att=0.003): t = tt(d); return np.minimum(1, t / att) * np.exp(-t / dec)
def add(bus, t0, sig, g=1.0, pan=0.0, rev=0.0):
    i = int(t0 * SR)
    if i >= N or i < 0: return
    sig = sig[: N - i] * g
    l, r = np.cos((pan + 1) * np.pi / 4) * 1.414, np.sin((pan + 1) * np.pi / 4) * 1.414
    bus[i:i + len(sig), 0] += sig * l; bus[i:i + len(sig), 1] += sig * r
    if rev: send[i:i + len(sig), 0] += sig * l * rev; send[i:i + len(sig), 1] += sig * r * rev
def wav(path):
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float) / 32768
        return x.reshape(-1, w.getnchannels())

# ---------- instruments ----------
def saw(f, t, det=(-0.006, 0.0, 0.006)):
    s = 0
    for d in det: ph = f * (1 + d) * t; s = s + 2 * (ph - np.floor(ph + .5))
    return s / len(det)
def pluck(m, d=0.35, bright=3500):
    t = tt(d); s = filt(saw(mtof(m), t), 'low', bright)
    return s * env(d, 0.12, 0.002) * 0.6
def glock(m, d=1.0):
    f, t = mtof(m), tt(d)
    s = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.45) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.12) \
        + 0.12 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.05)
    return s * np.minimum(1, t / 0.001) * 0.4
def lead(m, d):
    t = tt(d); f = mtof(m) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.25))
    s = filt(saw(f, t), 'low', 3200) * 0.6 + 0.4 * np.sin(2 * np.pi * f * t)
    return s * np.minimum(1, t / 0.01) * np.minimum(1, np.clip((d - t) / 0.05, 0, 1)) * 0.35
def pad(ms, d):
    t = tt(d); s = sum(saw(mtof(m), t) for m in ms) / len(ms)
    s = filt(s, 'low', 1400)
    return s * np.minimum(1, t / 0.3) * np.minimum(1, np.clip((d - t) / 0.4, 0, 1))
def kick(g=1.0):
    t = tt(0.4); f = 46 + 110 * np.exp(-t / 0.025)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2)
    s[:int(0.004 * SR)] += filt(noise(0.004), 'high', 2000) * 0.5
    return s * g
def clap():
    s = np.zeros(int(SR * 0.3))
    for o in (0, 0.011, 0.022):
        c = filt(noise(0.3 - o), 'band', [900, 4500]) * env(0.3 - o, 0.012 if o < 0.02 else 0.09, 0.0005)
        i = int(o * SR); s[i:i + len(c)] += c[:len(s) - i]
    return s * 0.8
def hat(open_=False):
    d = 0.35 if open_ else 0.06; return filt(noise(d), 'high', 7500) * env(d, 0.12 if open_ else 0.018, 0.0005) * 0.5
def snare():
    d = 0.25; t = tt(d)
    return (filt(noise(d), 'band', [1500, 6000]) * 0.7 + np.sin(2 * np.pi * 190 * t) * 0.4) * env(d, 0.07, 0.001)
def bass(m, d):
    t = tt(d); f = mtof(m)
    s = np.tanh(2.2 * (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)))
    return filt(s, 'low', 900) * np.minimum(1, t / 0.005) * np.minimum(1, np.clip((d - t) / 0.02, 0, 1)) * 0.5
def riser(d, f0=300, f1=7000):
    t = tt(d); n = noise(d); out = np.zeros_like(n); seg = int(0.02 * SR)
    for i in range(0, len(n), seg):
        fc = f0 * (f1 / f0) ** (i / len(n))
        out[i:i + seg] = filt(n[max(0, i - seg):i + seg], 'band', [fc * 0.7, min(fc * 1.4, SR / 2 - 100)])[-len(n[i:i + seg]):]
    return out * (t / d) ** 2 * 0.6
def whoosh(d=0.4):
    t = tt(d); e = np.sin(np.pi * t / d) ** 2
    return filt(noise(d), 'band', [500, 4000]) * e * 0.6
def impact():
    d = 1.6; t = tt(d); f = 38 + 60 * np.exp(-t / 0.06)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5) + filt(noise(d), 'low', 1200) * np.exp(-t / 0.15) * 0.4
def shutter():
    s = np.zeros(int(SR * 0.12))
    for o, g in [(0, 1), (0.055, 0.7)]:
        i = int(o * SR); c = filt(noise(0.03), 'band', [1800, 7000]) * env(0.03, 0.006, 0.0005); s[i:i + len(c)] += c * g
    return s
def popper():
    d = 0.9; t = tt(d); s = filt(noise(d), 'band', [600, 6000]) * env(d, 0.03, 0.0005) * 1.2
    for k in range(40):  # paper crackle
        i = int(rng.uniform(0.03, 0.8) * SR); c = filt(noise(0.006), 'high', 3000) * env(0.006, 0.002, 0.0002)
        s[i:i + len(c)] += c * rng.uniform(0.1, 0.4)
    return s
def tick(m=96): return glock(m, 0.25) * 0.6
def chime(ms, gap=0.06):
    out = np.zeros(int(SR * 2.2))
    for k, m in enumerate(ms): g = glock(m, 1.8); i = int(k * gap * SR); out[i:i + len(g)] += g
    return out

# ---------- harmony: A – E – F#m – D ----------
CH = [([57, 61, 64, 69], 45), ([56, 59, 64, 68], 40), ([57, 61, 64, 66], 42), ([57, 62, 66, 69], 38)]
HOOK = [[(0, 73, .5), (1, 76, .5), (1.5, 78, .5), (2, 76, 1), (3, 73, 1)],
        [(0, 71, .5), (1, 73, .5), (1.5, 76, .5), (2, 73, 1), (3, 71, 1)],
        [(0, 69, .5), (1, 73, .5), (1.5, 76, .5), (2, 78, 1), (3, 76, 1)],
        [(0, 74, .5), (1, 73, .5), (2, 71, .5), (2.5, 69, .5), (3, 68, 1)]]

def groove(n, t0, level):
    """one bar of the band. level 1=groove, 2=+hook, 3=chorus"""
    voic, root = CH[n % 4]
    add(mus, t0, pad([m + 12 for m in voic], BAR + 0.3), 0.10 + 0.04 * (level > 2), 0, rev=0.3)
    for b in range(4): add(mus, t0 + b * BEAT, kick(), 0.9)
    for b in (1, 3): add(mus, t0 + b * BEAT, clap(), 0.45, 0.1, rev=0.25)
    for k in range(8): add(mus, t0 + k * BEAT / 2, hat(level > 2 and k % 2 == 1), 0.22 if k % 2 else 0.1, -0.35)
    for k in range(8):  # pumping offbeat bass + plucks
        if k % 2: add(mus, t0 + k * BEAT / 2, bass(root, BEAT / 2 - 0.02), 0.75)
        for j, m in enumerate(voic[1:]):
            if k % 2 or level > 1: add(mus, t0 + k * BEAT / 2 + j * 0.006, pluck(m + 12), 0.17, (-0.4, 0, 0.4)[j], rev=0.15)
    if level >= 2:
        for b, m, d in HOOK[n % 4]: add(mus, t0 + b * BEAT, glock(m + 12), 0.4, 0.3, rev=0.35)
    if level >= 3:
        for b, m, d in HOOK[n % 4]: add(mus, t0 + b * BEAT, lead(m, d * BEAT * 2 - 0.03), 0.55, -0.15, rev=0.3)

# intro (0–4): pad + "Happy Birthday" teaser on glockenspiel (3/4 phrasing), riser into the drop
add(mus, 0.2, pad([57, 61, 64, 69], 3.9), 0.12, 0, rev=0.4)
t = 0.55
for m, d in [(64, .375), (64, .125), (66, .5), (64, .5), (69, .5), (68, 1.0)]:
    add(mus, t, glock(m + 12, 1.4), 0.55, 0.2, rev=0.5); t += d * 2 * BEAT
add(mus, 2.0, riser(2.0), 0.35, 0, rev=0.2)
for k in range(8): add(mus, 2.0 + k * BEAT / 4, snare(), 0.08 + k * 0.03, 0.1)  # build roll

# band 4–24
for n in range(2, 12):
    groove(n, n * BAR, 1 if n < 6 else 2 if n < 10 else 3)
# photo booth 24–27: three big stabs, then roll + riser into the live clip
for k, c in enumerate([0, 1, 3]):
    t0 = 24 + k; voic, root = CH[c]
    add(mus, t0, kick(1.1), 1.0); add(mus, t0, bass(root, 0.6), 0.8)
    for j, m in enumerate(voic): add(mus, t0 + j * 0.005, pluck(m + 12, 0.6, 5000), 0.3, (-0.5, -0.2, 0.2, 0.5)[j], rev=0.4)
    add(mus, t0, lead(voic[-1] + 12, 0.5), 0.45, 0, rev=0.4)
for k in range(12): add(mus, 25.5 + k * BEAT / 4, snare(), 0.06 + k * 0.025, -0.1)
add(mus, 25.0, riser(2.0, 400, 9000), 0.4, 0, rev=0.3)
add(mus, CLIP0, impact(), 0.55, 0, rev=0.6)
add(mus, CLIP0, chime([81, 85, 88, 93]), 0.35, 0, rev=0.8)

# finale 52–60: drop (riser rises under the cheer first)
add(mus, FIN - 2.0, riser(2.0, 300, 8000), 0.3, 0, rev=0.2)
for n in range(3):
    groove(n, FIN + n * BAR, 3)
voic, root = CH[0]  # final A chord ring-out on 58
add(mus, 58, kick(1.1), 1.0); add(mus, 58, bass(root, 1.5), 0.7)
add(mus, 58, pad([57, 61, 64, 69, 73, 76], 2.0), 0.22, 0, rev=0.6)
add(mus, 58, chime([81, 85, 88, 93, 97]), 0.45, 0, rev=0.8)

# ---------- SFX on the edit ----------
for c in (6, 8, 14, 16, 22): add(sfx, c - 0.2, whoosh(0.4), 0.45, 0.3, rev=0.15)          # whips
for c in (4, 10, 18): add(sfx, c - 0.25, whoosh(0.3), 0.3, -0.2); add(sfx, c, kick(0.6), 0.4)  # zooms
for c in (12, 20): add(sfx, c - 0.3, whoosh(0.5), 0.4, 0, rev=0.2)                        # slide ups
for k, tp in enumerate([0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]):  # title letter pops
    add(sfx, tp, tick(88 + (k % 5) * 2), 0.25, -0.6 + (k % 7) * 0.2, rev=0.3)
for c in (4.15, 12.15, 20.15): add(sfx, c, chime([88, 93, 97], 0.05), 0.3, -0.3, rev=0.5)  # chapter pills
add(sfx, 11.0, shutter(), 0.8, 0.3)
for c in (24, 25, 26): add(sfx, c, shutter(), 0.7, 0.2)
add(sfx, FIN, popper(), 0.8, -0.6, rev=0.2); add(sfx, FIN + 0.07, popper(), 0.8, 0.6, rev=0.2)
add(sfx, FIN + 2.0, popper(), 0.6, 0, rev=0.2)
add(sfx, FIN + 2.4, chime([93, 97, 100]), 0.3, 0, rev=0.6)

# ---------- reverb + music bus ----------
L = int(2.2 * SR); ti = np.arange(L) / SR
ir = np.stack([filt(rng.uniform(-1, 1, L), 'low', 6000) * np.exp(-ti / 0.55) for _ in range(2)], 1); ir[:int(0.015 * SR)] = 0
wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1) * 0.07
band = mus + sfx + wet
band = np.stack([filt(band[:, c], 'high', 30) for c in range(2)], 1)

# ---------- the family's singing (music fully out underneath) ----------
voice = np.zeros((N, 2))
for path, t0 in (('assets/clipA.wav', CLIP0), ('assets/clipB.wav', CLIP0 + CLIPA)):
    x = wav(path); x = np.stack([filt(x[:, c], 'high', 90) for c in range(2)], 1)
    f = int(0.02 * SR); x[:f] *= np.linspace(0, 1, f)[:, None]; x[-f:] *= np.linspace(1, 0, f)[:, None]
    i = int(t0 * SR); x = x[:N - i]; voice[i:i + len(x)] += x
ref = np.sqrt((band[int(4 * SR):int(24 * SR)] ** 2).mean())
vs = voice[int(CLIP0 * SR):int(FIN * SR)]
voice *= ref * 1.05 / np.sqrt((vs[np.abs(vs).max(1) > 1e-4] ** 2).mean())
voice = np.tanh(voice / (ref * 4)) * ref * 4  # tame claps/shrieks

mix = band + voice
fade = np.ones(N); fi = int(0.3 * SR); fo = int(1.6 * SR)
fade[:fi] = np.linspace(0, 1, fi); fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
mix *= fade[:, None]
mix = np.tanh(mix / np.percentile(np.abs(mix), 99.9) * 1.1)
mix /= np.max(np.abs(mix)) / 0.92
with wave.open('audio.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
seg = lambda a, b: 20 * np.log10(np.sqrt((mix[int(a * SR):int(b * SR)] ** 2).mean()))
print(f'ok  band {seg(4, 24):.1f} dB  clip {seg(28, 48):.1f} dB  finale {seg(52, 58):.1f} dB')
