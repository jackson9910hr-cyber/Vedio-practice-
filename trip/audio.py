"""Upbeat summer-pop score for the 32s trip reel — 120 BPM, C major, I–V–vi–IV.
Bar 0–1 build + countdown, drop on bar 2, four-on-the-floor groove with sidechained
bass/stabs, 8th-note hits for the montage, tag ending. SFX (whoosh/impact) sit on the cuts."""
import numpy as np, wave
from scipy.signal import butter, sosfilt, fftconvolve

SR, DUR, BPM = 48000, 32.0, 120
BEAT = 60 / BPM; BAR = 4 * BEAT
N = int(SR * DUR)
rng = np.random.default_rng(7)
drums = np.zeros((N, 2)); music = np.zeros((N, 2)); sfx = np.zeros((N, 2)); send = np.zeros((N, 2))

def mtof(m): return 440 * 2 ** ((m - 69) / 12)
def tt(d): return np.arange(int(SR * d)) / SR
def noise(d): return rng.uniform(-1, 1, int(SR * d))
def filt(x, kind, f, order=2): return sosfilt(butter(order, f, btype=kind, fs=SR, output='sos'), x)
def env(d, dec, att=0.002): t = tt(d); return np.minimum(1, t / att) * np.exp(-t / dec)
def add(bus, t0, sig, g=1.0, pan=0.0, rev=0.0):
    i = int(round(t0 * SR))
    if i >= N or i < 0: return
    sig = sig[: N - i] * g
    l, r = np.cos((pan + 1) * np.pi / 4) * 1.414, np.sin((pan + 1) * np.pi / 4) * 1.414
    bus[i:i + len(sig), 0] += sig * l; bus[i:i + len(sig), 1] += sig * r
    if rev: send[i:i + len(sig), 0] += sig * l * rev; send[i:i + len(sig), 1] += sig * r * rev
def saw(f, t, det=(-0.006, 0, 0.006)):
    s = 0
    for d in det: ph = f * (1 + d) * t + rng.uniform(); s = s + 2 * (ph - np.floor(ph + .5))
    return s / len(det)

# ---------- drums ----------
def kick():
    t = tt(0.4); f = 50 + 110 * np.exp(-t / 0.025)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.2) + 0.3 * filt(noise(0.4), 'high', 3000) * np.exp(-t / 0.004)) * 0.95
def clap():
    s = np.zeros(int(SR * 0.3))
    for k, o in enumerate([0, 0.011, 0.022]):
        i = int(o * SR); c = filt(noise(0.3 - o), 'band', [900, 4500]) * env(0.3 - o, 0.012 if k < 2 else 0.09, 0.0005)
        s[i:i + len(c)] += c
    return s * 0.8
def hat(open_=False):
    d = 0.25 if open_ else 0.05
    return filt(noise(d), 'high', 7500) * env(d, 0.08 if open_ else 0.012, 0.0005) * 0.6
def snare():
    t = tt(0.2)
    return filt(noise(0.2), 'band', [1500, 7000]) * env(0.2, 0.05) * 0.7 + np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04) * 0.4
def crash():
    return filt(noise(2.0), 'high', 5000) * env(2.0, 0.6, 0.001) * 0.45

# ---------- tonal ----------
def stab(ms, d=0.22):
    t = tt(d); s = sum(saw(mtof(m), t) for m in ms) / len(ms)
    return filt(s, 'low', 3800) * env(d, 0.12, 0.003) * 0.9
def pluck(m, d=0.35):
    t = tt(d); f = mtof(m)
    s = np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + np.sin(2 * np.pi * f * 2 * t) * 0.3
    return filt(s, 'low', 5000) * env(d, 0.11, 0.002) * 0.55
def bell(m, d=0.8):
    f, t = mtof(m), tt(d)
    s = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.35) + 0.3 * np.sin(2 * np.pi * f * 3.0 * t) * np.exp(-t / 0.1)
    return s * np.minimum(1, t / 0.001) * 0.4
def bass(m, d):
    t = tt(d); f = mtof(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * filt(saw(f, t, (0,)), 'low', 700)
    return s * np.minimum(1, t / 0.004) * np.minimum(1, np.clip((d - t) / 0.02, 0, 1)) * 0.7
def pad(ms, d):
    t = tt(d); s = sum(saw(mtof(m), t) for m in ms) / len(ms)
    return filt(s, 'low', 1400) * np.minimum(1, t / 0.3) * np.minimum(1, np.clip((d - t) / 0.4, 0, 1)) * 0.5
def whoosh(d=0.35, up=True):
    t = tt(d); x = noise(d); out = np.zeros_like(x); seg = 256
    for i in range(0, len(x), seg):                     # swept band-pass
        q = i / len(x); fc = 400 + 5000 * (q if up else 1 - q)
        out[i:i + seg] = filt(x[i:i + seg + 512], 'band', [fc * 0.6, min(fc * 1.6, 20000)])[:len(out[i:i + seg])]
    return out * np.sin(np.pi * t / d) ** 2 * 0.8
def riser(d):
    t = tt(d); f = 200 * (12 ** (t / d))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    return (filt(noise(d), 'high', 2000) * 0.5 + tone) * (t / d) ** 2
def impact():
    t = tt(1.2); f = 40 + 60 * np.exp(-t / 0.05)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.4) * 0.9 + filt(noise(1.2), 'low', 3000) * np.exp(-t / 0.15) * 0.3

# ---------- harmony: C G Am F ----------
CH = [([60, 64, 67, 72], 36), ([59, 62, 67, 71], 43), ([60, 64, 69, 72], 45), ([60, 65, 69, 72], 41)]
# lead hook (beat, midi, len-in-beats) per chord bar
HOOK = [[(0, 79, .5), (.5, 76, .5), (1, 79, .5), (1.5, 81, .5), (2, 79, 1), (3, 76, .5), (3.5, 74, .5)],
        [(0, 74, .5), (.5, 71, .5), (1, 74, .5), (1.5, 76, .5), (2, 74, 1), (3, 79, .5), (3.5, 81, .5)],
        [(0, 81, .5), (.5, 84, .5), (1, 81, .5), (1.5, 79, .5), (2, 76, 1), (3, 79, .5), (3.5, 81, .5)],
        [(0, 77, .5), (.5, 81, .5), (1, 84, .5), (1.5, 81, .5), (2, 79, 1.5), (3.5, 76, .5)]]
duck = np.ones(N)                                         # sidechain envelope from the kick
NB = 16
for n in range(NB):
    t0 = n * BAR; voic, root = CH[n % 4]
    groove = 2 <= n <= 13
    if n < 2:
        # intro: muted plucked chords, then a snare roll + riser into the drop
        for k in range(8 if n else 4):
            add(music, t0 + k * BEAT / (2 if n else 1), stab(voic, 0.15), 0.25 + 0.1 * n, 0.2 * (-1) ** k, rev=0.3)
        if n == 1:
            for k in range(16):
                add(drums, t0 + k * BEAT / 4, snare(), 0.15 + 0.5 * k / 16, 0.1)
            for k, m in enumerate([84, 79, 72]):          # countdown beeps 3·2·1
                add(sfx, t0 + k * BEAT, bell(m + 12, 0.3), 0.5, 0, rev=0.3)
        continue
    if groove:
        for b in range(4):
            tb = t0 + b * BEAT
            add(drums, tb, kick(), 1.0)
            i = int(tb * SR); L = int(0.28 * SR)
            duck[i:i + L] = np.minimum(duck[i:i + L], 1 - 0.65 * np.exp(-np.arange(min(L, N - i)) / SR / 0.07))
            add(drums, tb + BEAT / 2, hat(True), 0.5, 0.3)
            for s16 in (0, 1, 3): add(drums, tb + s16 * BEAT / 4, hat(), 0.35 if s16 else 0.25, -0.35)
        for b in (1, 3): add(drums, t0 + b * BEAT, clap(), 0.8, 0, rev=0.25)
        # bass: offbeat-pumping octaves
        for k in range(8):
            add(music, t0 + k * BEAT / 2, bass(root + (12 if k % 2 else 0), BEAT / 2 * 0.9), 0.55)
        # house stabs on the offbeats
        for k in (1, 3, 5, 6):
            add(music, t0 + k * BEAT / 2, stab(voic), 0.32, 0.25 * (-1) ** k, rev=0.25)
        add(music, t0, pad(voic, BAR), 0.18, 0, rev=0.4)
        if n in (2, 3, 4, 5, 8, 9, 12, 13):
            for b, m, l in HOOK[n % 4]:
                add(music, t0 + b * BEAT, pluck(m, l * BEAT + 0.15), 0.5, 0.15, rev=0.35)
                add(music, t0 + b * BEAT + 0.75 * BEAT, pluck(m, 0.2), 0.12, -0.5, rev=0.4)   # dotted-8th echo
        if n == 11:  # montage: 8th-note accents + fill into the finale
            for k in range(8): add(sfx, t0 + k * BEAT / 2, bell(84 + [0, 4, 7, 12][k % 4], 0.3), 0.35, 0.5 * (-1) ** k, rev=0.3)
            for k in range(8): add(drums, t0 + 2 * BEAT + k * BEAT / 4, snare(), 0.3 + 0.4 * k / 8, -0.1)
    if n in (2, 7, 12): add(drums, t0, crash(), 0.6, 0.2)
# tag: final big chord on bar 14, light bells, ring out
tE = 14 * BAR
add(drums, tE, kick(), 1.0); add(drums, tE, crash(), 0.7)
add(music, tE, pad([48, 60, 64, 67, 72, 76], 3.6), 0.45, 0, rev=0.6)
add(music, tE, bass(36, 2.5), 0.6)
for k, m in enumerate([84, 88, 91, 96, 91, 88, 91, 96]):
    add(music, tE + 0.5 + k * BEAT / 2, bell(m, 0.9), 0.3, 0.4 * (-1) ** k, rev=0.5)
add(drums, tE + 3.5, kick(), 0.8); add(music, tE + 3.5, stab([60, 64, 67, 72], 0.5), 0.5, 0, rev=0.6)

# ---------- SFX on the cuts ----------
add(sfx, 2.0, riser(2.0), 0.5)
add(sfx, 4.0, impact(), 0.8, 0, rev=0.3)
for c in (8, 12, 20): add(sfx, c - 0.18, whoosh(0.36), 0.55, 0.3)
for c in (10, 18, 28): add(sfx, c - 0.3, whoosh(0.5, up=False), 0.45, -0.3)
for c in (14, 22, 24): add(sfx, c, impact(), 0.45, 0, rev=0.2)

# ---------- reverb + master ----------
L = int(1.8 * SR); ti = np.arange(L) / SR
ir = np.stack([filt(rng.uniform(-1, 1, L), 'low', 6000) * np.exp(-ti / 0.4) for _ in range(2)], 1); ir[:int(0.015 * SR)] = 0
wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1) * 0.06
mix = drums + music * duck[:, None] + sfx + wet * duck[:, None]
mix = np.stack([filt(mix[:, c], 'high', 28) for c in range(2)], 1)
fade = np.ones(N); fi = int(0.05 * SR); fo = int(1.2 * SR)
fade[:fi] = np.linspace(0, 1, fi); fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
mix *= fade[:, None]
mix = np.tanh(mix / np.percentile(np.abs(mix), 99.9) * 1.2)
mix /= np.max(np.abs(mix)) / 0.92
with wave.open('audio.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print('ok', 20 * np.log10(np.sqrt((mix ** 2).mean())))
