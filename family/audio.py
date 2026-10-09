"""Warm acoustic-pop score for the 44s family film — 90 BPM, G major, I–V–vi–IV.
Every bar boundary is a picture cut; SFX (shutter, page turn, chimes) sit on the edit."""
import numpy as np, wave
from scipy.signal import butter, sosfilt, fftconvolve

SR, DUR, BPM = 48000, 44.0, 90
BEAT = 60 / BPM; BAR = 4 * BEAT
N = int(SR * DUR)
rng = np.random.default_rng(11)
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

# ---------- instruments ----------
def piano(m, d=1.8, vel=1.0):
    f, t = mtof(m), tt(d); s = np.zeros_like(t)
    for k, a in enumerate([1, .5, .28, .16, .09, .05], 1):
        fk = f * k * (1 + 0.0004 * k * k)
        s += a * np.sin(2 * np.pi * fk * t) * np.exp(-t * (1.6 + k * 0.9))
    s += 0.05 * filt(noise(d), 'band', [1500, 5000]) * np.exp(-t / 0.01)
    return filt(s, 'low', 4500) * np.minimum(1, t / 0.003) * vel * 0.5
def pluck(m, d=1.4, bright=0.5):
    """Karplus–Strong string, block-vectorised"""
    f = mtof(m); n = max(2, int(SR / f)); L = int(SR * d)
    y = np.zeros(L + n + 1); y[:n] = filt(rng.uniform(-1, 1, n), 'low', 2000 + bright * 6000)
    dec = 0.996
    for s in range(n + 1, L + n + 1, n):
        e = min(s + n, L + n + 1)
        y[s:e] = dec * 0.5 * (y[s - n:e - n] + y[s - n - 1:e - n - 1])
    out = y[n:n + L]; return out * np.exp(-tt(d) / 0.7) * 0.6
def glock(m, d=1.2):
    f, t = mtof(m), tt(d)
    s = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.5) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.15) \
        + 0.15 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.06)
    return s * np.minimum(1, t / 0.001) * 0.35
def pad(ms, d):
    t = tt(d); s = np.zeros_like(t)
    for m in ms:
        for dt in (-0.005, 0.004):
            ph = mtof(m) * (1 + dt) * t; s += 2 * (ph - np.floor(ph + .5))
    s = filt(s / (len(ms) * 2), 'low', 900)
    a = np.minimum(1, t / 0.6) * np.minimum(1, np.clip((d - t) / 0.5, 0, 1))
    return s * a
def kick():
    t = tt(0.35); f = 48 + 60 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16) * 0.9
def snap():
    return filt(noise(0.18), 'band', [1200, 3500]) * env(0.18, 0.035, 0.001)
def shaker():
    d = 0.09; t = tt(d); return filt(noise(d), 'high', 6000) * np.sin(np.pi * t / d) ** 2 * 0.5
def bass(m, d):
    t = tt(d); f = mtof(m)
    s = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t)
    return s * np.minimum(1, t / 0.01) * np.exp(-t / 0.9) * 0.55
def bird(f0=3200):
    d = 0.12; t = tt(d); f = f0 * (1 + 0.35 * np.sin(np.pi * t / d)) + 300 * np.sin(2 * np.pi * 38 * t)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / d) ** 2 * 0.12
def shutter():
    s = np.zeros(int(SR * 0.12))
    for o, g in [(0, 1), (0.055, 0.7)]:
        i = int(o * SR); c = filt(noise(0.03), 'band', [1800, 7000]) * env(0.03, 0.006, 0.0005)
        s[i:i + len(c)] += c * g
    return s
def page():
    d = 0.6; t = tt(d); e = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 3
    return filt(noise(d), 'band', [600, 5000]) * e * 0.5
def chime(ms, gap=0.07):
    out = np.zeros(int(SR * 2.5))
    for k, m in enumerate(ms):
        g = glock(m, 2.0); i = int(k * gap * SR); out[i:i + len(g)] += g
    return out

# ---------- harmony ----------
CH = [([55, 59, 62, 67], 43), ([54, 57, 62, 66], 38), ([55, 59, 64, 67], 40), ([55, 60, 64, 67], 36)]
MEL = [[(0, 83), (1, 86), (2, 83), (3, 81)], [(0, 81), (1, 78), (2, 81), (3, 86)],
       [(0, 79), (1, 83), (2, 88), (3, 86)], [(0, 76), (1, 79), (2, 84), (3, 83)]]
NB = 16
for n in range(NB):
    t0 = n * BAR; voic, root = CH[n % 4]
    full = 4 <= n <= 13
    # pad all the way
    add(mus, t0, pad(voic, BAR + 0.6), 0.16, 0, rev=0.3)
    # piano arpeggio 8ths (sparser in the groove)
    pat = [0, 1, 2, 3, 2, 1, 2, 3]
    for k in range(8):
        if full and k % 2: continue
        m = voic[pat[k]] + 12
        add(mus, t0 + k * BEAT / 2, piano(m, 1.6, 0.75 if k else 1.0), 0.32, (-0.3 if k % 2 else 0.3), rev=0.35)
    # glockenspiel melody
    if n in (2, 3) or 9 <= n <= 13:
        for b, m in MEL[n % 4]:
            add(mus, t0 + b * BEAT, glock(m), 0.42, 0.25, rev=0.4)
    if full:
        # ukulele-style strum: D . D U . U D U
        for b, up in [(0, 0), (1, 0), (1.5, 1), (2.5, 1), (3, 0), (3.5, 1)]:
            strings = voic[::-1] if up else voic
            for j, m in enumerate(strings):
                add(mus, t0 + b * BEAT + j * 0.012, pluck(m + 12, 1.0, 0.6), 0.2 * (0.7 if up else 1), 0.35, rev=0.15)
        for b in (0, 2): add(mus, t0 + b * BEAT, kick(), 0.75)
        for b in (1, 3): add(mus, t0 + b * BEAT, snap(), 0.35, -0.1, rev=0.2)
        for k in range(8): add(mus, t0 + k * BEAT / 2, shaker(), 0.18 if k % 2 else 0.1, -0.4)
        add(mus, t0, bass(root, BEAT * 1.6), 0.7); add(mus, t0 + 2.5 * BEAT, bass(root, BEAT * 1.4), 0.55)
# final chord (bar 16 = 42.67s) + ring-out
tF = NB * BAR - BAR  # end card starts at bar 15; resolve on bar 15's downbeat too
for j, m in enumerate([43, 55, 59, 62, 67, 71, 74]):
    add(mus, NB * BAR - 2.0 + j * 0.06, piano(m, 3.5, 0.8), 0.3, -0.4 + j * 0.13, rev=0.5)

# ---------- ambience + SFX on the edit ----------
for k in range(14):
    tb = rng.uniform(0.3, 9.5); add(sfx, tb, bird(rng.uniform(2600, 4200)), 0.6, rng.uniform(-.8, .8), rev=0.3)
add(sfx, 1.0, chime([79, 83, 86, 91]), 0.35, 0, rev=0.6)                 # title
for c in (4 * BAR, 9 * BAR): add(sfx, c - 0.35, page(), 0.5, 0.2, rev=0.2)  # chapter cards
for k in range(4): add(sfx, 7 * BAR + 0.15 + k * BAR / 4, shutter(), 0.7, -0.3 + k * 0.2, rev=0.1)  # polaroids
add(sfx, 2 * BAR, shutter(), 0.35, 0.4)
add(sfx, 12 * BAR + 2.2, glock(91, 1.0), 0.4, 0.3, rev=0.5)               # heart pop
add(sfx, 15 * BAR + 2.2, chime([86, 91, 95]), 0.3, 0, rev=0.7)            # end heart

# ---------- reverb + master ----------
L = int(2.8 * SR); ti = np.arange(L) / SR
ir = np.stack([filt(rng.uniform(-1, 1, L), 'low', 5000) * np.exp(-ti / 0.7) for _ in range(2)], 1); ir[:int(0.02 * SR)] = 0
wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1) * 0.08
mix = mus + sfx + wet
mix = np.stack([filt(mix[:, c], 'high', 30) for c in range(2)], 1)
fade = np.ones(N); fi = int(0.4 * SR); fo = int(2.2 * SR)
fade[:fi] = np.linspace(0, 1, fi); fade[-fo:] = np.linspace(1, 0, fo) ** 1.5
mix *= fade[:, None]
mix = np.tanh(mix / np.percentile(np.abs(mix), 99.95) * 1.0)
mix /= np.max(np.abs(mix)) / 0.9
with wave.open('audio.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print('ok', 20 * np.log10(np.sqrt((mix ** 2).mean())))
