"""Soundtrack for the 15s showreel — 120 BPM, A minor, every hit synced to the picture."""
import numpy as np, wave
from scipy.signal import butter, sosfilt, fftconvolve

SR, DUR, BPM = 48000, 15.0, 120
BEAT = 60 / BPM
N = int(SR * DUR)
rng = np.random.default_rng(7)

music = np.zeros((N, 2)); sfx = np.zeros((N, 2)); verb_send = np.zeros((N, 2))

def mtof(m): return 440 * 2 ** ((m - 69) / 12)
def tt(d): return np.arange(int(SR * d)) / SR
def noise(d): return rng.uniform(-1, 1, int(SR * d))
def filt(x, kind, f, order=2):
    sos = butter(order, f, btype=kind, fs=SR, output='sos'); return sosfilt(sos, x)
def add(bus, t0, sig, g=1.0, pan=0.0, send=0.0):
    i = int(t0 * SR)
    if i >= N: return
    sig = sig[: N - i] * g
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    bus[i:i + len(sig), 0] += sig * l * 1.414; bus[i:i + len(sig), 1] += sig * r * 1.414
    if send:
        verb_send[i:i + len(sig), 0] += sig * l * send; verb_send[i:i + len(sig), 1] += sig * r * send
def expenv(d, decay, attack=0.002):
    t = tt(d); return np.minimum(1, t / attack) * np.exp(-t / decay)
def saw(f, t, detune=0):
    ph = (f * (1 + detune)) * t; return 2 * (ph - np.floor(ph + 0.5))
def sweep_filter(x, f0, f1, kind='low'):
    """time-varying filter via block processing"""
    out = np.zeros_like(x); B = 512
    for k in range(0, len(x), B):
        f = f0 * (f1 / f0) ** (k / max(1, len(x)))
        out[k:k + B] = filt(x[max(0, k - 2048):k + B], kind, min(f, SR / 2 - 100))[-len(x[k:k + B]):]
    return out

# ---------- instruments ----------
def kick(g=1.0):
    t = tt(0.5); f = 45 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t / 0.22) + 0.3 * filt(noise(0.5), 'high', 3000) * np.exp(-t / 0.006)
    return np.tanh(s * 1.6) * g
def clap():
    d = 0.35; s = np.zeros(int(SR * d)); n = filt(noise(d), 'band', [900, 2600])
    for o in [0, 0.011, 0.022]:
        i = int(o * SR); m = len(s) - i; s[i:] += n[:m] * np.exp(-np.arange(m) / SR / (0.01 if o < 0.02 else 0.12))
    return s
def hat(open_=False):
    d = 0.35 if open_ else 0.08
    return filt(noise(d), 'high', 7500) * expenv(d, 0.09 if open_ else 0.018)
def crash(d=2.2):
    return filt(noise(d), 'high', 4000) * expenv(d, 0.7, 0.001)
def boom(d=1.6, f0=70):
    t = tt(d); f = 28 + f0 * np.exp(-t / 0.15)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55)
    s += 0.6 * filt(noise(d), 'low', 700) * np.exp(-t / 0.25)
    return np.tanh(s * 1.8)
def whoosh(d, f0=300, f1=6000, peak=0.6):
    t = tt(d); env = np.sin(np.pi * np.clip(t / d, 0, 1)) ** 2
    env *= np.where(t / d < peak, 1, 1)
    return sweep_filter(noise(d), f0, f1, 'band' if False else 'low') * env
def riser(d, f0=200, f1=9000):
    t = tt(d); env = (t / d) ** 2
    n = sweep_filter(noise(d), f0, f1, 'low')
    tone = np.sin(2 * np.pi * np.cumsum(220 * (1 + 3 * (t / d) ** 2)) / SR) * 0.25
    return (n + tone) * env
def blip(f, d=0.08, drop=0.5):
    t = tt(d); fr = f * (1 - drop * t / d)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * expenv(d, d / 3, 0.001)
def tick():
    return filt(noise(0.025), 'band', [2500, 6000]) * expenv(0.025, 0.004, 0.0005)
def pssht(d=0.4):
    return filt(noise(d), 'band', [1800, 9000]) * (np.minimum(1, tt(d) / 0.01) * np.exp(-tt(d) / 0.12))
def bloop(f0=300, f1=900, d=0.15):
    t = tt(d); fr = f0 + (f1 - f0) * (t / d) ** 0.5
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.sin(np.pi * t / d)
def pluck(f, d=0.25):
    t = tt(d); s = saw(f, t) * 0.6 + np.sign(np.sin(2 * np.pi * f * t)) * 0.3
    return filt(s, 'low', 2600) * expenv(d, 0.07, 0.001)
def stab(midis, d=1.6):
    t = tt(d); s = sum(saw(mtof(m), t, dt) for m in midis for dt in (-0.006, 0, 0.007))
    return filt(s / (len(midis) * 3), 'low', 3500) * expenv(d, 0.45, 0.003)

# ---------- arrangement ----------
CHORDS = [  # (start, bass midi, pad midis)
    (0.0, 33, [57, 60, 64, 69]), (2.0, 33, [57, 60, 64, 69]), (4.0, 29, [53, 57, 60, 65]),
    (6.0, 36, [55, 60, 64, 67]), (8.0, 31, [55, 59, 62, 67]), (10.0, 33, [57, 60, 64, 69]),
    (12.0, 29, [53, 57, 60, 65]), (13.0, 31, [55, 59, 62, 67]), (14.0, 36, [55, 60, 64, 67, 72]),
]
def chord_at(t):
    c = CHORDS[0]
    for ch in CHORDS:
        if t >= ch[0]: c = ch
    return c

# pads (whole piece)
for i, (st, _, notes) in enumerate(CHORDS):
    en = CHORDS[i + 1][0] if i + 1 < len(CHORDS) else DUR
    d = en - st + 0.3; t = tt(d)
    s = sum(saw(mtof(m), t, dt) for m in notes for dt in (-0.004, 0.005))
    s = filt(s, 'low', 1400 if st < 14 else 2200) / (len(notes) * 2)
    env = np.minimum(1, t / (0.9 if st == 0 else 0.08)) * np.minimum(1, np.clip((d - t) / 0.3, 0, 1))
    if st >= 14: env *= np.exp(-t / 0.6)
    add(music, st, s * env, 0.22, 0, send=0.35)

GROOVE = [(2.0, 9.5), (10.0, 12.0)]
in_groove = lambda t: any(a <= t < b for a, b in GROOVE)
kicks = []
for b in range(int(DUR / BEAT)):
    t = b * BEAT
    if in_groove(t):
        add(music, t, kick(), 0.95); kicks.append(t)
        if b % 2 == 1: add(music, t, clap(), 0.45, 0, send=0.25)
        add(music, t + BEAT / 2, hat(True), 0.16, 0.3)
        for k in range(4):
            if k != 2: add(music, t + k * BEAT / 4, hat(), 0.12 if k % 2 else 0.07, -0.3)
        # bass 8ths
        _, bm, notes = chord_at(t)
        for k in range(2):
            d = 0.22; tb = tt(d)
            s = saw(mtof(bm), tb) + 0.5 * np.sin(2 * np.pi * mtof(bm - 12) * tb)
            s = filt(s, 'low', 520) * expenv(d, 0.12, 0.004)
            add(music, t + k * BEAT / 2 + (0.02 if k else 0), s, 0.5)
        # arp 16ths
        pat = [0, 1, 2, 3, 2, 1, 3, 2]
        for k in range(4):
            idx = pat[(b * 4 + k) % len(pat)]
            m = notes[idx % len(notes)] + 12
            add(music, t + k * BEAT / 4, pluck(mtof(m)), 0.13, 0.4 if k % 2 else -0.4, send=0.2)
# heartbeat kicks + arp in outro build
for t in [13.0, 13.5]: add(music, t, kick(0.7), 0.7); kicks.append(t)
for k in range(8):
    _, _, notes = chord_at(13.0)
    add(music, 13.0 + k * BEAT / 4, pluck(mtof(notes[k % 4] + 12)), 0.08 + k * 0.008, (-1) ** k * 0.5, send=0.4)
# snare roll 12.0 → 12.5 and 13.5 → 14
for (a, b) in [(12.0, 12.5), (13.5, 14.0)]:
    n = 12
    for k in range(n):
        add(music, a + (b - a) * (1 - (1 - k / n) ** 1.6), clap(), 0.12 + 0.3 * k / n, 0, send=0.2)

# sidechain pump on music
gain = np.ones(N); tt_all = np.arange(N) / SR
for k in kicks:
    m = tt_all >= k
    gain[m] *= 1 - 0.55 * np.exp(-(tt_all[m] - k) / 0.11)
music[:, 0] *= gain; music[:, 1] *= gain
# keep the kick itself unpumped: re-add kicks to sfx bus lightly for punch
for k in kicks: add(sfx, k, kick(), 0.25)

# ---------- SFX (synced to picture) ----------
add(sfx, 0.0, riser(0.5, 150, 5000), 0.5, 0, send=0.2)
add(sfx, 0.5, boom(1.8), 1.0, 0, send=0.3); add(sfx, 0.5, crash(2.0), 0.35, 0, send=0.4)
for i in range(6): add(sfx, 0.55 + i * 0.05 + 0.1, blip(900 + i * 140, 0.06), 0.25, -0.5 + i * 0.2, send=0.15)
add(sfx, 0.95, whoosh(0.5, 400, 7000), 0.45, 0.6)
add(sfx, 1.4, riser(0.6, 300, 12000), 0.65, 0, send=0.2)
add(sfx, 2.0, boom(1.2, 90), 0.8); add(sfx, 2.0, crash(), 0.55, 0, send=0.4)
for i, b in enumerate([2.0, 2.5, 3.0, 3.5]): add(sfx, b + 0.02, blip(600 + 200 * i, 0.12, 0.7), 0.3, 0, send=0.2)
add(sfx, 3.65, whoosh(0.35, 500, 9000), 0.5, -0.3)
add(sfx, 4.0, boom(1.0, 60), 0.6, 0, send=0.2)
for i in range(8): add(sfx, 4.05 + i * 0.045, whoosh(0.12, 1500, 9000), 0.12, -0.8 + i * 0.22)
for i in range(26): add(sfx, 4.3 + i * 0.5 / 26, tick(), 0.35, -0.4)
for b in [4.85, 5.0]: add(sfx, b, blip(2400, 0.05, 0), 0.18, 0.4)
add(sfx, 5.15, blip(3200, 0.12, 0), 0.18, 0.4, send=0.2)
for b in [5.0, 5.5, 6.0, 6.5]:
    add(sfx, b, filt(noise(0.25), 'band', [3000, 9000]) * expenv(0.25, 0.06, 0.01), 0.3, 0.3)
for i in range(16): add(sfx, 5.5 + i * 0.05, blip(1500 + rng.uniform(0, 2500), 0.03, 0), 0.07, -0.6)
add(sfx, 6.5, whoosh(0.55, 150, 4000), 0.7, 0.5)
for i in range(10): add(sfx, 6.55 + rng.uniform(0, 0.42), bloop(rng.uniform(250, 500), rng.uniform(700, 1400), 0.06), 0.18, rng.uniform(-.8, .8))
add(sfx, 7.0, filt(noise(1.2), 'high', 1200) * expenv(1.2, 0.25, 0.002), 0.55, 0, send=0.4)
add(sfx, 7.0, boom(1.0, 80), 0.6)
add(sfx, 7.02, bloop(180, 700, 0.18), 0.4)
t = tt(0.4); add(sfx, 7.25, np.sin(2 * np.pi * np.cumsum(300 + 500 * t / 0.4 + 40 * np.sin(2 * np.pi * 18 * t)) / SR) * expenv(0.4, 0.15), 0.3, 0.4, send=0.3)
for i in range(7): add(sfx, 7.4 + i * 0.04, bloop(500 + 60 * i, 1200 + 80 * i, 0.07), 0.2, 0.5)
for st, pan in [(7.55, 0.4), (8.05, -0.6), (8.55, 0.4), (9.05, -0.6), (9.3, 0.4)]:
    add(sfx, st, pssht(), 0.55, pan, send=0.25); add(sfx, st, bloop(400, 150, 0.08), 0.25, pan)
add(sfx, 7.8, bloop(600, 1500, 0.09), 0.3, -0.7)
for i in range(26): add(sfx, 7.7 + i * 0.5 / 26, tick(), 0.3, 0.4)
# glitch + CRT power-down
g = np.zeros(int(0.22 * SR))
for i in range(12):
    s0 = int(rng.uniform(0, 0.2) * SR); d = int(rng.uniform(0.008, 0.03) * SR)
    f = rng.uniform(200, 3000); tg = np.arange(d) / SR
    g[s0:s0 + d] += np.sign(np.sin(2 * np.pi * f * tg))[: len(g[s0:s0 + d])]
add(sfx, 9.5, np.round(g * 4) / 4, 0.18, 0)
add(sfx, 9.5, filt(noise(0.22), 'band', [400, 5000]) * 0.5, 0.15, 0)
t = tt(0.3); add(sfx, 9.72, np.sin(2 * np.pi * np.cumsum(1400 * np.exp(-t / 0.07) + 40) / SR) * np.exp(-t / 0.12), 0.35, 0, send=0.2)
# montage cuts
for c in [10.0, 10.5, 11.0, 11.5, 12.0]:
    add(sfx, c, boom(0.7, 100), 0.55); add(sfx, c, filt(noise(0.3), 'high', 2500) * expenv(0.3, 0.06), 0.4, 0, send=0.2)
t = tt(0.5); add(sfx, 11.0, sum(np.sin(2 * np.pi * f * t) for f in [1760, 2217, 2637, 3520]) / 4 * expenv(0.5, 0.2), 0.2, 0, send=0.6)
add(sfx, 11.5, np.round(filt(noise(0.4), 'band', [300, 4000]) * 3) / 3 * expenv(0.4, 0.12), 0.25, 0)
add(sfx, 12.0, riser(0.5, 300, 14000), 0.65, 0, send=0.2)
add(sfx, 12.5, boom(2.0, 70), 1.0, 0, send=0.3); add(sfx, 12.5, crash(2.5), 0.5, 0, send=0.5)
add(sfx, 12.5, stab([45, 57, 60, 64, 69]), 0.5, 0, send=0.6)
for i in range(6): add(sfx, 12.75 + i * 0.045, whoosh(0.1, 2000, 9000), 0.1, -0.5 + i * 0.2)
for i in range(32): add(sfx, 13.2 + i * 0.55 / 32, tick(), 0.25, 0.2)
add(sfx, 13.4, riser(0.6, 400, 14000), 0.55, 0, send=0.2)
add(sfx, 14.0, kick(), 1.0); add(sfx, 14.0, boom(1.0, 60), 0.8, 0, send=0.3)
add(sfx, 14.0, crash(1.0), 0.6, 0, send=0.6)
add(sfx, 14.0, stab([48, 60, 64, 67, 72, 76], 1.0), 0.7, 0, send=0.8)
t = tt(0.45); rev = filt(noise(0.45), 'low', 3000) * (t / 0.45) ** 3
add(sfx, 14.35, rev, 0.35, 0)
add(sfx, 14.78, blip(2600, 0.06, 0), 0.25, 0, send=0.6)

# ---------- reverb + master ----------
irlen = int(2.4 * SR); ti = np.arange(irlen) / SR
ir = np.stack([filt(rng.uniform(-1, 1, irlen), 'low', 6000) * np.exp(-ti / 0.55) for _ in range(2)], 1)
ir[: int(0.012 * SR)] = 0
wet = np.stack([fftconvolve(verb_send[:, c], ir[:, c])[:N] for c in range(2)], 1) * 0.12
mix = music * 0.85 + sfx + wet
mix = np.stack([filt(mix[:, c], 'high', 28) for c in range(2)], 1)
# tail fade (last 0.12s)
fade = np.ones(N); fl = int(0.12 * SR); fade[-fl:] = np.linspace(1, 0, fl)
mix *= fade[:, None]
mix = mix / np.percentile(np.abs(mix), 99.95)  # gentle limiting: only the top transients saturate
mix = np.tanh(mix * 1.1)
mix /= np.max(np.abs(mix)) / 0.93
with wave.open('audio.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype('<i2').tobytes())
print('audio.wav written', mix.shape)
