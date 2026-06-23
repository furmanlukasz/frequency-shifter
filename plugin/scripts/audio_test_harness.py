#!/usr/bin/env python3
"""
Holy Shifter — objective offline audio test harness.

Loads the real VST3 via pedalboard, runs synthetic reference signals through
Spectral / peak-snap mode, and prints a pass/fail table of objective measurements.
This is the *regression net*; the manual Ableton suite (TEST_PLAN.md) is the
"does it sound musical" judgement layer.

Usage:
    run_audio_tests.sh                      # wrapper handles the venv
    python3 audio_test_harness.py [PLUGIN.vst3]

All measurements are latency-robust (steady-state windows / peak-finding), so
they don't depend on pedalboard compensating the plugin's FFT latency.
"""
import sys
import numpy as np
from pedalboard import load_plugin

SR = 48000
DEFAULT_PLUGIN = "/Users/benjaminvaughan/Library/Audio/Plug-Ins/VST3/Holy Shifter v119 PeakSnap.vst3"
PLUGIN_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PLUGIN

# ---------------------------------------------------------------- helpers
def process(plugin, x, **params):
    """Reset, apply params (baseline + overrides), process a mono signal."""
    plugin.reset()
    base = dict(mode="Spectral", dry_wet=100.0, shift_hz_hz=0.0,
                root_note="C", scale="Major")
    base.update(params)
    for k, v in base.items():
        setattr(plugin, k, v)
    y = np.asarray(plugin(x.astype(np.float32), SR)).astype(np.float64)
    return y.flatten()

def seg(sig, t0, t1):
    return sig[int(t0 * SR):int(t1 * SR)]

def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0

def c_major_freqs(fmin=40, fmax=8000):
    pcs = {0, 2, 4, 5, 7, 9, 11}
    return np.array([440.0 * 2 ** ((m - 69) / 12) for m in range(128)
                     if m % 12 in pcs and fmin <= 440.0 * 2 ** ((m - 69) / 12) <= fmax])

def dominant_freq(sig, fmin=50, fmax=4000):
    if len(sig) < 64:
        return 0.0
    w = sig * np.hanning(len(sig))
    sp = np.abs(np.fft.rfft(w))
    fr = np.fft.rfftfreq(len(sig), 1 / SR)
    sp = np.where((fr >= fmin) & (fr <= fmax), sp, 0.0)
    i = int(np.argmax(sp))
    if 1 <= i < len(sp) - 1:                      # parabolic interpolation
        a, b, c = (np.log(sp[i + d] + 1e-12) for d in (-1, 0, 1))
        den = a - 2 * b + c
        i = i + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
    return i * SR / len(sig)

def cents(f1, f2):
    return 1200.0 * np.log2(f1 / f2) if f1 > 0 and f2 > 0 else 1e9

def nearest_scale_cents(f, scale):
    return min(abs(cents(f, s)) for s in scale)

def spectral_flatness(sig):
    w = sig * np.hanning(len(sig))
    ps = np.abs(np.fft.rfft(w)) ** 2 + 1e-12
    return float(np.exp(np.mean(np.log(ps))) / np.mean(ps))

def acf_pitch(sig, fmin=80, fmax=1500):
    """True perceived pitch via autocorrelation — robust to the phase-vocoder's
    bin-quantized magnitude (the audible frequency comes from the phase advance,
    not the magnitude-peak bin, so a magnitude-FFT reading is misleading here)."""
    sig = sig - np.mean(sig)
    n = len(sig)
    if n < 256 or rms(sig) < 1e-6:
        return 0.0
    sp = np.fft.rfft(sig, 2 * n)
    corr = np.fft.irfft(sp * np.conj(sp))[:n]
    lo, hi = int(SR / fmax), min(int(SR / fmin), n - 1)
    if hi <= lo + 1:
        return 0.0
    lag = lo + int(np.argmax(corr[lo:hi]))
    if lo < lag < n - 1:                                  # parabolic interpolation
        a, b, c = corr[lag - 1], corr[lag], corr[lag + 1]
        den = a - 2 * b + c
        lag = lag + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0)
    return SR / lag if lag > 0 else 0.0

def frame_pitches(sig, frame=8192, hop=2048):
    out = [acf_pitch(sig[s:s + frame]) for s in range(0, len(sig) - frame, hop)]
    return np.array([p for p in out if p > 0])

def rms_envelope(sig, frame=2048, hop=512):
    return np.array([rms(sig[s:s + frame]) for s in range(0, len(sig) - frame, hop)])

# ---------------------------------------------------------------- signals
def tone(freq, secs, amp=0.3):
    t = np.arange(int(secs * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)

def harmonic(f0, secs, n=6, amp=0.25):
    t = np.arange(int(secs * SR)) / SR
    s = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, n + 1))
    return amp * s / np.max(np.abs(s))

def chord(freqs, secs, amp=0.25):
    t = np.arange(int(secs * SR)) / SR
    s = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return amp * s / np.max(np.abs(s))

def percussion(secs=2.0, hits=4, seed=0):
    n = int(secs * SR)
    x = np.zeros(n)
    rng = np.random.default_rng(seed)
    L = int(0.15 * SR)
    env = np.exp(-np.arange(L) / (0.03 * SR))          # 30 ms decay, sharp attack
    for i in range(hits):
        s = int((i + 0.3) / hits * n)
        x[s:s + L] += rng.standard_normal(L) * env
    return 0.3 * x / np.max(np.abs(x))

def tone_then_silence(freq, on=1.5, off=1.5):
    return np.concatenate([harmonic(freq, on), np.zeros(int(off * SR))])

# ---------------------------------------------------------------- tests
RESULTS = []
def record(name, detail, value, thresh_txt, ok):
    RESULTS.append((name, detail, value, thresh_txt, ok))

def run(plugin):
    scale = c_major_freqs()

    # T1 — snap accuracy: off-scale partial should land on a C-major note.
    # Pure sine (single partial) so we measure the snap, not harmonic redistribution.
    fin = 269.4  # ~quarter-tone above C4 (261.63), clearly in C4's basin
    y = process(plugin, tone(fin, 2.0), peak_snap=True, quantize=100.0, noise=30.0)
    fout = acf_pitch(seg(y, 0.7, 1.7))
    err = nearest_scale_cents(fout, scale)
    moved = abs(cents(fout, fin))
    record("T1 snap accuracy", f"in {fin:.1f}Hz -> out {fout:.1f}Hz", f"{err:.1f}c off-scale",
           "<40c & moved>20c", err < 40 and moved > 20)

    # T2 — stability/chatter: a boundary tone (max ambiguity) must lock to ONE note.
    # Pure sine isolates note-flicker; harmonics would confound the pitch tracker.
    fbnd = 277.2  # equidistant (100c) between C4 and D4 -> stresses note-flicker/hysteresis
    y = process(plugin, tone(fbnd, 2.5), peak_snap=True, quantize=100.0, noise=20.0, peak_sens=50.0)
    pitches = frame_pitches(seg(y, 0.7, 2.3))
    pstd = float(np.std([cents(p, np.median(pitches)) for p in pitches])) if len(pitches) > 3 else 1e9
    record("T2 stability (chatter)", f"pitch wobble over {len(pitches)} frames", f"{pstd:.1f}c std",
           "<12c", pstd < 12)

    # T3 — attack: percussion transients must stay sharp (crest factor).
    xp = percussion(2.0, hits=4)
    y = process(plugin, xp, peak_snap=True, quantize=100.0, noise=30.0)
    in_cr = float(np.max(np.abs(xp)) / (rms(xp) + 1e-12))
    out_cr = float(np.max(np.abs(y)) / (rms(y) + 1e-12))
    record("T3 attack (transients)", f"out crest {out_cr:.1f} (in {in_cr:.1f})", f"{out_cr:.1f}",
           ">3.5", out_cr > 3.5)

    # T4 — pumping: steady chord must hold a steady output level (deferred anti-pump check).
    y = process(plugin, chord([261.63, 329.63, 392.0], 2.5), peak_snap=True, noise=40.0, quantize=100.0)
    env = rms_envelope(seg(y, 0.7, 2.3))
    cv = float(np.std(env) / (np.mean(env) + 1e-12)) if len(env) else 1e9
    record("T4 level stability (pump)", "RMS coeff-of-variation", f"{cv:.3f}",
           "<0.15", cv < 0.15)

    # T5 — ringing: output must decay to silence after the input stops.
    y = process(plugin, tone_then_silence(329.63, 1.5, 1.5), peak_snap=True, quantize=100.0)
    active = rms(seg(y, 0.5, 1.3))
    tail = rms(seg(y, 2.6, 3.0))
    db = 20 * np.log10((tail + 1e-12) / (active + 1e-12))
    record("T5 ringing/decay", f"tail vs active", f"{db:.1f} dB",
           "<-55 dB", db < -55)

    # T6 — noise control: raising Noise must restore broadband (residual passthrough works).
    # (With peak-snap ON, snapping noise's random peaks to tones is by design; what must
    #  hold is that the Noise knob brings the broadband residual back.)
    rng = np.random.default_rng(1)
    xn = 0.2 * rng.standard_normal(int(2.0 * SR))
    f_lo = spectral_flatness(seg(process(plugin, xn, peak_snap=True, quantize=100.0, noise=0.0), 0.7, 1.7))
    f_hi = spectral_flatness(seg(process(plugin, xn, peak_snap=True, quantize=100.0, noise=100.0), 0.7, 1.7))
    ratio = f_hi / (f_lo + 1e-9)
    # Note: peak-snap is inherently tonal on pure broadband input (it snaps noise's random
    # peaks). This characterises that Noise at least does not REDUCE broadband; it is not a
    # bug if both are low. Treat a big drop as the regression signal.
    record("T6 noise vs broadband", f"flatness N0={f_lo:.3f} N100={f_hi:.3f}", f"x{ratio:.2f}",
           ">=0.9 (no worse)", ratio >= 0.9)

# ---------------------------------------------------------------- main
def main():
    print(f"Loading: {PLUGIN_PATH}")
    plugin = load_plugin(PLUGIN_PATH)
    print("Loaded. Running tests…\n")
    run(plugin)

    name_w = max(len(r[0]) for r in RESULTS)
    print(f"{'TEST'.ljust(name_w)}  {'MEASURED'.ljust(22)}  {'THRESHOLD'.ljust(16)}  RESULT")
    print("-" * (name_w + 52))
    npass = 0
    for name, detail, value, thr, ok in RESULTS:
        npass += ok
        print(f"{name.ljust(name_w)}  {value.ljust(22)}  {thr.ljust(16)}  {'PASS' if ok else 'FAIL'}")
        print(f"{' ' * name_w}  └ {detail}")
    print("-" * (name_w + 52))
    print(f"{npass}/{len(RESULTS)} passed")
    sys.exit(0 if npass == len(RESULTS) else 1)

if __name__ == "__main__":
    main()
