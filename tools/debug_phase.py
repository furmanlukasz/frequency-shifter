"""Debug the phase computation for allpass chains."""
import numpy as np


def allpass_phase_direct(a, omega):
    """Compute phase of H(z) = (a + z^-1)/(1 + a*z^-1) by direct complex evaluation."""
    z = np.exp(1j * omega)
    H = (a + z**(-1)) / (1.0 + a * z**(-1))
    return np.angle(H)


def allpass_phase_formula(a, omega):
    """Compute phase using the formula from verification function."""
    return -omega + 2.0 * np.arctan2(a * np.sin(omega), 1.0 + a * np.cos(omega))


def chain_phase_direct(coeffs, omega):
    """Total phase of a chain of allpass sections, computed directly."""
    z = np.exp(1j * omega)
    H_total = np.ones_like(z)
    for a in coeffs:
        H = (a + z**(-1)) / (1.0 + a * z**(-1))
        H_total *= H
    return np.angle(H_total)


def chain_phase_unwrapped(coeffs, omega):
    """Total phase of chain, unwrapped to avoid 2π jumps."""
    z = np.exp(1j * omega)
    H_total = np.ones_like(z)
    for a in coeffs:
        H = (a + z**(-1)) / (1.0 + a * z**(-1))
        H_total *= H
    return np.unwrap(np.angle(H_total))


# Known-good positive coefficients
coeffsI_pos = [0.4021921162426, 0.8561710882420, 0.9722909545651,
               0.9952884791278, 0.9990657381831, 0.9998766533010]
coeffsQ_pos = [0.1684919243525, 0.7024051466406, 0.9351665954634,
               0.9862259517082, 0.9979710606470, 0.9997089053332]

# Nathan Ho 6-stage coefficients (negative)
coeffsI_neg = [-0.99897188, -0.99221914, -0.96870467, -0.88010622, -0.59245316, 0.06438377]
coeffsQ_neg = [-0.99643764, -0.98426124, -0.93829943, -0.77352469, -0.3160085, 0.59569912]

# Nathan Ho 8-stage coefficients at 44100 Hz
coeffsI_8 = [-0.99923507, -0.99513724, -0.98556779, -0.95958694, -0.88952789,
             -0.71574562, -0.35611623, 0.23895745]
coeffsQ_8 = [-0.99750929, -0.9914857, -0.97579142, -0.93288954, -0.82078137,
             -0.56338361, -0.09119001, 0.68281418]

sample_rate = 44100.0
freqs = [20, 100, 500, 1000, 2000, 5000, 10000, 15000, 20000]
omega_vals = [2.0 * np.pi * f / sample_rate for f in freqs]

print("=== Step 1: Verify formula matches direct computation for single allpass ===")
print("Testing with a = 0.4, a = -0.4, a = -0.99")
for a_test in [0.4, -0.4, -0.99, 0.06]:
    print(f"\n  a = {a_test}")
    for f, w in zip([1000, 10000, 20000], [2*np.pi*1000/44100, 2*np.pi*10000/44100, 2*np.pi*20000/44100]):
        direct = allpass_phase_direct(a_test, w)
        formula = allpass_phase_formula(a_test, w)
        print(f"    f={f:5d}: direct={np.degrees(direct):8.3f}°  formula={np.degrees(formula):8.3f}°  match={np.abs(direct-formula) < 1e-10}")

print("\n=== Step 2: Chain phase — wrapped vs unwrapped ===")
print("\nKnown-good POSITIVE coefficients:")
for f, w in zip(freqs, omega_vals):
    pI_wrapped = chain_phase_direct(coeffsI_pos, w)
    pQ_wrapped = chain_phase_direct(coeffsQ_pos, w)
    diff_wrapped = np.degrees(pI_wrapped - pQ_wrapped)
    print(f"  f={f:5d}: phaseI={np.degrees(pI_wrapped):8.2f}°  phaseQ={np.degrees(pQ_wrapped):8.2f}°  diff={diff_wrapped:8.2f}°")

# Now do unwrapped
print("\nKnown-good POSITIVE coefficients (UNWRAPPED phases):")
omega_dense = np.linspace(0.001, np.pi * 0.99, 10000)
pI_uw = chain_phase_unwrapped(coeffsI_pos, omega_dense)
pQ_uw = chain_phase_unwrapped(coeffsQ_pos, omega_dense)
diff_uw = pI_uw - pQ_uw

for f in freqs:
    w = 2.0 * np.pi * f / sample_rate
    idx = np.argmin(np.abs(omega_dense - w))
    print(f"  f={f:5d}: phaseI={np.degrees(pI_uw[idx]):8.2f}°  phaseQ={np.degrees(pQ_uw[idx]):8.2f}°  diff={np.degrees(diff_uw[idx]):8.2f}°")

print("\nNathan Ho 6-stage NEGATIVE coefficients (UNWRAPPED phases):")
pI_uw = chain_phase_unwrapped(coeffsI_neg, omega_dense)
pQ_uw = chain_phase_unwrapped(coeffsQ_neg, omega_dense)
diff_uw = pI_uw - pQ_uw

for f in freqs:
    w = 2.0 * np.pi * f / sample_rate
    idx = np.argmin(np.abs(omega_dense - w))
    print(f"  f={f:5d}: phaseI={np.degrees(pI_uw[idx]):8.2f}°  phaseQ={np.degrees(pQ_uw[idx]):8.2f}°  diff={np.degrees(diff_uw[idx]):8.2f}°")

print("\nNathan Ho 8-stage NEGATIVE coefficients at 44100 Hz (UNWRAPPED phases):")
pI_uw = chain_phase_unwrapped(coeffsI_8, omega_dense)
pQ_uw = chain_phase_unwrapped(coeffsQ_8, omega_dense)
diff_uw = pI_uw - pQ_uw

for f in freqs:
    w = 2.0 * np.pi * f / sample_rate
    idx = np.argmin(np.abs(omega_dense - w))
    print(f"  f={f:5d}: phaseI={np.degrees(pI_uw[idx]):8.2f}°  phaseQ={np.degrees(pQ_uw[idx]):8.2f}°  diff={np.degrees(diff_uw[idx]):8.2f}°")

# Compute sideband rejection for each set
print("\n=== Step 3: Sideband rejection ===")
omega_audio = np.geomspace(2*np.pi*20/sample_rate, 2*np.pi*20000/sample_rate, 10000)

for name, cI, cQ in [("Positive 6-stage", coeffsI_pos, coeffsQ_pos),
                       ("Nathan Ho 6-stage", coeffsI_neg, coeffsQ_neg),
                       ("Nathan Ho 8-stage", coeffsI_8, coeffsQ_8)]:
    pI = chain_phase_unwrapped(cI, omega_audio)
    pQ = chain_phase_unwrapped(cQ, omega_audio)
    diff = pI - pQ

    # The phase difference should be approximately constant (either +90 or -90)
    # Find the median to determine the target
    median_diff = np.median(diff)
    target = np.round(median_diff / (np.pi/2)) * (np.pi/2)

    error = diff - target
    max_err = np.max(np.abs(error))
    max_err_deg = np.degrees(max_err)

    if 0 < max_err < np.pi/2:
        rejection = -20.0 * np.log10(np.abs(np.sin(max_err)))
    else:
        rejection = 0.0

    print(f"\n  {name}:")
    print(f"    Median phase diff: {np.degrees(median_diff):.2f}°")
    print(f"    Target: {np.degrees(target):.2f}°")
    print(f"    Max error: {max_err_deg:.4f}°")
    print(f"    Sideband rejection: ~{rejection:.1f} dB")
