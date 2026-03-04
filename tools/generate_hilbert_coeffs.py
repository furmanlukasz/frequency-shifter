"""
Generate Hilbert transform allpass filter coefficients using the
Weaver/Darlington phase-difference network design method.

Implementation based on Nathan Ho's code (snappizz/filter-design), which
implements the classical Weaver design for wideband 90° phase-difference networks.

CRITICAL: Uses frequency prewarping so the digital passband matches the
specified [f_low, f_high] range after bilinear transform.

This is ANALYTICAL (deterministic, instant) — no iterative optimization.

References:
- Nathan Ho, "An Analog-Style Frequency Shifter"
  https://github.com/snappizz/filter-design/blob/master/phase_difference_network.py
- D.E.G. Weaver (1954), "Design of RC Wide-Band 90-Degree Phase-Difference Network"
- Bernie Hutchins, "Musical Engineer's Handbook"
"""

import numpy as np


def design_phase_difference_network(min_freq, max_freq, n_total_poles):
    """
    Design a wideband 90° phase-difference network (analog domain).

    Args:
        min_freq: Lower frequency bound in Hz (ANALOG, pre-warped)
        max_freq: Upper frequency bound in Hz (ANALOG, pre-warped)
        n_total_poles: Total number of poles (split evenly between two paths).
                       Must be even. Each path gets n_total_poles/2 allpass stages.

    Returns:
        poles_a: Analog pole locations for path A (rad/s, negative)
        poles_b: Analog pole locations for path B (rad/s, negative)
        error_degrees: Maximum phase error in degrees
    """
    assert n_total_poles % 2 == 0

    k_prime = min_freq / max_freq
    k = np.sqrt(1.0 - k_prime * k_prime)
    sqrt_k = np.sqrt(k)
    ell = 0.5 * (1.0 - sqrt_k) / (1.0 + sqrt_k)
    q_prime = ell + 2.0 * ell**5 + 15.0 * ell**9
    ln_q = np.pi**2 / np.log(q_prime)
    q = np.exp(ln_q)

    # Phase error estimate
    error_degrees = 720.0 * q**n_total_poles / np.pi

    # Compute pole locations
    r = np.arange(1, n_total_poles + 1)
    phi = np.pi / (4.0 * n_total_poles) * (2.0 * r - 1.0)
    phi_prime = np.arctan(
        (q**2 - q**6) * np.sin(4.0 * phi)
        / (1.0 + (q**2 + q**6) * np.cos(4.0 * phi))
    )
    poles = -2.0 * np.pi * min_freq * np.tan(phi - phi_prime) / np.sqrt(k_prime)

    # Split alternately into two paths
    poles_a = poles[0::2]
    poles_b = poles[1::2]

    return poles_a, poles_b, error_degrees


def analog_pole_to_digital_allpass(pole, sample_rate):
    """
    Convert an analog pole to a digital first-order allpass coefficient.

    Bilinear transform: s = 2*fs * (z-1)/(z+1)
    Coefficient: k = (p + 2*fs) / (p - 2*fs)
    For H(z) = (k + z^-1) / (1 + k*z^-1)
    """
    k = (pole + 2.0 * sample_rate) / (pole - 2.0 * sample_rate)
    return k


def prewarp_frequency(f_digital, sample_rate):
    """
    Pre-warp a digital frequency to its analog equivalent for bilinear transform.

    f_analog = (fs/pi) * tan(pi * f_digital / fs)
    """
    return (sample_rate / np.pi) * np.tan(np.pi * f_digital / sample_rate)


def compute_hilbert_coefficients(n_stages_per_chain, min_freq, max_freq, sample_rate):
    """
    Compute Hilbert transform allpass coefficients for a given sample rate.

    Applies frequency prewarping so the 90° phase-difference band matches
    [min_freq, max_freq] in the DIGITAL domain after bilinear transform.

    Args:
        n_stages_per_chain: Number of allpass stages per chain (e.g., 8)
        min_freq: Lower frequency bound in Hz (digital)
        max_freq: Upper frequency bound in Hz (digital)
        sample_rate: Sample rate in Hz

    Returns:
        coeffs_I: Sorted array of coefficients for I chain
        coeffs_Q: Sorted array of coefficients for Q chain
        error_deg: Estimated phase error in degrees
    """
    n_total = 2 * n_stages_per_chain

    # Pre-warp frequency bounds for bilinear transform
    min_freq_warped = prewarp_frequency(min_freq, sample_rate)
    max_freq_warped = prewarp_frequency(max_freq, sample_rate)

    # Design analog prototype with warped frequencies
    poles_a, poles_b, error_deg = design_phase_difference_network(
        min_freq_warped, max_freq_warped, n_total
    )

    # Convert to digital allpass coefficients via bilinear transform
    coeffs_a = np.array([analog_pole_to_digital_allpass(p, sample_rate) for p in poles_a])
    coeffs_b = np.array([analog_pole_to_digital_allpass(p, sample_rate) for p in poles_b])

    # Sort ascending
    coeffs_I = np.sort(coeffs_a)
    coeffs_Q = np.sort(coeffs_b)

    return coeffs_I, coeffs_Q, error_deg


def verify_phase_accuracy(coeffs_I, coeffs_Q, sample_rate, f_low=20.0, f_high=20000.0):
    """
    Verify the phase accuracy of the Hilbert transform coefficients.

    Uses unwrapped phase computation with direct complex evaluation.
    No z^-1 delay (matching HilbertShifter.h which has no extra delay).

    Returns max phase error in degrees and sideband rejection in dB.
    """
    n_points = 10000
    f_upper = min(f_high, sample_rate * 0.49)
    freqs = np.geomspace(f_low, f_upper, n_points)
    omega = 2.0 * np.pi * freqs / sample_rate
    z = np.exp(1j * omega)

    # Compute total transfer function for each chain
    H_I = np.ones_like(z)
    for a in coeffs_I:
        H_I *= (a + z**(-1)) / (1.0 + a * z**(-1))

    H_Q = np.ones_like(z)
    for a in coeffs_Q:
        H_Q *= (a + z**(-1)) / (1.0 + a * z**(-1))

    # Unwrap phases for accurate difference computation
    phase_I = np.unwrap(np.angle(H_I))
    phase_Q = np.unwrap(np.angle(H_Q))

    # Phase difference
    diff = phase_I - phase_Q

    # Find the target (should be near ±π/2 + n*π for some integer n)
    median_diff = np.median(diff)
    # Round to nearest multiple of π/2
    target = np.round(median_diff / (np.pi / 2)) * (np.pi / 2)

    error = diff - target
    max_error_rad = np.max(np.abs(error))
    max_error_deg = np.degrees(max_error_rad)

    if 0 < max_error_rad < np.pi / 2:
        rejection_db = -20.0 * np.log10(np.abs(np.sin(max_error_rad)))
    else:
        rejection_db = 0.0

    return max_error_deg, rejection_db, np.degrees(target)


def format_cpp_array(coeffs, indent=8):
    """Format coefficients as C++ array initialization."""
    spaces = " " * indent
    lines = [f"{spaces}{c:.13f}{',' if i < len(coeffs)-1 else ''}"
             for i, c in enumerate(coeffs)]
    return "\n".join(lines)


def main():
    sample_rates = [44100, 48000, 88200, 96000, 176400, 192000]
    n_stages = 8  # 8 allpass stages per chain
    f_low = 20.0
    f_high = 20000.0

    print(f"Generating {n_stages}-stage coefficients with frequency prewarping...")
    print(f"Digital passband: {f_low:.0f} Hz to {f_high:.0f} Hz\n")

    all_results = {}

    for sr in sample_rates:
        actual_f_high = min(f_high, 0.49 * sr)

        # Show prewarping
        f_low_w = prewarp_frequency(f_low, sr)
        f_high_w = prewarp_frequency(actual_f_high, sr)
        print(f"  {sr} Hz: prewarped band = [{f_low_w:.1f}, {f_high_w:.1f}] Hz (ratio {f_high_w/f_low_w:.0f}:1)")

        coeffs_I, coeffs_Q, est_err = compute_hilbert_coefficients(
            n_stages, f_low, actual_f_high, sr
        )

        # Verify
        err_deg, rej_db, target_deg = verify_phase_accuracy(
            coeffs_I, coeffs_Q, sr, f_low, actual_f_high
        )

        print(f"    Estimated error: {est_err:.4f}°")
        print(f"    Verified error:  {err_deg:.4f}° (target: {target_deg:.1f}°, rejection: ~{rej_db:.1f} dB)")
        print(f"    I: {coeffs_I}")
        print(f"    Q: {coeffs_Q}")

        all_results[sr] = (coeffs_I, coeffs_Q, err_deg, rej_db)

    # Also test 6-stage for comparison
    print(f"\n6-stage comparison at 44100 Hz:")
    ci6, cq6, err6_est = compute_hilbert_coefficients(6, f_low, f_high, 44100)
    err6_v, rej6_v, tgt6 = verify_phase_accuracy(ci6, cq6, 44100, f_low, f_high)
    print(f"  Verified error: {err6_v:.4f}° (target: {tgt6:.1f}°, rejection: ~{rej6_v:.1f} dB)")
    print(f"  I: {ci6}")
    print(f"  Q: {cq6}")

    # Print C++ code
    n = n_stages
    print(f"\n\n// ===== GENERATED HILBERT COEFFICIENTS =====")
    print(f"// {n}-stage allpass pairs for sample-rate adaptive Hilbert transform")
    print(f"// Generated using Weaver/Darlington phase-difference network design")
    print(f"// with frequency prewarping for bilinear transform accuracy")
    print(f"// {n} first-order allpass sections per chain (I and Q)")
    print(f"// Digital passband: {f_low:.0f} Hz to {f_high:.0f} Hz")
    print()
    print(f"static constexpr int HILBERT_ORDER = {n};")
    print()

    for sr, (ci, cq, err, rej) in all_results.items():
        print(f"// Sample rate: {sr} Hz (max phase error: {err:.4f}°, rejection: ~{rej:.1f} dB)")
        print(f"static constexpr std::array<double, {n}> coeffsI_{sr} = {{{{")
        print(format_cpp_array(ci))
        print(f"}}}};")
        print(f"static constexpr std::array<double, {n}> coeffsQ_{sr} = {{{{")
        print(format_cpp_array(cq))
        print(f"}}}};")
        print()

    # Print the selector function
    print("void selectCoefficients(double sampleRate) {")
    first = True
    for sr in sample_rates:
        kw = "if" if first else "} else if"
        print(f"    {kw} (sampleRate <= {sr + 100}.0) {{")
        print(f"        std::copy(coeffsI_{sr}.begin(), coeffsI_{sr}.end(), coeffsI.begin());")
        print(f"        std::copy(coeffsQ_{sr}.begin(), coeffsQ_{sr}.end(), coeffsQ.begin());")
        first = False
    print("    } else {")
    sr_last = sample_rates[-1]
    print(f"        std::copy(coeffsI_{sr_last}.begin(), coeffsI_{sr_last}.end(), coeffsI.begin());")
    print(f"        std::copy(coeffsQ_{sr_last}.begin(), coeffsQ_{sr_last}.end(), coeffsQ.begin());")
    print("    }")
    print("}")


if __name__ == "__main__":
    main()
