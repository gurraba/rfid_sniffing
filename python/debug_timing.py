"""Debug script to compare detect_preamble_fm0 vs derivative_burst_detection timing"""
import numpy as np
import json
import argparse
import RFID_signal_processing as rfid

DATA_DIR = "C:/Users/gustav/Documents/programmering/rfid_sniffing/data/captures/"

def load_capture(prefix):
    """Load I/Q and metadata from files"""
    iq_file = f"{DATA_DIR}/{prefix}_iq.npy"
    meta_file = f"{DATA_DIR}{prefix}_meta.json"

    print(f"Loading {iq_file}...")
    samples = np.load(iq_file)

    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded {len(samples):,} samples")
    return samples, metadata

def debug_timing_comparison(samples, sample_rate):
    """Compare timing between the two detection methods"""

    # Preprocess for detect_preamble_fm0
    iq_clean = rfid.preprocess_rfid_iq(samples, sample_rate)

    # Method 1: detect_preamble_fm0
    preamble_indices_fm0 = rfid.detect_preamble_fm0(iq_clean, sample_rate, bit_rate=640e3)

    # Method 2: derivative_burst_detection (as called in process.py)
    preamble_indices_deriv = rfid.derivative_burst_detection(np.diff(samples), sample_rate)

    print("=== TIMING COMPARISON ===")
    print(f"FM0 detection found {len(preamble_indices_fm0)} preambles")
    print(f"Derivative detection found {len(preamble_indices_deriv)} preambles")

    if len(preamble_indices_fm0) > 0 and len(preamble_indices_deriv) > 0:
        print("First few FM0 indices:", preamble_indices_fm0[:5])
        print("First few Derivative indices:", preamble_indices_deriv[:5])

        # Calculate timing differences
        if len(preamble_indices_fm0) > 0 and len(preamble_indices_deriv) > 0:
            diff = preamble_indices_fm0[0] - preamble_indices_deriv[0]
            diff_seconds = diff / sample_rate
            print(diff_seconds)
    # Check what preprocessing does
    print("=== PREPROCESSING CHECK ===")
    print(f"Original samples mean: {np.mean(np.abs(samples)):.6f}")
    print(f"DC-removed samples mean: {np.mean(np.abs(iq_clean)):.6f}")
    print(f"Are they the same? {np.allclose(samples, iq_clean)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file prefix')
    args = parser.parse_args()

    samples, metadata = load_capture(args.input)
    debug_timing_comparison(samples, metadata['sample_rate'])