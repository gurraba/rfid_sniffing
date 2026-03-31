# generate_synthetic.py

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).parent.parent
CAPTURES_DIR = PROJECT_ROOT / "data" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)





def generate_fm0_signal(bits, sample_rate, bit_rate):
    """
    Generate FM0 as AMPLITUDE modulation (0 to 1
    """
    samples_per_bit = int(sample_rate / bit_rate)
    total_samples = len(bits) * samples_per_bit
    amplitude = np.zeros(total_samples)  # Start with zeros
    
    state = 1  # High state
    sample_idx = 0
    
    for bit in bits:
        state = -state  # Toggle for each bit start
        
        if bit == 1:
            # Transition in middle
            # First half
            amplitude[sample_idx:sample_idx + samples_per_bit//2] = 1 if state > 0 else 0
            state = -state  # Flip in middle
            # Second half  
            amplitude[sample_idx + samples_per_bit//2:sample_idx + samples_per_bit] = 1 if state > 0 else 0
        else:
            # No transition
            amplitude[sample_idx:sample_idx + samples_per_bit] = 1 if state > 0 else 0
        
        sample_idx += samples_per_bit

    return amplitude  # Returns 0 or 1, not ±1


def generate_tag_response(sample_rate=2e6, tag_phase=1.5, 
                               duration=0.002, snr_db=20, bit_rate=320e3):
    
    # Generate bits
    num_bits = int(duration * bit_rate)
    preamble = np.array([0, 1, 0, 1, 0, 1])
    data_bits = np.random.randint(0, 2, num_bits - len(preamble))
    bits = np.concatenate([preamble, data_bits])
    
    # Get amplitude modulation (0 or 1)
    amplitude_mod = generate_fm0_signal(bits, sample_rate, bit_rate)
    
    # Truncate/pad
    target_samples = int(duration * sample_rate)
    if len(amplitude_mod) > target_samples:
        amplitude_mod = amplitude_mod[:target_samples]
    else:
        amplitude_mod = np.pad(amplitude_mod, (0, target_samples - len(amplitude_mod)))
    
    # Add modulation depth (don't go all the way to zero)
    modulation_depth = 0.8
    amplitude_mod = (1 - modulation_depth) + modulation_depth * amplitude_mod
    # Now ranges from 0.2 to 1.0
    
    # Burst envelope
    from scipy.signal.windows import tukey
    envelope = tukey(len(amplitude_mod), alpha=0.2)
    
    # Carrier with constant phase
    carrier = np.exp(1j * tag_phase)
    
    # Combine
    signal = amplitude_mod * envelope * carrier
    
    # Add noise
    t = np.arange(len(signal)) / sample_rate
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(t)) + 1j*np.random.randn(len(t)))
    
    return signal + noise




def plot_detection_debug(iq_data, sample_rate, bursts, threshold_db=10):
    """Visualize detection process"""
    
    power = np.abs(iq_data)**2
    window_samples = int(sample_rate * 100e-6)
    kernel = np.ones(window_samples) / window_samples
    smoothed_power = np.convolve(power, kernel, mode='same')
    
    noise_floor = np.percentile(smoothed_power, 20)
    threshold = noise_floor * (10 ** (threshold_db / 10))
    
    t = np.arange(len(iq_data)) / sample_rate
    
    plt.figure(figsize=(14, 8))
    
    # Power over time
    plt.subplot(3, 1, 1)
    plt.plot(t, 10*np.log10(power + 1e-10), alpha=0.3, label='Raw power')
    plt.plot(t, 10*np.log10(smoothed_power + 1e-10), label='Smoothed power')
    plt.axhline(10*np.log10(threshold), color='r', linestyle='--', label='Threshold')
    plt.axhline(10*np.log10(noise_floor), color='orange', linestyle='--', label='Noise floor')
    
    # Mark detected bursts
    for start, end, peak in bursts:
        plt.axvspan(start/sample_rate, end/sample_rate, alpha=0.2, color='green')
    
    plt.ylabel('Power (dB)')
    plt.legend()
    plt.grid(True)
    plt.title(f'Burst Detection ({len(bursts)} bursts found)')
    
    # Zoom on first burst if exists
    if bursts:
        plt.subplot(3, 1, 2)
        start, end, _ = bursts[0]
        margin = int(0.001 * sample_rate)  # 1ms margin
        zoom_start = max(0, start - margin)
        zoom_end = min(len(iq_data), end + margin)
        
        plt.plot(t[zoom_start:zoom_end], 10*np.log10(power[zoom_start:zoom_end] + 1e-10), alpha=0.3)
        plt.plot(t[zoom_start:zoom_end], 10*np.log10(smoothed_power[zoom_start:zoom_end] + 1e-10))
        plt.axhline(10*np.log10(threshold), color='r', linestyle='--')
        plt.axvspan(start/sample_rate, end/sample_rate, alpha=0.2, color='green')
        plt.ylabel('Power (dB)')
        plt.grid(True)
        plt.title('Zoom on First Burst')
    
    # Histogram of power
    plt.subplot(3, 1, 3)
    plt.hist(10*np.log10(smoothed_power + 1e-10), bins=100, alpha=0.7)
    plt.axvline(10*np.log10(threshold), color='r', linestyle='--', label='Threshold')
    plt.axvline(10*np.log10(noise_floor), color='orange', linestyle='--', label='Noise floor')
    plt.xlabel('Power (dB)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.title('Power Distribution')
    
    plt.tight_layout()
    plt.show()


def generate_experiment(num_tags=3, sample_rate=2e6, duration=10.0,
                       center_freq=866.5e6, gain=20):
    """
    Generate full synthetic capture with multiple tag reads
    
    Returns:
        samples: I/Q data
        events: List of ground truth events (time, phase, rssi)
        metadata: Metadata dict for saving
    """
    samples = np.zeros(int(sample_rate * duration), dtype=np.complex64)
    
    # Add noise floor
    samples += 0.01 * (np.random.randn(len(samples)) + 1j*np.random.randn(len(samples)))
    
    # Add tag responses at known times/phases
    events = []
    for i in range(num_tags):
        time_offset = duration*(i+1)/(num_tags+1)  # Spread out evenly
        tag_phase = np.random.uniform(-np.pi, np.pi)
        tag_phase = np.random.uniform(0, 2*np.pi)  # Random phase for each tag
        tag_rssi = -20.0 + np.random.randn() * 2  # Some variation
        
        tag_signal = generate_tag_response(sample_rate, tag_phase=tag_phase)
        start_idx = int(time_offset * sample_rate)
        
        # Scale by RSSI (roughly)
        scale = 10**(tag_rssi / 20)
        samples[start_idx:start_idx+len(tag_signal)] += tag_signal * scale
        
        events.append({
            'time_offset': float(time_offset),
            'phase': float(tag_phase),
            'rssi': float(tag_rssi)
        })
    
    # Create metadata
    start_time = datetime.now()
    metadata = {
        'center_freq': float(center_freq),
        'sample_rate': float(sample_rate),
        'gain': float(gain),
        'duration': float(duration),
        'start_time': start_time.isoformat(),
        'num_samples': int(len(samples)),
        'synthetic': True,  # Mark as synthetic
        'ground_truth_events': events  # Include ground truth
    }
    
    return samples, events, metadata

def save_synthetic_capture(exp_name, samples, metadata):
    """Save synthetic capture with proper metadata"""
    
    iq_file = CAPTURES_DIR / f"{exp_name}_iq.npy"
    meta_file = CAPTURES_DIR / f"{exp_name}_meta.json"
    
    # Save I/Q
    np.save(iq_file, samples)
    print(f"Saved I/Q: {iq_file}")
    
    # Save metadata
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {meta_file}")
    
    # Print ground truth
    print(f"\n Ground Truth Events:")
    for i, event in enumerate(metadata['ground_truth_events'], 1):
        print(f"  Event {i}: t={event['time_offset']:.2f}s, "
              f"φ={event['phase']:.3f} rad, RSSI={event['rssi']:.1f} dB")
    
    return iq_file, meta_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic RFID data')
    parser.add_argument('--output', default='synthetic', 
                       help='Experiment name')
    parser.add_argument('--num-tags', type=int, default=3,
                       help='Number of tag reads to generate')
    parser.add_argument('--duration', type=float, default=1.0,
                       help='Capture duration (seconds)')
    parser.add_argument('--rate', type=float, default=2e6,
                       help='Sample rate (Hz)')
    
    args = parser.parse_args()
    
    print(f"Generating synthetic capture: {args.output}")
    print(f"  {args.num_tags} tag reads over {args.duration}s")
    
    # Generate
    samples, events, metadata = generate_experiment(
        num_tags=args.num_tags,
        sample_rate=args.rate,
        duration=args.duration
    )
    

    # Save
    save_synthetic_capture(args.output, samples, metadata)

    print(f"\n✓ Synthetic data ready!")
    print(f"\nTest it:")
    print(f"  python process.py {args.output} --print --plot")

if __name__ == "__main__":
    main()