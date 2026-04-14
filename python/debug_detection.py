"""Debug script to see what's happening in burst detection"""
import numpy as np
import json
import argparse
import RFID_signal_processing as rfid
import matplotlib.pyplot as plt

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


def debug_detection(samples, sample_rate):
    """Debug the detection to see what's being filtered out"""
    
    # Get raw second derivative
    second_deriv = np.diff(samples)
    abs_second_deriv = np.abs(second_deriv)
    
    # Calculate threshold
    threshold = np.percentile(abs_second_deriv, 99.98)
    print(f"\n99.98 percentile threshold: {threshold:.6f}")
    print(f"Max second derivative: {np.max(abs_second_deriv):.6f}")
    print(f"Min second derivative (of peaks): {np.min(abs_second_deriv[abs_second_deriv > threshold]):.6f}")
    
    # Find ALL peaks above threshold
    all_peaks = np.where(abs_second_deriv > threshold)[0]
    print(f"\nALL peaks above threshold: {len(all_peaks)}")
    
    # Show distribution
    if len(all_peaks) > 0:
        print(f"Peak indices (first 20): {all_peaks[:20]}")
        print(f"Peak values (first 20): {abs_second_deriv[all_peaks[:20]]}")
        
        # Check spacing
        if len(all_peaks) > 1:
            spacings = np.diff(all_peaks)
            print(f"\nSpacing between consecutive peaks:")
            print(f"  Min: {np.min(spacings)} samples ({np.min(spacings)/sample_rate*1e6:.2f} us)")
            print(f"  Max: {np.max(spacings)} samples ({np.max(spacings)/sample_rate*1e6:.2f} us)")
            print(f"  Mean: {np.mean(spacings):.0f} samples ({np.mean(spacings)/sample_rate*1e6:.2f} us)")
            
            # How many peaks are within 1ms of the previous one?
            min_distance = int(sample_rate * 0.001)  # 1ms
            close_peaks = np.sum(spacings < min_distance)
            print(f"\nPeaks within 1ms ({min_distance} samples) of previous: {close_peaks}/{len(spacings)}")
    
    # Now apply the actual filtering
    filtered_indices = rfid.derivative_burst_detection(samples, sample_rate)
    print(f"\nFinal detected bursts after filtering: {len(filtered_indices)}")
    if len(filtered_indices) > 0:
        print(f"Indices: {filtered_indices}")
    
    # Plot to visualize
    time_axis = np.arange(len(samples)) / sample_rate
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Second derivative
    axes[0].plot(time_axis[:-1], abs_second_deriv, 'b-', linewidth=0.5, label='|d²x/dt²|')
    axes[0].axhline(threshold, color='r', linestyle='--', label=f'99.98% threshold = {threshold:.6f}')
    axes[0].scatter(time_axis[all_peaks], abs_second_deriv[all_peaks], 
                   color='orange', s=20, alpha=0.6, label=f'All peaks ({len(all_peaks)})')
    axes[0].scatter(time_axis[filtered_indices], abs_second_deriv[filtered_indices], 
                   color='green', s=50, marker='*', label=f'After 1ms filter ({len(filtered_indices)})')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Second Derivative of I/Q Samples')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Raw I/Q envelope
    envelope = np.abs(samples)
    axes[1].plot(time_axis, envelope, 'b-', linewidth=0.5, label='I/Q envelope')
    axes[1].scatter(time_axis[filtered_indices], envelope[filtered_indices], 
                   color='green', s=50, marker='*', label='Detected bursts')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('I/Q Envelope with Detected Bursts')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input file prefix')
    args = parser.parse_args()
    
    samples, metadata = load_capture(args.input)
    debug_detection(samples, metadata['sample_rate'])
