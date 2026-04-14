

"""
process.py - Process captured I/Q files
"""

import csv
from dataclasses import dataclass
from sys import prefix
import time


import numpy as np
import json
import argparse
import RFID_signal_processing as rfid
import matplotlib.pyplot as plt


#DATA_DIR = "C:/Users/gusta/Documents/programmering/RFID_project/data/captures/"
DATA_DIR = "C:/Users/gustav/Documents/programmering/rfid_sniffing/data/captures/"

@dataclass 
class ReaderEvent:
    timestamp: float
    phase: float
    rssi: float
    epc: str
    offset: float = 0.0  # To be calculated later based on capture start time

class ReaderSession:
    def __init__(self, events: list[ReaderEvent], metadata: dict):
        self.events = events
        self.metadata = metadata

    @classmethod
    def from_files(cls, prefix, capture_start_time=None):
        """Load reader CSV events"""
        csv_file = f"{DATA_DIR}/{prefix}_reader.csv"
        metadata_file = f"{DATA_DIR}/{prefix}_reader_meta.json"

        # Check if files exist
        import os
        if not os.path.exists(csv_file):
            print(f"Warning: Reader CSV file not found: {csv_file}")
            return [], {}


        #offsets 
        

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            events = []
            for row in reader:
                events.append(ReaderEvent(
                    timestamp=float(row['Timestamp']),
                    phase=float(row['Phase']),
                    rssi=float(row['RSSI']),
                    epc=row['EPC'],
                    offset = float(row['Timestamp']) - capture_start_time if capture_start_time else 0.0
                ))
    

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        return cls(events, metadata)


class RFIDSignal:
    def __init__(self, samples, sample_rate, center_freq, start_time):
        self.samples = samples
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.start_time = start_time

        self._preprocessed = None
        self._envelope = None
        self._derivative = None
        self._second_derivative = None
        self._preamble_indices = None
        self._correlation = None

    @property
    def preprocessed(self):
        if self._preprocessed is None:
            self._preprocessed = rfid.preprocess_rfid_iq(self.samples, self.sample_rate)
        return self._preprocessed

    @property
    def envelope(self):
        if self._envelope is None:
            self._envelope = np.abs(self.preprocessed)
        return self._envelope
    
    @property
    def derivative(self):
        if self._derivative is None:
            self._derivative = np.abs(np.diff(self.envelope))
        return self._derivative

    @property
    def second_derivative(self):
        if self._second_derivative is None:
            self._second_derivative = np.abs(np.diff(self.derivative))
        return self._second_derivative
    
    @classmethod
    def from_file(cls, prefix, datadir = DATA_DIR):
        """Load I/Q and metadata from files"""  
        
        iq_file = f"{datadir}/{prefix}_iq.npy"
        meta_file = f"{datadir}{prefix}_meta.json"
        
        print(f"Loading {iq_file}...")
        samples = np.load(iq_file)
        
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {len(samples):,} samples")
        return cls(samples, metadata['sample_rate'], metadata['center_freq'], metadata['start_time'])


class RFIDProcessor:
    def __init__(self, capture: RFIDSignal):
        self.capture = capture
        self.results = []

    def detect_preambles(self, bit_rate=640e3):
        peaks = rfid.derivative_burst_detection(
            self.capture.second_derivative, 
            self.capture.sample_rate,
            percentile=99.7)
        self.capture._preamble_indices = peaks
        return peaks
    
    def sliding_Window_preamble_detection(self, bit_rate=640e3):
        #divide the signal into windows of 100ms and use derivative burst detection in each window, then combine results
        window_size = int(0.1 * self.capture.sample_rate)  # 100
        step_size = window_size // 2  # 50% overlap
        preamble_indices = []
        for start in range(0, len(self.capture.samples), step_size):
            end = min(start + window_size, len(self.capture.samples))
            window_samples = self.capture.samples[start:end]
            window_derivative = np.abs(np.diff(np.abs(window_samples)))
            peaks = rfid.derivative_burst_detection(window_derivative, self.capture.sample_rate, percentile=99)
            preamble_indices.extend(peaks + start)  # Adjust indices to original signal

        self.capture._preamble_indices = preamble_indices
        return preamble_indices

    def extract_measurements(self):
        results = []
        for i, idx in enumerate(self.capture._preamble_indices):
            idx = int(idx) 

            preamble_duration = int(6/640e3 * self.capture.sample_rate) # 6 bits at 640 kbps

            segment_start = idx  # we want the rn16 response, which comes after the preamble
            segment_end = segment_start + int(25e-6 * self.capture.sample_rate)  # Analyze 25us segment after preamble

            CW_duration = int(30e-6 * self.capture.sample_rate)  # 30us of CW for frequency estimation
            CW_start = idx - CW_duration - int(3e-6 * self.capture.sample_rate)  # Start 10us before preamble to ensure we capture CW
            CW_end = CW_start + CW_duration

            if CW_start < 0 or CW_end > len(self.capture.samples):
                print(f"Warning: CW segment for burst {i} is out of bounds (start={CW_start}, end={CW_end}, len={len(self.capture.samples)})")
                continue

            CW_segment = self.capture.samples[CW_start:CW_end]
            if CW_segment.size == 0:
                print(f"Warning: CW segment for burst {i} is empty")
                continue

            phase_unwrapped = np.unwrap(np.angle(CW_segment))
            t = np.arange(len(CW_segment)) / self.capture.sample_rate

            if len(t) < 2:
                print(f"Warning: Not enough CW samples for burst {i} to estimate frequency")
                continue

            slope, intercept = np.polyfit(t, phase_unwrapped, 1)

            rn16_time_offset = (idx - CW_start) / self.capture.sample_rate
            expected_phase_offset = (slope * rn16_time_offset + intercept) 

            if segment_end > len(self.capture.samples):
                print(f"Warning: Segment {i} extends beyond capture")
                continue
        
            segment = self.capture.samples[segment_start:segment_end]

            measurements = rfid.extract_measurements(segment)
    
            #phase corection using the CW segment
            phase = measurements['phase'] - expected_phase_offset
            local_offset = slope * rn16_time_offset + intercept - measurements['phase']
          

            results.append({
                'burst_id': int(i),
                'preamble_index': int(idx),
                'offset_seconds': float(idx / self.capture.sample_rate),
                'phase': measurements['phase'], 
                'rssi': measurements['rssi'],
                'frequency_corrected_phase': float(phase),
                'frequency_correction_offset_hz': float(local_offset)
            })

        self.results = results

        return results
        





def plot_phase_rssi(signal_obj, reader_events=None):
    """Plot phase and RSSI from detected preambles stored in an RFIDSignal object."""
    if not signal_obj.results:
        signal_obj.process_preambles()

    results = signal_obj.results
    if len(results) == 0:
        print("No results to plot")
        return

    times = np.array([r['offset_seconds'] for r in results])
    phases = np.array([r['phase'] for r in results])
    rssis = np.array([r['rssi'] for r in results])
    corrected_phases = np.array([r['frequency_corrected_phase'] for r in results])

    phases_unwrapped = rfid.custom_phase_unwrap(phases)
    corrected_phases_unwrapped = rfid.custom_phase_unwrap(corrected_phases)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # Raw phase (wrapped)
    axes[0].plot(times, phases, 'bo-', alpha=0.7, markersize=4)
    axes[0].set_ylabel('Phase (rad)', fontsize=12)
    axes[0].set_title('Raw Phase (Wrapped) - Jumps at ±π', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(np.pi, color='r', linestyle='--', alpha=0.3, label='±π boundary')
    axes[0].axhline(-np.pi, color='r', linestyle='--', alpha=0.3)
    axes[0].legend()

    # Unwrapped phase
    axes[1].plot(times, phases_unwrapped, 'go-', alpha=0.7, markersize=4)
    axes[1].set_ylabel('Phase (rad)', fontsize=12)
    axes[1].set_title('Unwrapped Phase - Continuous', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # RSSI
    axes[2].plot(times, rssis, 'ro-', alpha=0.7, markersize=4)
    axes[2].set_xlabel('Time (s)', fontsize=12)
    axes[2].set_ylabel('RSSI (dB)', fontsize=12)
    axes[2].set_title('RSSI After Each Detected Preamble', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Frequency-corrected phase
    axes[3].plot(times, corrected_phases_unwrapped, 'mo-', alpha=0.7, markersize=4)
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].set_ylabel('Phase (rad)', fontsize=12)
    axes[3].set_title('Frequency-Corrected Phase', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*60}")
    print(f"Phase/RSSI Statistics ({len(results)} preambles)")
    print(f"{'='*60}")
    print(f"Raw phase:       mean={np.mean(phases):.3f} rad, std={np.std(phases):.3f} rad")
    print(f"Unwrapped phase: mean={np.mean(phases_unwrapped):.3f} rad, std={np.std(phases_unwrapped):.3f} rad")
    print(f"RSSI:            mean={np.mean(rssis):.1f} dB, std={np.std(rssis):.1f} dB")
    print(f"Phase range (raw):       [{np.min(phases):.3f}, {np.max(phases):.3f}] rad")
    print(f"Phase range (unwrapped): [{np.min(phases_unwrapped):.3f}, {np.max(phases_unwrapped):.3f}] rad")


def plot_capture(signal_obj, output_file=None, event_lines=None, reader_events=None):
    """Plot a capture from an RFIDSignal object."""
    samples = signal_obj.samples
    sample_rate = signal_obj.sample_rate
    center_freq = signal_obj.center_freq

    print("Generating plots...")
    t = np.arange(len(samples)) / sample_rate
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    non_zero_indices = np.where(np.abs(samples) > 1e-6)[0]
    if len(non_zero_indices) > 0:
        start_idx = max(0, non_zero_indices[0] - int(0.01 * sample_rate))
        end_idx = min(len(samples), non_zero_indices[-1] + int(0.01 * sample_rate))
        samples = samples[start_idx:end_idx]
        t = t[start_idx:end_idx]

    nonzero = np.where(np.abs(samples) > 0.001)[0]
    if len(nonzero) == 0:
        print("Warning: No significant signal found")
        return

    axes[0].plot(t[nonzero[0]:nonzero[-1]], np.abs(samples[nonzero[0]:nonzero[-1]]), linewidth=0.5, color='green')
    axes[0].set_title('I/Q Magnitude', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid(True, alpha=0.3)

    if event_lines is not None:
        for event_name, event_data in event_lines.items():
            indices = event_data['indices']
            color = event_data.get('color', 'black')
            linestyle = event_data.get('linestyle', '--')
            label = event_data.get('label', event_name)
            for i, idx in enumerate(indices):
                axes[0].axvline(idx / sample_rate, color=color, linestyle=linestyle,
                               alpha=0.7, linewidth=1.5,
                               label=label if i == 0 else '')
        axes[0].legend(loc='upper right')

    axes[1].plot(t, np.abs(signal_obj.preprocessed), linewidth=0.5, color='blue')
    axes[1].set_title('Preprocessed I/Q Envelope', fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Envelope')
    axes[1].grid(True, alpha=0.3)

    fft = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))
    power_db = 20 * np.log10(np.abs(fft) + 1e-10)
    axes[2].plot(freqs / 1e6 + center_freq / 1e6, power_db, linewidth=0.5)
    axes[2].set_title('Frequency Spectrum', fontweight='bold')
    axes[2].set_xlabel('Frequency (MHz)')
    axes[2].set_ylabel('Power (dB)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([center_freq/1e6 - sample_rate/2e6, center_freq/1e6 + sample_rate/2e6])

    axes[3].plot(t[1:], signal_obj.derivative, linewidth=0.5, color='orange')
    axes[3].set_title('Envelope Derivative (shows transitions)', fontweight='bold')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('d|x|/dt')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {output_file}")
    plt.show()


    """
    Plot I/Q capture data
    
    Args:
        samples: I/Q samples
        sample_rate: Sample rate in Hz
        center_freq: Center frequency in Hz
        output_file: Optional file to save plot
        event_lines: dict of events to mark with lines, e.g.:
                    {
                        'preambles': {'indices': [...], 'color': 'red', 'linestyle': '--', 'label': 'Preamble'},
                        'reader_on': {'indices': [...], 'color': 'green', 'linestyle': ':', 'label': 'Reader ON'}
                    }
        reader_df: Optional DataFrame with reader events to overlay
    """
    
    print("Generating plots...")
    
    t = np.arange(len(samples)) / sample_rate
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # Find signal region
    non_zero_indices = np.where(np.abs(samples) > 1e-6)[0]
    if len(non_zero_indices) > 0:
        start_idx = max(0, non_zero_indices[0] - int(0.01 * sample_rate))
        end_idx = min(len(samples), non_zero_indices[-1] + int(0.01 * sample_rate))
        samples = samples[start_idx:end_idx]
        t = t[start_idx:end_idx]
    
    nonzero = np.where(np.abs(samples) > 0.001)[0]
    
    if len(nonzero) == 0:
        print("Warning: No significant signal found")
        return
    

    # Plot 1: I/Q magnitude
    axes[0].plot(t[nonzero[0]:nonzero[-1]], 
                 np.abs(samples[nonzero[0]:nonzero[-1]]), 
                 linewidth=0.5, color='green')
    axes[0].set_title('I/Q Magnitude', fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid(True, alpha=0.3)
    
    # Mark event lines
    if event_lines is not None:
        for event_name, event_data in event_lines.items():
            indices = event_data['indices']
            color = event_data.get('color', 'black')
            linestyle = event_data.get('linestyle', '--')
            label = event_data.get('label', event_name)
            
            for i, idx in enumerate(indices):
                axes[0].axvline(idx / sample_rate, color=color, linestyle=linestyle, 
                               alpha=0.7, linewidth=1.5,
                               label=label if i == 0 else '')
        
        axes[0].legend(loc='upper right')

    # Plot 3: Frequency spectrum
    fft = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sample_rate))
    power_db = 20 * np.log10(np.abs(fft) + 1e-10)
    
    axes[2].plot(freqs/1e6 + center_freq/1e6, power_db, linewidth=0.5)
    axes[2].set_title('Frequency Spectrum', fontweight='bold')
    axes[2].set_xlabel('Frequency (MHz)')
    axes[2].set_ylabel('Power (dB)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([center_freq/1e6 - sample_rate/2e6, 
                      center_freq/1e6 + sample_rate/2e6])
    
    # Plot 4: Derivative
    derivative = np.diff(samples, 2)
    derivative = np.abs(derivative)

    axes[3].plot(t[2:], derivative, linewidth=0.5, color='orange')
    axes[3].set_title('Derivative of Magnitude (shows transitions)', fontweight='bold')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('dM/dt')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {output_file}")
    
    plt.show()





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


def load_reader_csv(prefix):

    """Load reader CSV events"""
    csv_file = f"{DATA_DIR}/{prefix}_reader.csv"
    metadata_file = f"{DATA_DIR}/{prefix}_reader_meta.json"

    # Check if files exist
    import os
    if not os.path.exists(csv_file):
        print(f"Warning: Reader CSV file not found: {csv_file}")
        return [], {}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        events = []
        for row in reader:
           events.append({
               'timestamp': float(row['Timestamp']),
                'phase': float(row['Phase']),
                'rssi': float(row['RSSI']),
                'epc': row['EPC']
           })
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    return events, metadata



def process_samples(samples, sample_rate, threshold=3, reader_timing=False):
    """
    Process I/Q samples to detect tag reads
    
    Returns: list of detected bursts with phase/RSSI
    """

    
    print("Preprocessing...")
    iq_clean = rfid.preprocess_rfid_iq(samples, sample_rate)

    # let's find preambles using the derivative of the signal
    #preamble_indices = rfid.derivative_burst_detection(np.diff(samples), sample_rate)

    preamble_indices, pulse_template = rfid.detect_fm0_periodic(iq_clean, sample_rate, bit_rate=640e3)
    
    # plt.figure(figsize=(12, 4))
    # plt.plot(pulse_template, label='FM0 Pulse Template')
    # plt.title('FM0 Pulse Template Detected from Capture', fontweight='bold')
    


    results = []
    for i, idx in enumerate(preamble_indices):

        preamble_duration = int(6/640e3 * sample_rate) # 6 bits at 640 kbps

        segment_start = idx  # we want the rn16 response, which comes after the preamble
        segment_end = segment_start + int(25e-6 * sample_rate)  # Analyze 25us segment after preamble

        CW_duration = int(30e-6 * sample_rate)  # 30us of CW for frequency estimation
        CW_start = idx - CW_duration - int(3e-6 * sample_rate)  # Start 10us before preamble to ensure we capture CW
        CW_end = CW_start + CW_duration

        if CW_start < 0 or CW_end > len(iq_clean):
            print(f"Warning: CW segment for burst {i} is out of bounds (start={CW_start}, end={CW_end}, len={len(iq_clean)})")
            continue

        CW_segment = iq_clean[CW_start:CW_end]
        if CW_segment.size == 0:
            print(f"Warning: CW segment for burst {i} is empty")
            continue

        phase_unwrapped = np.unwrap(np.angle(CW_segment))
        t = np.arange(len(CW_segment)) / sample_rate

        if len(t) < 2:
            print(f"Warning: Not enough CW samples for burst {i} to estimate frequency")
            continue

        slope, intercept = np.polyfit(t, phase_unwrapped, 1)

        rn16_time_offset = (idx - CW_start) / sample_rate
        expected_phase_offset = (slope * rn16_time_offset + intercept) 



        if segment_end > len(iq_clean):
            print(f"Warning: Segment {i} extends beyond capture")
            continue
    
        segment = iq_clean[segment_start:segment_end]

        measurements = rfid.extract_measurements(segment)
        # extracted_frequency = rfid.estimate_local_frequency_offset(CW_segment, sample_rate)
        # corrected_segment = rfid.correct_frequency_offset(segment, extracted_frequency, sample_rate)
        # phase, local_offset = rfid.extract_phase_with_local_correction(corrected_segment, sample_rate)

        #phase corection using the CW segment
        phase = measurements['phase'] - expected_phase_offset
        local_offset = slope * rn16_time_offset + intercept - measurements['phase']
        #print(local_offset)

        

        results.append({
            'burst_id': int(i),
            'preamble_index': int(idx),
            'offset_seconds': float(idx / sample_rate),
            'phase': measurements['phase'], 
            'rssi': measurements['rssi'],
            'frequency_corrected_phase': float(phase),
            'frequency_correction_offset_hz': float(local_offset)
        })
    return results, preamble_indices


def add_absolute_timestamps(results, start_time):
    """Add absolute timestamps to results"""
    from datetime import datetime
    
    for result in results:
        result['timestamp'] = start_time + result['offset_seconds']
    
    return results


def save_results(output_file, results, metadata, processing_params):
    """Save processing results to JSON"""
    
    output = {
        'metadata': metadata,
        'processing': processing_params,
        'num_bursts': len(results),
        'bursts': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved results: {output_file}")


def print_summary(results):
    """Print summary of results"""
    print(f"\n{'='*60}")
    print(f"Processing Results ({len(results)} bursts)")
    print(f"{'='*60}\n")
    
    for r in results[:5]:  # Print first 5
        print(f"Burst {r['burst_id']}:")
        print(f"  Time:  {r['offset_seconds']:.3f} s")
        print(f"  Phase: {r['phase']:.3f} rad")
        print(f"  RSSI:  {r['rssi']:.1f} dB")
        print()
    
    if len(results) > 5:
        print(f"... and {len(results) - 5} more")





def detect_reader_bursts(iq_data, sample_rate):
    """
    Detect large energy bursts when reader turns on
    
    Returns:
        Array of indices where reader turns on
    """
    envelope = np.abs(iq_data)
    derivative = np.abs(np.diff(envelope))
    
    # Large positive derivatives = reader turning on
    threshold = np.percentile(derivative, 99.5)  # Top 0.5%
    reader_on = np.where(derivative > threshold)[0]
    
    # Filter: reader bursts should be at least 1ms apart
    min_spacing = int(0.001 * sample_rate)
    filtered = []
    last_idx = -min_spacing
    
    for idx in reader_on:
        if idx - last_idx > min_spacing:
            filtered.append(idx)
            last_idx = idx
    
    return np.array(filtered)


def main():
    parser = argparse.ArgumentParser(description='Process RFID I/Q data')
    parser.add_argument('input', help='Input file prefix (without _iq.npy)')
    parser.add_argument('--threshold', type=float, default=3)
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--print', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-phase', action='store_true')
    args = parser.parse_args()

    # Load
    signal = RFIDSignal.from_file(args.input)
    reader_session = ReaderSession.from_files(args.input)

    # Align reader event timestamps to capture
    for event in reader_session.events:
        event.offset = event.timestamp - signal.start_time

    # Process
    processor = RFIDProcessor(signal)
    processor.sliding_Window_preamble_detection()
    results = processor.extract_measurements()

    # Add absolute timestamps
    for r in results:
        r['timestamp'] = signal.start_time + r['offset_seconds']

    # Save
    if args.output:
        save_results(args.output, results, signal, {'threshold': args.threshold})

    # Print
    if args.print or not args.output:
        print_summary(results)

    # Plot
    if args.plot:
        event_lines = {
            'preambles': {
                'indices': signal._preamble_indices,
                'color': 'red', 'linestyle': '--', 'label': 'Preamble'
            }
        }
        plot_capture(signal, event_lines=event_lines,
                     reader_events=reader_session.events or None)

    if args.plot_phase:
        plot_phase_rssi(processor, reader_session.events or None)

if __name__ == "__main__":
    main()