

"""
process.py - Process captured I/Q files
"""

import csv
from dataclasses import dataclass
from platform import processor
from sys import prefix
import time


import numpy as np
import json
import argparse
import RFID_signal_processing as rfid
import matplotlib.pyplot as plt


DATA_DIR = "C:/Users/gusta/Documents/programmering/RFID_project/data/captures/"
#DATA_DIR = "C:/Users/gustav/Documents/programmering/rfid_sniffing/data/captures/"


#store all the tags "ideal" epc numbers in a dict for easy lookup when decoding.
IDEAL_EPCs = ["532FFC75FFB9BD6BF3FE6BFFFF77771A48", "532FFC75FFB9BD6BF3FE6BFFFFBBBA4190", "533FFC75FFB9B96BF3FE641C69734E9BB8" ]

@dataclass 
class RFIDBurst:
    start_index: int
    end_index: int
    peak_index: int
    burst_type: str = "unknown"  # Could be 'epc', 'rn16'

    phase: float = 0.0
    rssi: float = 0.0

    frequency_corrected_phase: float = 0.0
    frequency_correction_offset_hz: float = 0.0

    tag_id: int = -1 # the index 

    @property
    def duration_samples(self):
        return self.end_index - self.start_index
    
    def duration_us(self, sample_rate):
        return self.duration_samples / sample_rate * 1e6
    

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
        self._correlation = None

        self._smoothed_derivative = None

        self.bursts: list[RFIDBurst] = []

        self.peaks = 0

   
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
            self._derivative = np.abs(np.diff(self.samples))
        return self._derivative

    @property
    def second_derivative(self):
        if self._second_derivative is None:
            #smoothed_envelope = rfid.moving_average_filter(self.samples, self.sample_rate, window=1e-5)
            self._second_derivative = np.abs(np.diff(self.samples, 2))

        return self._second_derivative
    
    @property
    def smoothed_derivative(self):
        #return THE FUCKING SMOOTHED DERIVATIVE AND DOES NOTHING ELSE
        from scipy.ndimage import uniform_filter1d
        window_samples = int(5e-6 * self.sample_rate)  # 50us window
        moving_avg = uniform_filter1d(self.envelope, size=window_samples)
        self._smoothed_derivative = np.abs(np.diff(moving_avg))
        return self._smoothed_derivative    
    
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

    def detect_peaks(self, bit_rate=640e3):
        peaks = rfid.derivative_burst_detection(
            self.capture.second_derivative, 
            self.capture.sample_rate,
            percentile=99.98)
        self.capture._peak_indicies = peaks
        return peaks
    
    def sliding_window_peak_detection(self, bit_rate=640e3):    
        window_size = int(0.01 * self.capture.sample_rate)
        step_size = window_size
        self.capture.bursts = []
        samples_per_bit = int(self.capture.sample_rate / bit_rate)

        for start in range(0, len(self.capture.second_derivative), step_size):
            end = min(start + window_size, len(self.capture.second_derivative))
            window = self.capture.second_derivative[start:end]
            
            peaks = rfid.derivative_burst_detection(window, self.capture.sample_rate, percentile=99.98)
            #append peaks to capture.peaks, but adjust for the window offset
      

            for peak_idx in peaks:

                self.capture.peaks += 1

                global_peak_idx = int(peak_idx + start)
                
                # Skip if this peak is within an already-detected burst
                if any(b.start_index <= global_peak_idx <= b.end_index 
                    for b in self.capture.bursts):
                    continue
                
                boundaries = rfid.find_burst_boundaries(
                    self.capture.second_derivative,
                    global_peak_idx,
                    self.capture.sample_rate)
                
                if boundaries is None:
                    continue
                
                burst = RFIDBurst(
                    start_index=boundaries[0],
                    end_index=boundaries[1],
                    peak_index=global_peak_idx
                )
                
                dur = burst.duration_us(self.capture.sample_rate)
                if dur < 60:
                    burst.burst_type = 'rn16'
                elif dur > 100:
                    burst.burst_type = 'epc'
                
                self.capture.bursts.append(burst)

        print(f"Detected {len(self.capture.bursts)} bursts "
            f"({sum(1 for b in self.capture.bursts if b.burst_type == 'rn16')} RN16, "
            f"{sum(1 for b in self.capture.bursts if b.burst_type == 'epc')} EPC, "
            f"{sum(1 for b in self.capture.bursts if b.burst_type == 'unknown')} unknown)")

    def extract_measurements(self):
        for burst in self.capture.bursts:
            # if burst.burst_type != 'epc':
            #     continue
            
            idx = burst.start_index
            #if we have boundaries, we should have those as the start and end.
 
            CW_duration = int(30e-6 * self.capture.sample_rate)
            CW_start = idx - CW_duration - int(3e-6 * self.capture.sample_rate)
            CW_end = CW_start + CW_duration

            if CW_start < 0 or CW_end > len(self.capture.samples):
                print(f"Warning: CW segment out of bounds for burst at {idx}")
                continue

            CW_segment = self.capture.samples[CW_start:CW_end]
            if CW_segment.size == 0:
                print(f"Warning: CW segment empty for burst at {idx}")
                continue

            phase_unwrapped = np.unwrap(np.angle(CW_segment))
            t = np.arange(len(CW_segment)) / self.capture.sample_rate

            if len(t) < 2:
                continue

            slope, intercept = np.polyfit(t, phase_unwrapped, 1)
            rn16_time_offset = (idx - CW_start) / self.capture.sample_rate
            expected_phase_offset = slope * rn16_time_offset + intercept

            segment_end = idx + int(25e-6 * self.capture.sample_rate)
            if burst.end_index > len(self.capture.samples):
                print(f"Warning: Segment extends beyond capture for burst at {idx}")
                continue

            segment = self.capture.samples[idx:segment_end]
            measurements = rfid.extract_measurements(segment)

            burst.phase = measurements['phase']
            burst.rssi = measurements['rssi']
            burst.frequency_corrected_phase = float(measurements['phase'] - expected_phase_offset)
            burst.frequency_correction_offset_hz = float(slope * rn16_time_offset + intercept - measurements['phase'])

    def _pair_rn16_epc(self):
        pairs = []
        unpaired_epc = 0
        for i, burst in enumerate(self.capture.bursts):
            if burst.burst_type != 'epc':
                continue
            for j in range(i-1, -1, -1):
                if self.capture.bursts[j].burst_type == 'rn16':
                    pairs.append((self.capture.bursts[j], burst))
                    break
            else:
                unpaired_epc += 1
                dur = burst.duration_us(self.capture.sample_rate)
                print(f"Unpaired EPC at {burst.peak_index/self.capture.sample_rate:.4f}s, "
                    f"dur={dur:.1f}us, preceding burst types: "
                    f"{[self.capture.bursts[k].burst_type for k in range(max(0,i-3), i)]}")
        
        print(f"Paired {len(pairs)} EPCs, {unpaired_epc} unpaired")
        return pairs

    def _hex_to_bits(self, hex_strings):
        result = []
        for epc_hex in hex_strings:
            bits = []
            for nibble in epc_hex.replace(':', '').replace(' ', ''):
                val = int(nibble, 16)
                bits.extend([(val >> (3 - i)) & 1 for i in range(4)])
            result.append(bits)
        return result
    
    def _bits_to_hex(self, bits):
        hex_string = ''
        for i in range(0, len(bits), 4):
            nibble_bits = bits[i:i+4]
            if len(nibble_bits) < 4:
                nibble_bits += [0] * (4 - len(nibble_bits))
            val = (nibble_bits[0] << 3) | (nibble_bits[1] << 2) | (nibble_bits[2] << 1) | nibble_bits[3]
            hex_string += f"{val:X}"
        return hex_string

    
    def decode_epcs(self, known_epcs_hex: list[str]):
        
        """Returns list of (rn16_burst, epc_burst, tag_id) for matched bursts."""
        known_epcs_bits = self._hex_to_bits(known_epcs_hex)
        print(f"Known EPC bit lengths: {[len(b) for b in known_epcs_bits]}")
        matches = []
        all_decoded = []
        i = 0
        
        for rn16, epc_burst in self._pair_rn16_epc():
            segment = self.capture.samples[epc_burst.start_index:epc_burst.end_index]
            bits = rfid.decode_fm0_burst(segment, self.capture.sample_rate)

            if bits is None:          # check BEFORE using bits
                all_decoded.append(None)
                continue

            hex_decoded = self._bits_to_hex(bits)
            all_decoded.append(hex_decoded)
            if bits is None:
                continue

            result = rfid.match_epc(bits, known_epcs_bits, max_shift=3)
            if result is not None:
                tag_id, shift = result
                matches.append((rn16, epc_burst, tag_id))
            i+=1
            
        #print the most common decoded EPCs for debugging
        from collections import Counter
        decoded_counts = Counter(all_decoded)
        print("Decoded EPCs (most common first):")
        for epc, count in decoded_counts.most_common(10):
            print(f"{epc}: {count} times")
        
        
        return matches

    
    def decode_epcs_without_shifting(self, known_epcs_hex: list[str]):
        matches = []
        all_decoded = []
        
        # Normalize known EPCs for comparison (remove colons, uppercase)
        known_normalized = [e.replace(':', '').upper() for e in known_epcs_hex]
        
        for rn16, epc_burst in self._pair_rn16_epc():
            segment = self.capture.samples[epc_burst.start_index:epc_burst.end_index]
            bits = rfid.decode_fm0_burst(segment, self.capture.sample_rate)

            if bits is None:
                all_decoded.append(None)
                continue

            hex_decoded = self._bits_to_hex(bits)
            all_decoded.append(hex_decoded)

            # #no shifting. just compare the decoded hex directly to the known EPCs. if it matches any, assign that tag ID, otherwise None.
            # tag_id = None
            # for shift in range(-3, 4):
            #     if shift >= 0:
            #         shifted_bits = bits[shift:]
            #     else:
            #         shifted_bits = [0] * (-shift) + list(bits)
                
            #     shifted_hex = self._bits_to_hex(shifted_bits)
                
            #     for i, known in enumerate(known_normalized):
            #         compare_len = min(len(known), len(shifted_hex))
            #         if shifted_hex[:compare_len] == known[:compare_len]:
            #             tag_id = i
            #             break
            #     if tag_id is not None:
            #         break

            # if tag_id is not None:
            #     matches.append((rn16, epc_burst, tag_id))

            tag_id = None
            if hex_decoded in known_normalized:
                tag_id = known_normalized.index(hex_decoded)
                matches.append((rn16, epc_burst, tag_id))

        from collections import Counter
        decoded_counts = Counter(all_decoded)
        print("Decoded EPCs (most common first):")
        for epc, count in decoded_counts.most_common(10):
            print(f"  {epc}: {count} times")
        print(f"Matched {len(matches)}/{len(all_decoded)} bursts")

        return matches

    
    
    
    def assign_tag_ids(self, matches):
        for rn16, epc_burst, tag_id in matches:
            rn16.tag_id = tag_id
            epc_burst.tag_id = tag_id 
    
    def assign_tag_ids_from_reader(self, reader_session):
        """
        Assign tag IDs to RN16 bursts based on reader-reported intervals.
        All RN16 bursts between two consecutive reader timestamps belong
        to the tag reported at the END of that interval.
        """
        if not reader_session.events:
            print("No reader events to assign from")
            return

        events = sorted(reader_session.events, key=lambda e: e.offset)
        
        uniqe_epcs = []
        for event in events:
            if event.epc not in uniqe_epcs:
                uniqe_epcs.append(event.epc)

        # Build intervals: (start_offset, end_offset, tag_id)
        intervals = []
        for i, event in enumerate(events):
            start = events[i-1].offset if i > 0 else 0.0
            end = event.offset
            tag_id = uniqe_epcs.index(event.epc)
            intervals.append((start, end, tag_id))

        # Assign each RN16 burst to the interval it falls in
        matched = 0
        for burst in self.capture.bursts:
            if burst.burst_type != 'rn16':
                continue
            
            burst_time = burst.peak_index / self.capture.sample_rate
            for start, end, tag_id in intervals:
                if start <= burst_time <= end:
                    burst.tag_id = tag_id
                    matched += 1
                    break

        print(f"Assigned {matched} RN16 bursts from {len(intervals)} reader intervals")
        print(f"Unassigned: {sum(1 for b in self.capture.bursts if b.burst_type == 'rn16' and b.tag_id == -1)}")


def plot_phase_rssi(processor, reader_events=None):
    bursts = [b for b in processor.capture.bursts 
              if b.burst_type == 'rn16']
    
    if len(bursts) == 0:
        print("No measured RN16 bursts to plot")
        return

    sample_rate = processor.capture.sample_rate
    times = np.array([b.peak_index / sample_rate for b in bursts])
    phases = np.array([b.phase for b in bursts])
    rssis = np.array([b.rssi for b in bursts])
    corrected_phases = np.array([b.frequency_corrected_phase for b in bursts])

    phases_unwrapped = rfid.custom_phase_unwrap(phases)
    corrected_phases_unwrapped = rfid.custom_phase_unwrap(corrected_phases)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    axes[0].plot(times, phases, 'bo-', alpha=0.7, markersize=4)
    axes[0].set_ylabel('Phase (rad)')
    axes[0].set_title('Raw Phase (Wrapped)', fontweight='bold')
    axes[0].axhline(np.pi, color='r', linestyle='--', alpha=0.3, label='±π boundary')
    axes[0].axhline(-np.pi, color='r', linestyle='--', alpha=0.3)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, phases_unwrapped, 'go-', alpha=0.7, markersize=4)
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_title('Unwrapped Phase', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, rssis, 'ro-', alpha=0.7, markersize=4)
    axes[2].set_ylabel('RSSI (dB)')
    axes[2].set_title('RSSI', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(times, corrected_phases_unwrapped, 'mo-', alpha=0.7, markersize=4)
    axes[3].set_ylabel('Phase (rad)')
    axes[3].set_title('Frequency-Corrected Phase', fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*60}")
    print(f"Statistics ({len(bursts)} RN16 bursts)")
    print(f"{'='*60}")
    print(f"Raw phase:       mean={np.mean(phases):.3f}, std={np.std(phases):.3f} rad")
    print(f"Unwrapped phase: mean={np.mean(phases_unwrapped):.3f}, std={np.std(phases_unwrapped):.3f} rad")
    print(f"RSSI:            mean={np.mean(rssis):.1f}, std={np.std(rssis):.1f} dB")

    
def plot_phase_rssi_per_tag(processor, reader_events=None):
    
    tag_ids = sorted(set(b.tag_id for b in processor.capture.bursts if b.burst_type == 'rn16' and b.phase != 0.0 and b.tag_id != -1))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    def tag_label(tag_id):
        if 0 <= tag_id < len(IDEAL_EPCs):
            return IDEAL_EPCs[tag_id].replace(':', '')[-4:]
        return str(tag_id)

    for i, tag_id in enumerate(tag_ids):
        color = colors[i % len(colors)]
        bursts = [b for b in processor.capture.bursts 
                  if b.burst_type == 'rn16' and b.phase != 0.0 and b.tag_id == tag_id]
        
        sample_rate = processor.capture.sample_rate
        times = np.array([b.peak_index / sample_rate for b in bursts])
        rssis = np.array([b.rssi for b in bursts])
        corrected_phases = np.array([b.frequency_corrected_phase for b in bursts])
        corrected_phases_unwrapped = rfid.custom_phase_unwrap(corrected_phases)

        label = f'Tag {tag_label(tag_id)}'
        axes[0].plot(times, rssis, marker='o', linestyle='-', alpha=0.7, markersize=4, color=color, label=label)
        axes[1].plot(times, corrected_phases_unwrapped, marker='o', linestyle='-', alpha=0.7, markersize=4, color=color, label=label)

    axes[0].set_ylabel('RSSI (dB)')
    axes[0].set_title('RSSI per Tag', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Phase (rad)')
    axes[1].set_title('Corrected Phase per Tag', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for tag_id in sorted(tag_ids):
        color = colors[tag_id % len(colors)]
        tag_bursts = [b for b in processor.capture.bursts if b.tag_id == tag_id]
        
        if not tag_bursts:
            continue

        peak_times = []
        peak_values = []

        for burst in tag_bursts:
            start = burst.start_index
            end = burst.end_index
            segment = processor.capture.second_derivative[start:end]
            if len(segment) == 0:
                continue
            peak_idx = np.argmax(segment)
            peak_times.append((start + peak_idx) / sample_rate)
            peak_values.append(segment[peak_idx])

        #filter away too low values

        


        axes[2].plot(peak_times, peak_values, 'o', markersize=3, 
                    color=color, label=f'Tag {tag_id}')
    plt.tight_layout()
    plt.show()


def plot_capture(signal_obj, output_file=None, event_lines=None, reader_events=None):
    """Plot a capture from an RFIDSignal object."""
    samples = signal_obj.samples
    sample_rate = signal_obj.sample_rate
    center_freq = signal_obj.center_freq

    print("Generating plots...")
    t = np.arange(len(samples)) / sample_rate
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # non_zero_indices = np.where(np.abs(samples) > 1e-6)[0]
    # if len(non_zero_indices) > 0:
    #     start_idx = max(0, non_zero_indices[0] - int(0.01 * sample_rate))
    #     end_idx = min(len(samples), non_zero_indices[-1] + int(0.01 * sample_rate))
    #     samples = samples[start_idx:end_idx]
    #     t = t[start_idx:end_idx]

    t = np.arange(len(samples)) / sample_rate

    axes[0].plot(t, np.abs(samples), linewidth=0.5, color='green')
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

    # axes[1].plot(t, np.abs(signal_obj.envelope), linewidth=0.5, color='blue')
    # axes[1].set_title('Preprocessed I/Q Envelope', fontweight='bold')
    # axes[1].set_xlabel('Time (s)')
    # axes[1].set_ylabel('Envelope')
    # axes[1].grid(True, alpha=0.3)
    

    axes[1].plot(t[2:], signal_obj.second_derivative, linewidth=0.5, color='orange')
    axes[1].set_title('Second Derivative', fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('dx/dt')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {output_file}")


    from scipy.ndimage import uniform_filter1d

    #uniform moving average filter
    window_samples = int(3e-5 * sample_rate)  # 50us window
    envelope = np.abs(samples)
    moving_avg = rfid.uniform_filter(envelope, window_samples)
    normalized = envelope - moving_avg  # signal oscillates around zero

    #do fft, then set all frequencies above 10khz to zero, then do inverse fft
    fft = np.fft.fft(envelope)
    freqs = np.fft.fftfreq(len(envelope), d=1/sample_rate)
    fft[np.abs(freqs) > 100e3] = 0
    normalized = np.fft.ifft(fft)

    axes[2].plot(envelope-normalized, linewidth=0.5, color='purple')
    axes[2].set_title('Moving Average of First EPC Burst', fontweight='bold')
    plt.show()



def plot_reader(reader_session):
    """plot the phase and rssi of reader events over time 
    with different colors for different tags if tag IDs are available in the reader events."""

    if not reader_session.events:
        print("No reader events to plot")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    unique_epcs = sorted(set(e.epc for e in reader_session.events))
    
    for i, epc in enumerate(unique_epcs):
        color = colors[i % len(colors)]
        events = [e for e in reader_session.events if e.epc == epc]
        
        times = np.array([e.timestamp for e in events])
        phases = np.array([e.phase for e in events])
        rssis = np.array([e.rssi for e in events])

        phases = rfid.custom_phase_unwrap(phases)

        
        axes[0].plot(times, phases, 'o-', alpha=0.7, markersize=4, color=color, label=f'Tag {epc}')
        axes[1].plot(times, rssis, 'o-', alpha=0.7, markersize=4, color=color, label=f'Tag {epc}')

    axes[0].set_ylabel('Phase (rad)')
    axes[0].set_title('Reader-Reported Phase', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('RSSI (dB)')
    axes[1].set_title('Reader-Reported RSSI', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


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
    parser.add_argument('--r', action='store_true')
    parser.add_argument('--no-match', action='store_true')

    args = parser.parse_args()

    # Load
    signal = RFIDSignal.from_file(args.input)
    reader_session = ReaderSession.from_files(args.input)

    # Align reader event timestamps to capture
    for event in reader_session.events:
        event.offset = event.timestamp - signal.start_time

    processor = RFIDProcessor(signal)
    
    processor.sliding_window_peak_detection()  # fixed capitalisation
    processor.extract_measurements()           # no longer returns anything meaningful
    if(args.r):
        processor.assign_tag_ids_from_reader(reader_session)  # assign tag IDs based on reader events
    else:
        if args.no_match:
            matches = processor.decode_epcs_without_shifting(IDEAL_EPCs)  # pass known EPCs for matching        
        else:
            matches = processor.decode_epcs(IDEAL_EPCs)
        processor.assign_tag_ids(matches)          # assign tag IDs to matched bursts        

                
        print(f"matches found: {len(matches)}")

    

    if args.plot:
        event_lines = {}
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        tag_ids = set(b.tag_id for b in signal.bursts if b.tag_id >= 0)
        
        for i, tag_id in enumerate(sorted(tag_ids)):
            color = colors[i % len(colors)]
            event_lines[f'rn16_tag_{tag_id}'] = {
                'indices': [idx for b in signal.bursts if b.burst_type == 'rn16' and b.tag_id == tag_id for idx in [b.start_index, b.end_index]],
                'color': color, 'linestyle': '--', 'label': f'RN16 Tag {tag_id}'
            }
            event_lines[f'epc_tag_{tag_id}'] = {
                'indices': [idx for b in signal.bursts if b.burst_type == 'epc' and b.tag_id == tag_id for idx in [b.start_index, b.end_index]],
                'color': color, 'linestyle': '-', 'label': f'EPC Tag {tag_id}'
            }
        
        # Add unassigned bursts
        unassigned_rn16 = [idx for b in signal.bursts if b.burst_type == 'rn16' and b.tag_id < 0 for idx in [b.start_index, b.end_index]]
        if unassigned_rn16:
            event_lines['rn16_unassigned'] = {
                'indices': unassigned_rn16,
                'color': 'green', 'linestyle': '--', 'label': 'RN16 Unassigned'
            }
        
        unassigned_epc = [idx for b in signal.bursts if b.burst_type == 'epc' and b.tag_id < 0 for idx in [b.start_index, b.end_index]]
        if unassigned_epc:
            event_lines['epc_unassigned'] = {
                'indices': unassigned_epc,
                'color': 'red', 'linestyle': '--', 'label': 'EPC Unassigned'
            }

        plot_capture(signal, event_lines=event_lines,
                     reader_events=reader_session.events or None)

    if args.plot_phase:
        plot_phase_rssi_per_tag(processor, reader_session.events or None)

    plot_reader(reader_session)

if __name__ == "__main__":
    main()