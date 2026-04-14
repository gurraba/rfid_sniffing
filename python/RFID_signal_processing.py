"""
Signal preprocessing functions
"""
from tempfile import template

import numpy as np
from scipy import signal

from synthetic_data import generate_fm0_signal

def remove_dc_carrier(iq_data):
    """Remove DC component (carrier at zero frequency)"""
    return iq_data - np.mean(iq_data)

def bandpass_filter(iq_data, sample_rate, low_freq=10e3, high_freq=100e3):  #keep between 10kHz and 100kHz
    """Apply bandpass filter to keep backscatter band"""
    sos = signal.butter(6, [low_freq, high_freq], 'bandpass', 
                       fs=sample_rate, output='sos')
    return signal.sosfilt(sos, iq_data)

def preprocess_rfid_iq(iq_data, sample_rate):
    """Complete preprocessing pipeline"""
    iq_dc_removed = remove_dc_carrier(iq_data)
    #iq_filtered = bandpass_filter(iq_dc_removed, sample_rate)
    #high pass filter to remove low frequency noise
    sos = signal.butter(6, 10e3, 'highpass', fs=sample_rate, output='sos')
    iq_filtered = signal.sosfilt(sos, iq_dc_removed)
    return iq_dc_removed




"""
Frequency correction 
"""

def estimate_frequency(iq_data, sample_rate):
    """Estimate dominant frequency using FFT"""
    n = len(iq_data)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)
    spectrum = np.fft.fft(iq_data)
    print(f"Estimated frequency offset: {freqs[np.argmax(np.abs(spectrum))]:.1f} Hz")
    
    # Find peak in spectrum
    peak_idx = np.argmax(np.abs(spectrum))
    frequency_offset = freqs[peak_idx]
    
    return frequency_offset


def estimate_local_frequency_offset(iq_segment, sample_rate):
    """
    Estimate frequency offset from a short segment
    Uses instantaneous frequency method
    """
    # Calculate instantaneous phase
    phase = np.unwrap(np.angle(iq_segment))
    
    # Instantaneous frequency = derivative of phase
    inst_freq = np.diff(phase) * sample_rate / (2 * np.pi)
    
    # Mean frequency offset
    offset = np.mean(inst_freq)
    
    return offset


def extract_phase_with_local_correction(iq_segment, sample_rate):
    """
    Extract phase after correcting for local frequency offset
    """
    # Estimate offset in this segment
    offset = estimate_local_frequency_offset(iq_segment, sample_rate)
    
    # Correct this segment
    t = np.arange(len(iq_segment)) / sample_rate
    correction = np.exp(-1j * 2 * np.pi * offset * t)
    iq_corrected = iq_segment * correction
    
    # Now extract phase from corrected segment
    weights = np.abs(iq_corrected)
    phase = np.angle(np.sum(iq_corrected * weights) / np.sum(weights))
    
    return phase, offset

def correct_frequency_offset(iq_data, sample_rate, freq_offset):
    """Mix down to baseband to correct frequency offset"""
    n = len(iq_data)
    t = np.arange(n) / sample_rate
    correction_signal = np.exp(-1j * 2 * np.pi * freq_offset * t)
    return iq_data * correction_signal


def derivative_of_magnitude(iq_data):
    """Calculate the derivative of the magnitude to highlight transitions"""
    magnitude = np.abs(iq_data)
    return np.diff(magnitude)



"""
Backscatter burst detection
"""
from scipy.ndimage import uniform_filter1d

# 
# def derivative_burst_detection(iq_data, sample_rate, percentile=95):
#     """Detect bursts using percentile-based threshold"""
#     derivative = derivative_of_magnitude(iq_data)
    
    
#     burst_indices = np.where(derivative > 0.004)[0]
    
#     # Filter close bursts
#     min_distance = int(sample_rate * 0.001)
#     filtered_indices = []
#     if len(burst_indices) > 0:
#         filtered_indices.append(burst_indices[0])
#         for i in range(1, len(burst_indices)):
#             if burst_indices[i] - filtered_indices[-1] >= min_distance:
#                 filtered_indices.append(burst_indices[i])
    
#     print(f"Found {len(filtered_indices)} bursts")
#     return np.array(filtered_indices)


def derivative_burst_detection(iq_data, sample_rate, percentile=99.98):
    """Detect bursts using percentile-based threshold"""
    #derivative = np.diff(iq_data)
    
    
    #find 10 percent peaks
    threshold = np.percentile(iq_data, percentile)
    burst_indices = np.where(iq_data > threshold)[0]

    
    # Filter close bursts
    min_distance = int(sample_rate * 0.001)
    filtered_indices = []
    if len(burst_indices) > 0:
        filtered_indices.append(burst_indices[0])
        for i in range(1, len(burst_indices)):
            if burst_indices[i] - filtered_indices[-1] >= min_distance:
                filtered_indices.append(burst_indices[i])
            print("found at amplitude threshold:", iq_data[burst_indices[0]])
    print(f"Found {len(filtered_indices)} bursts")
    return np.array(filtered_indices)




def magnitude_burst_detection(iq_data, sample_rate, threshold_factor=0.01):
    """
    Detect energy bursts (tag responses)
    Returns: list of (start_idx, end_idx, peak_power)
    """
    envelope = np.abs(iq_data)
    
    window_size = int(sample_rate * 0.001)  # 0.1ms window
    smoothed = uniform_filter1d(envelope, window_size)
    
    noise_floor = np.median(smoothed)
    threshold = noise_floor * threshold_factor
    
    above_threshold = smoothed > threshold
    
    bursts = []
    in_burst = False
    start_idx = 0
    
    for i, val in enumerate(above_threshold):
        if val and not in_burst:
            start_idx = i
            in_burst = True
        elif not val and in_burst:
            end_idx = i
            peak_power = np.max(smoothed[start_idx:end_idx])
            bursts.append((start_idx, end_idx, peak_power))
            in_burst = False
    
    return bursts



def generate_fm0_preamble_derivative(sample_rate, bit_rate=640e3):
    """
    Generate the expected derivative pattern for FM0 preamble detection
    FM0 preamble: specific transition pattern
    """
    samples_per_bit = int(sample_rate / bit_rate)
    total_samples = 6 * samples_per_bit  # 6 bits
    
    # Generate FM0 signal with proper phase transitions
    signal = np.zeros(total_samples, dtype=complex)
    phase = 0  # Start with phase 0
    
    sample_idx = 0
    for bit_idx, bit in enumerate([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]):  # -1 indicates violation
        if bit == -1:
            # Violation: no transition where there should be one
            signal[sample_idx:sample_idx + samples_per_bit] = np.exp(1j * phase)
        elif bit == 1:
            # Transition in middle
            half_bit = samples_per_bit // 2
            signal[sample_idx:sample_idx + half_bit] = np.exp(1j * phase)
            phase += np.pi  # 180° transition
            signal[sample_idx + half_bit:sample_idx + samples_per_bit] = np.exp(1j * phase)
        else:  # bit == 0
            # No transition
            signal[sample_idx:sample_idx + samples_per_bit] = np.exp(1j * phase)
        
        sample_idx += samples_per_bit
    
    # Return the derivative (what we correlate with)
    return np.diff(signal)


def detect_preamble_fm0(iq_data, sample_rate, bit_rate=640e3):
    """
    Find FM0 preambles using matched filter on derivative
    """
    # Generate expected preamble derivative pattern
    preamble_derivative = generate_fm0_preamble_derivative(sample_rate, bit_rate)
    
    # Get derivative of received signal
    received_derivative = np.diff(iq_data, 2)
    received_derivative = np.abs(received_derivative)
    
    # Correlate the complex derivatives
    correlation = np.convolve(np.abs(preamble_derivative), received_derivative, mode='same')
    
    # Take magnitude for peak detection (since phase might vary)
    correlation_magnitude = np.abs(correlation)
    
    # Find peaks in correlation
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(correlation_magnitude, 
                                   height=np.max(correlation_magnitude) * 0.3,  # Adjustable threshold
                                   distance=int(sample_rate * 0.0005))  # Min 0.5ms apart
    
    print(f"Matched filter correlation peaks: {len(peaks)} found")
    print(f"Max correlation magnitude: {np.max(correlation_magnitude):.4f}")
    
    # Adjust indices back (since we used 'valid' mode)
    #adjusted_peaks = peaks + len(preamble_derivative) - 1
    
    return peaks



def generate_periodic_pulse_template(sample_rate, bit_rate=640e3, num_pulses=20):
    """
    Generate a simple periodic pulse train at the bit rate
    Matches any FM0-like periodic modulation
    """
    samples_per_bit = int(sample_rate / bit_rate)
    total_samples = num_pulses * samples_per_bit
    
    # Create pulse train - sharp transitions every bit period
    template = np.zeros(total_samples)
    
    for i in range(num_pulses):
        pulse_idx = i * samples_per_bit
        # Sharp pulse (1-2 samples wide)
        template[pulse_idx] = 1.0
        if pulse_idx + 1 < len(template):
            template[pulse_idx + 1] = -1.0  # Create sharp transition

    return np.abs(template)


def detect_fm0_periodic(iq_data, sample_rate, bit_rate=640e3):
    """
    Detect FM0 by looking for periodic 640kHz transitions
    More robust than specific preamble matching
    """
    # Generate periodic pulse template
    pulse_template = generate_periodic_pulse_template(sample_rate, bit_rate, num_pulses=30)
    
    # Second derivative emphasizes sharp transitions
    received_derivative = np.abs(np.diff(iq_data, 2))
    
    # Normalize both to avoid amplitude issues
    pulse_template = pulse_template / np.max(np.abs(pulse_template))
    received_derivative = received_derivative / (np.max(received_derivative) + 1e-10)
    
    # Correlate
    correlation = np.convolve(received_derivative, pulse_template, mode='same')
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.abs(correlation),
                         height=np.max(np.abs(correlation)) * 0.3,
                         distance=int(sample_rate * 0.001))
    
    print(f"Periodic pattern detector: {len(peaks)} found")
    
    return peaks

    


"""
Phase and RSSI extraction
"""

def extract_phase_old(iq_segment):
    """Extract phase measurement from I/Q segment"""
    weights = np.abs(iq_segment)
    weighted_phase = np.angle(np.sum(iq_segment * weights) / np.sum(weights))
    return weighted_phase

def extract_phase(iq_segment):
    """Extract phase measurement from I/Q segment using mean angle"""
    return np.angle(np.mean(iq_segment))

def extract_rssi(iq_segment):
    """Extract RSSI in dB"""
    power_linear = np.mean(np.abs(iq_segment)**2)
    rssi_db = 10 * np.log10(power_linear)
    return rssi_db

def extract_measurements(iq_segment):
    """Extract both phase and RSSI"""
    return {
        'phase': extract_phase(iq_segment),
        'rssi': extract_rssi(iq_segment)
    }



def custom_phase_unwrap(phi):
    """Correct large 180° phase jumps, similar to the provided MATLAB logic."""
    phi = np.asarray(phi, dtype=float)
    if len(phi) == 0:
        return phi

    phi_corrected = np.copy(phi)
    prev_phase = phi_corrected[0]

    for i in range(1, len(phi_corrected)):
        phase_diff = phi[i] - prev_phase

        if abs(phase_diff) > np.pi / 2:
            if phase_diff > 0:
                phi_corrected[i] = phi[i] - np.pi
            else:
                phi_corrected[i] = phi[i] + np.pi
        else:
            phi_corrected[i] = phi[i]

        prev_phase = phi_corrected[i]

    return phi_corrected
