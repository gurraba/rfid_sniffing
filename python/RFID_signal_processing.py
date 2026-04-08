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
    derivative = derivative_of_magnitude(iq_data)
    
    
    #find 10 percent peaks
    threshold = np.percentile(derivative, percentile)
    burst_indices = np.where(derivative > threshold)[0]

    
    # Filter close bursts
    min_distance = int(sample_rate * 0.001)
    filtered_indices = []
    if len(burst_indices) > 0:
        filtered_indices.append(burst_indices[0])
        for i in range(1, len(burst_indices)):
            if burst_indices[i] - filtered_indices[-1] >= min_distance:
                filtered_indices.append(burst_indices[i])
    
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



def detect_preamble_fm0(iq_data, sample_rate, bit_rate=640e3):
    """
    Find FM0 preambles using matched filter
    """
    bits = [1, 0, 1, 0, -1, 1]  # -1 = violation (no transition)
    
    # Convert to amplitude modulation
    preamble = generate_fm0_signal(bits, sample_rate, bit_rate)
    
    # Get envelope (amplitude)
    envelope = np.abs(iq_data)
    
    # Correlate
    correlation = np.correlate(envelope, preamble, mode='same')
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(correlation, 
                         height=np.max(correlation) * 0.95,
                         distance=int(sample_rate * 0.0001))  # Min 10ms apart
    print(f"Correlation peaks: {len(peaks)} found")
    return peaks  # Indices where preambles detected
    
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
