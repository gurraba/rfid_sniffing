"""
FM Radio Receiver - Test USRP + Antenna
Tunes to an FM radio station and saves audio to MP3
"""

import uhd
import numpy as np
import scipy.signal as signal
from datetime import datetime
import time

def fm_demodulate(iq_samples, sample_rate):
    """
    Demodulate FM signal from I/Q samples
    """
    # Calculate instantaneous phase
    phase = np.unwrap(np.angle(iq_samples))
    
    # Derivative of phase is frequency (FM demodulation)
    audio = np.diff(phase)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio

def setup_usrp_for_fm(usrp, fm_freq=100.7e6):
    """
    Configure USRP for FM radio reception
    FM broadcast is ~88-108 MHz
    """
    sample_rate = 2.4e6  # 2.4 MS/s (gives us enough bandwidth for FM)
    gain = 40
    
    print(f"\nConfiguring for FM Radio:")
    print(f"  Station: {fm_freq/1e6:.1f} MHz")
    print(f"  Sample Rate: {sample_rate/1e6} MS/s")
    print(f"  Gain: {gain} dB")
    
    usrp.set_rx_rate(sample_rate)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(fm_freq))
    usrp.set_rx_gain(gain)
    
    time.sleep(0.1)
    
    return sample_rate

def capture_fm(usrp, sample_rate, duration=10):
    """
    Capture FM signal
    """
    print(f"\nCapturing {duration} seconds of FM radio...")
    
    # Setup streaming
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_streamer = usrp.get_rx_stream(stream_args)
    
    # Calculate samples needed
    num_samples = int(sample_rate * duration)
    samples = np.zeros(num_samples, dtype=np.complex64)
    
    # Start stream
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samples
    stream_cmd.stream_now = True
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    # Receive
    metadata = uhd.types.RXMetadata()
    recv_buffer = np.zeros(rx_streamer.get_max_num_samps(), dtype=np.complex64)
    
    samples_received = 0
    start_time = time.time()
    
    while samples_received < num_samples:
        num_rx = rx_streamer.recv(recv_buffer, metadata)
        
        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
            print(f"Error: {metadata.strerror()}")
            break
        
        samples[samples_received:samples_received + num_rx] = recv_buffer[:num_rx]
        samples_received += num_rx
        
        # Progress indicator
        progress = samples_received / num_samples * 100
        if int(progress) % 10 == 0:
            print(f"  Progress: {progress:.0f}%", end='\r')
    
    elapsed = time.time() - start_time
    print(f"\n✓ Captured {samples_received:,} samples in {elapsed:.1f}s")
    
    # Check signal strength
    power_db = 10 * np.log10(np.mean(np.abs(samples)**2))
    print(f"Signal power: {power_db:.1f} dB")
    
    if power_db < -60:
        print("WARNING: Signal very weak - check antenna connection!")
    elif power_db > -40:
        print("Good signal strength")
    
    return samples

def process_to_audio(samples, sample_rate):
    """
    Demodulate and process FM to audio
    """
    print("\nProcessing FM signal to audio...")
    
    # Demodulate
    audio = fm_demodulate(samples, sample_rate)
    
    # Decimate to audio sample rate (48 kHz)
    audio_sample_rate = 48000
    decimation_factor = int(sample_rate / audio_sample_rate)
    
    print(f"  Decimating by factor of {decimation_factor}...")
    audio_resampled = signal.decimate(audio, decimation_factor)
    
    # Apply audio filtering (remove DC, limit bandwidth)
    print(f"  Applying audio filters...")
    audio_filtered = signal.sosfilt(
        signal.butter(6, [50, 15000], 'bandpass', fs=audio_sample_rate, output='sos'),
        audio_resampled
    )
    
    # De-emphasis filter (FM broadcast uses pre-emphasis)
    tau = 50e-6  # 50 microseconds for Europe
    d = np.exp(-1 / (audio_sample_rate * tau))
    audio_deemph = signal.lfilter([1 - d], [1, -d], audio_filtered)
    
    # Normalize
    audio_final = audio_deemph / np.max(np.abs(audio_deemph)) * 0.9
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_final * 32767).astype(np.int16)
    
    print(f"✓ Audio processed: {len(audio_int16)/audio_sample_rate:.1f} seconds")
    
    return audio_int16, audio_sample_rate

def save_as_wav(audio, sample_rate, filename):
    """
    Save audio as WAV file (simpler than MP3, no extra dependencies)
    """
    from scipy.io import wavfile
    
    wavfile.write(filename, sample_rate, audio)
    print(f"✓ Saved to: {filename}")

def main():
    """
    FM Radio Test
    """
    print("="*60)
    print("FM Radio Receiver - Antenna Test")
    print("="*60)
    
    # Get FM frequency from user
    print("\nCommon FM stations in Gothenburg:")
    print("  Mix Megapol: 106.7 MHz")
    print("  P3: 103.3 MHz") 
    print("  P4: 88.9 MHz")
    print("  P1: 92.4 MHz")
    
    freq_input = input("\nEnter FM frequency in MHz (or press Enter for 106.7): ").strip()
    
    if freq_input:
        fm_freq = float(freq_input) * 1e6
    else:
        fm_freq = 106.7e6  # Default
    
    duration_input = input("Duration in seconds (or press Enter for 10): ").strip()
    
    if duration_input:
        duration = int(duration_input)
    else:
        duration = 10
    
    try:
        # Connect to USRP
        print("\nConnecting to USRP...")
        usrp = uhd.usrp.MultiUSRP()
        print(f"✓ Connected: {usrp.get_pp_string()}")
        
        # Configure
        sample_rate = setup_usrp_for_fm(usrp, fm_freq)
        
        # Capture
        samples = capture_fm(usrp, sample_rate, duration)
        
        # Process to audio
        audio, audio_rate = process_to_audio(samples, sample_rate)
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw I/Q
        iq_filename = f'fm_iq_{fm_freq/1e6:.1f}MHz_{timestamp}.npy'
        np.save(iq_filename, samples)
        print(f"\n✓ Raw I/Q saved to: {iq_filename}")
        
        # Save audio as WAV
        wav_filename = f'fm_audio_{fm_freq/1e6:.1f}MHz_{timestamp}.wav'
        save_as_wav(audio, audio_rate, wav_filename)
        
        print("\n" + "="*60)
        print("✓ Test complete!")
        print(f"Listen to the WAV file to verify your antenna works")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()