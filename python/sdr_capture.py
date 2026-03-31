"""
capture.py - Capture I/Q from USRP and save
"""

import uhd
import numpy as np
import json
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CAPTURES_DIR = DATA_DIR / "captures"
RESULTS_DIR = DATA_DIR / "results"

CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = "C:/Users/gusta/Documents/programmering/RFID_project/data/captures"

class UsrpCapture:
    """Manages USRP connection and capture"""
    
    def __init__(self, freq, sample_rate, gain):
        self.freq = freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.usrp = None
    
    def connect(self):
        """Initialize USRP hardware"""
        print(f"Initializing USRP at {self.freq/1e6} MHz...")
        self.usrp = uhd.usrp.MultiUSRP()
        self.usrp.set_rx_rate(self.sample_rate)
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.freq))
        self.usrp.set_rx_gain(self.gain)
        print("USRP ready")
    
    def capture(self, duration):
        """Capture I/Q samples"""
        num_samples = int(self.sample_rate * duration)
        print(f"Capturing {num_samples:,} samples...")
        
        # ... actual capture code ...
        samples = self._do_capture(num_samples)
        
        return samples
    
    def _do_capture(self, num_samples):
    
        # Setup streaming
        stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_streamer = self.usrp.get_rx_stream(stream_args)
        
        # Prepare buffer
        samples = np.zeros(num_samples, dtype=np.complex64)
        

        # Start stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = num_samples
        stream_cmd.stream_now = True
        rx_streamer.issue_stream_cmd(stream_cmd)
        
        # Receive samples
        metadata = uhd.types.RXMetadata()
        recv_buffer = np.zeros(rx_streamer.get_max_num_samps(), dtype=np.complex64)
        
        samples_received = 0
        #time in UTC
    
        while samples_received < num_samples:
            num_rx = rx_streamer.recv(recv_buffer, metadata)
            
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"Error during receive: {metadata.strerror()}")
                break
            
            samples[samples_received:samples_received + num_rx] = recv_buffer[:num_rx]
            samples_received += num_rx
        
     
    
        print(f"  Average power: {10*np.log10(np.mean(np.abs(samples)**2)):.1f} dB")
        print(f"  Max power: {10*np.log10(np.max(np.abs(samples)**2)):.1f} dB")
        print("=" * 60)
        
        return samples
            






def save_capture(output_prefix, samples, metadata):
    """Save I/Q and metadata to files"""
    
    # Save raw I/Q
    iq_file = f"{DATA_DIR}/{output_prefix}_iq.npy"
    np.save(iq_file, samples)
    print(f"Saved I/Q: {iq_file}")
    
    # Save metadata
    meta_file = f"{DATA_DIR}/{output_prefix}_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_file}")
    
    return iq_file, meta_file


def create_metadata(freq, sample_rate, gain, duration, start_time):
    """Create metadata dictionary"""
    return {
        'center_freq': freq,
        'sample_rate': sample_rate,
        'gain': gain,
        'duration': duration,
        'start_time': start_time.timestamp(),
        'num_samples': int(sample_rate * duration)
    }


def main():
    parser = argparse.ArgumentParser(description='Capture RFID signals')
    parser.add_argument('--freq', type=float, default=865.7e6)
    parser.add_argument('--rate', type=float, default=5e6)
    parser.add_argument('--gain', type=float, default=0)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--output', required=True, help="Output prefix for saved files")
    
    args = parser.parse_args()
    
    # Create capturer
    capturer = UsrpCapture(args.freq, args.rate, args.gain)
    capturer.connect()
    
    # Wait for input
    wait = input("Press ENTER to start")
    #dtime.sleep(10)  # Short delay to ensure USRP is ready
    # Capture
    start_time = datetime.now(timezone.utc)
    samples = capturer.capture(args.duration)
    
    # Save
    metadata = create_metadata(args.freq, args.rate, args.gain, 
                               args.duration, start_time)
    save_capture(args.output, samples, metadata)
    
    print(f"\nCapture complete!")
    print(f"Captured {len(samples):,} samples")


if __name__ == "__main__":
    main()