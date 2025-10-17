"""
Audio Processor Module - Faster-Whisper Version
Handles real-time audio capture and transcription using Faster-Whisper (local).

Key Components:
1. Audio capture from microphone (PyAudio)
2. Real-time transcription (Faster-Whisper - runs locally)
3. Threaded processing for non-blocking operation
"""

# Suppress ONNX Runtime GPU discovery warnings
import os
os.environ["ORT_DISABLE_DEVICE_DISCOVERY"] = "1"

import pyaudio
import threading
import queue
import numpy as np
from faster_whisper import WhisperModel
import time
from typing import Callable, Optional

# ============================================================================
# CONFIGURATION - Adjust these based on your needs
# ============================================================================

# Audio recording settings
CHUNK_SIZE = 8192              # Audio buffer size (samples per read)
FORMAT = pyaudio.paInt16       # 16-bit audio
CHANNELS = 1                   # Mono audio (sufficient for speech)
RATE = 16000                   # 16kHz sample rate (Whisper optimal)

# Transcription chunk settings
CHUNK_DURATION_SECONDS = 3     # Process audio every 3 seconds (faster updates)
OVERLAP_SECONDS = 1.5          # 1.5-second overlap (50% for better continuity)

# Whisper model settings
WHISPER_MODEL_SIZE = "base"    # Options: tiny, base, small, medium, large
                               # tiny = fastest, least accurate
                               # base = good balance (RECOMMENDED)
                               # small = more accurate, slower
                               # medium/large = very accurate, very slow
DEVICE = "cpu"                 # Use "cuda" if you have GPU
COMPUTE_TYPE = "int8"          # int8 = faster, float16 = more accurate (needs GPU)

# ============================================================================


class AudioProcessor:
    """
    Handles real-time audio recording and transcription.
    
    Architecture:
    1. Main thread: Captures audio from microphone continuously
    2. Processing thread: Transcribes audio chunks in background
    3. Queue: Connects capture → processing without blocking
    """
    
    def __init__(self, on_transcript: Callable[[str], None]):

        # Audio recording components
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # Threading components
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcript_callback = on_transcript
        
        # Audio buffer for accumulating chunks with overlap
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        # Initialize Whisper model (runs once at startup)
        print("Loading Whisper model... (this may take 10-30 seconds)")
        self.whisper_model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE
        )
        print("✓ Whisper model loaded successfully")
        
        # Processing thread
        self.processing_thread: Optional[threading.Thread] = None
    
    def start_recording(self):
        """Start recording audio from microphone."""
        if self.is_recording:
            print("Already recording!")
            return
        
        self.is_recording = True
        self.audio_buffer = []
        self.buffer_duration = 0.0
        
        # Find PipeWire/default device (Linux audio routing fix)
        device_index = None
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                # Prefer 'pipewire' or 'default' device
                if 'pipewire' in info['name'].lower() or 'default' in info['name'].lower():
                    device_index = i
                    print(f"✓ Using audio device: {info['name']}")
                    break
        
        # If no PipeWire found, try default
        if device_index is None:
            try:
                device_index = self.audio.get_default_input_device_info()['index']
                print(f"✓ Using default audio device")
            except:
                # Fallback to device index 11 (pipewire from your list)
                device_index = 11
                print(f"✓ Using fallback device index: {device_index}")
        
        # Open microphone stream
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,  # Use detected device
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            print("✓ Microphone stream started")
            print("🎤 Speak into your microphone!")
        except Exception as e:
            print(f"✗ Error opening microphone: {e}")
            print("Tip: Make sure no other app is using your microphone")
            self.is_recording = False
            return
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        print("✓ Processing thread started")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Called automatically by PyAudio for each audio chunk."""
        if status:
            # Only print if there's an actual error (not just info)
            pass
        
        # Add audio to queue for processing
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _processing_loop(self):
        """Background thread that processes queued audio."""
        print("Processing loop started")
        
        while self.is_recording:
            try:
                # Get audio from queue (wait up to 0.5 seconds)
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Convert raw bytes to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Add to buffer
                self.audio_buffer.extend(audio_np)
                self.buffer_duration += len(audio_np) / RATE
                
                # Check if we have enough audio to process
                if self.buffer_duration >= CHUNK_DURATION_SECONDS:
                    self._transcribe_buffer()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
    
    def _transcribe_buffer(self):
        """Transcribe the accumulated audio buffer using Whisper."""
        try:
            # Convert buffer to float32 numpy array (Whisper requirement)
            audio_np = np.array(self.audio_buffer, dtype=np.float32)
            
            # Normalize audio to [-1.0, 1.0] range
            audio_np = audio_np / 32768.0
            
            # Transcribe with Whisper
            segments, info = self.whisper_model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500
                )
            )
            
            # Combine all segments into one transcript
            transcript_text = ""
            for segment in segments:
                transcript_text += segment.text + " "
            
            # Only send non-empty transcripts
            transcript_text = transcript_text.strip()
            if transcript_text:
                print(f"✓ Transcribed: {transcript_text}")
                self.transcript_callback(transcript_text)
            
            # Keep overlap for next chunk
            overlap_samples = int(OVERLAP_SECONDS * RATE)
            
            if len(self.audio_buffer) > overlap_samples:
                self.audio_buffer = self.audio_buffer[-overlap_samples:]
                self.buffer_duration = OVERLAP_SECONDS
            else:
                self.audio_buffer = []
                self.buffer_duration = 0.0
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_recording(self):
        """Stop recording and clean up resources."""
        print("Stopping recording...")
        self.is_recording = False
        
        # Close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("✓ Audio stream closed")
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            print("✓ Processing thread stopped")
        
        # Process any remaining audio
        if self.buffer_duration > 1.0:
            print("Processing remaining audio...")
            self._transcribe_buffer()
        
        print("✓ Recording stopped")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'audio'):
            self.audio.terminate()


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    """Test the audio processor independently."""
    
    def on_transcript(text):
        print(f"\n{'='*60}")
        print(f"[TRANSCRIPT] {text}")
        print('='*60 + "\n")
    
    processor = AudioProcessor(on_transcript)
    
    print("\n" + "="*60)
    print("AUDIO PROCESSOR TEST")
    print("="*60)
    print("Starting 30-second recording test...")
    print("Speak into your microphone!")
    print("="*60 + "\n")
    
    processor.start_recording()
    time.sleep(30)  # Record for 30 seconds
    processor.stop_recording()
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)