"""
Audio Processor Module - OpenAI Realtime API Version
Handles real-time audio capture and transcription using OpenAI's Realtime API.

Key Components:
1. Audio capture from microphone (PyAudio)
2. Real-time transcription (OpenAI Realtime API with gpt-4o-transcribe)
3. WebSocket connection for streaming
4. Built-in VAD and noise reduction
"""

import pyaudio
import threading
import queue
import numpy as np
import time
import json
import base64
import asyncio
import websockets
from typing import Callable, Optional
import requests
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# CONFIGURATION - Adjust these based on your needs
# ============================================================================

# Audio recording settings
CHUNK_SIZE = 4096              # Audio buffer size (samples per read)
FORMAT = pyaudio.paInt16       # 16-bit audio
CHANNELS = 1                   # Mono audio
RATE = 24000                   # 24kHz sample rate (OpenAI Realtime API optimal)

# OpenAI Realtime API settings
OPENAI_API_URL = "https://api.openai.com/v1/realtime/transcription_sessions"
OPENAI_WS_URL = "wss://api.openai.com/v1/realtime"

# Transcription model
TRANSCRIPTION_MODEL = "gpt-4o-transcribe"  # Options:
                                           # - gpt-4o-transcribe (BEST, most accurate)
                                           # - gpt-4o-mini-transcribe (faster, cheaper)
                                           # - whisper-1 (legacy)

# VAD (Voice Activity Detection) settings
VAD_THRESHOLD = 0.5            # 0.0 to 1.0 - higher = less sensitive
                               # 0.3 = very sensitive (picks up quiet speech)
                               # 0.5 = balanced (RECOMMENDED)
                               # 0.7 = less sensitive (only clear speech)

VAD_PREFIX_PADDING_MS = 300    # Audio to include before speech starts (ms)
VAD_SILENCE_DURATION_MS = 700  # Silence duration before considering speech ended (ms)
                               # Lower = more responsive, but may cut off pauses
                               # Higher = more complete sentences, but slower updates

# Noise reduction
NOISE_REDUCTION_TYPE = "near_field"  # Options:
                                     # - "near_field" = close microphone (desk/headset)
                                     # - "far_field" = room microphone (conference room)

# ============================================================================


class AudioProcessor:
    """
    Handles real-time audio recording and transcription using OpenAI Realtime API.
    
    Architecture:
    1. Main thread: Captures audio from microphone continuously
    2. WebSocket thread: Maintains connection to OpenAI and sends audio
    3. Async event loop: Handles incoming transcription events
    4. Callback: Sends transcripts back to main application
    
    OpenAI Realtime API Flow:
    1. Get ephemeral token from REST API
    2. Connect WebSocket with token
    3. Configure transcription session (model, VAD, noise reduction)
    4. Stream audio as base64-encoded PCM16
    5. Receive transcription events (speech_started, committed, completed)
    """
    
    def __init__(self, on_transcript: Callable[[str], None], api_key_openai: str):
        """
        Initialize audio processor with OpenAI Realtime API.
        
        Args:
            on_transcript: Callback function called when new transcript is ready
                          Example: on_transcript("Hello world")
            api_key: OpenAI API key from https://platform.openai.com/api-keys
        """
        self.api_key = api_key_openai
        self.transcript_callback = on_transcript
        
        # Audio recording components
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # Threading components
        self.is_recording = False
        self.audio_queue = queue.Queue()      # Thread-safe queue for audio data
        
        # WebSocket components
        self.websocket = None
        self.client_secret = None
        self.ws_thread: Optional[threading.Thread] = None
        self.event_loop = None
        
        print("✓ Audio processor initialized with OpenAI Realtime API")
    
    def start_recording(self):
        """
        Start recording audio and connect to OpenAI Realtime API.
        
        Steps:
        1. Get ephemeral token from OpenAI REST API
        2. Open WebSocket connection
        3. Configure transcription session
        4. Start audio capture
        5. Begin streaming audio to OpenAI
        """
        if self.is_recording:
            print("Already recording!")
            return
        
        # Step 1: Get ephemeral token
        # The ephemeral token is a temporary credential for WebSocket auth
        # It's safer than using your API key directly in WebSocket
        print("Getting ephemeral token from OpenAI...")
        try:
            response = requests.post(
                OPENAI_API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={}  # Empty body
            )
            
            if response.status_code != 200:
                print(f"✗ Error getting token: {response.status_code}")
                print(f"Response: {response.text}")
                return
            
            data = response.json()
            self.client_secret = data.get("client_secret")
            
            if not self.client_secret:
                print("✗ No client_secret in response")
                return
            
            print("✓ Ephemeral token obtained")
            
        except Exception as e:
            print(f"✗ Error getting ephemeral token: {e}")
            return
        
        # Step 2-5: Start WebSocket connection and audio capture
        self.is_recording = True
        
        # Start WebSocket in separate thread with its own event loop
        self.ws_thread = threading.Thread(
            target=self._run_websocket_loop,
            daemon=True
        )
        self.ws_thread.start()
        
        # Wait a moment for WebSocket to connect
        time.sleep(1)
        
        # Open microphone stream
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            print("✓ Microphone stream started")
            print("🎤 Speak into your microphone - transcription will appear in real-time!")
            
        except Exception as e:
            print(f"✗ Error opening microphone: {e}")
            print("Make sure your microphone is connected and not used by another app")
            self.is_recording = False
            return
    
    def _run_websocket_loop(self):
        """
        Run WebSocket connection in separate thread with its own event loop.
        
        Why separate event loop?
        - asyncio requires an event loop to run async code
        - We can't block the main thread with WebSocket operations
        - This creates a dedicated thread for all WebSocket communication
        """
        # Create new event loop for this thread
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        
        # Run WebSocket connection
        try:
            self.event_loop.run_until_complete(self._websocket_handler())
        except Exception as e:
            print(f"WebSocket error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.event_loop.close()
    
    async def _websocket_handler(self):
        """
        Main WebSocket connection handler (async).
        
        This function:
        1. Connects to OpenAI WebSocket
        2. Configures transcription session
        3. Sends audio continuously
        4. Receives and processes transcription events
        """
        # Build WebSocket URL with token
        # ws_url = f"{OPENAI_WS_URL}?intent=transcription&token={self.client_secret}"
        
        # try:
        #     # Connect to OpenAI WebSocket
        #     # websockets.connect() returns an async context manager
        #     async with websockets.connect(
        #         ws_url,
        #         extra_headers={
        #             "Authorization": f"Bearer {self.api_key}",
        #             "OpenAI-Beta": "realtime=v1"
        #         }
        #     ) as websocket:

        ws_url = f"{OPENAI_WS_URL}?intent=transcription"
        
        try:
            # Connect to OpenAI WebSocket
            # websockets.connect() returns an async context manager
            async with websockets.connect(
                ws_url,
                extra_headers={
                    # Use the ephemeral client_secret here:
                    "Authorization": f"Bearer {self.client_secret}",
                    "OpenAI-Beta": "realtime=v1"
                }
            ) as websocket:
                self.websocket = websocket
                print("✓ WebSocket connected to OpenAI")
                
                # Configure transcription session
                # This tells OpenAI how to process our audio
                config = {
                    "type": "transcription_session.update",
                    "input_audio_format": "pcm16",  # 16-bit PCM audio
                    "input_audio_transcription": {
                        "model": TRANSCRIPTION_MODEL,
                        "language": "en"  # Force English (or "" for auto-detect)
                    },
                    "turn_detection": {
                        "type": "server_vad",  # Server-side Voice Activity Detection
                        "threshold": VAD_THRESHOLD,
                        "prefix_padding_ms": VAD_PREFIX_PADDING_MS,
                        "silence_duration_ms": VAD_SILENCE_DURATION_MS
                    },
                    "input_audio_noise_reduction": {
                        "type": NOISE_REDUCTION_TYPE
                    }
                }
                
                await websocket.send(json.dumps(config))
                print("✓ Transcription session configured")
                
                # Create tasks for sending and receiving
                # These run concurrently (at the same time)
                send_task = asyncio.create_task(self._send_audio_loop(websocket))
                receive_task = asyncio.create_task(self._receive_events_loop(websocket))
                
                # Wait for both tasks (they run until recording stops)
                await asyncio.gather(send_task, receive_task)
        
        except Exception as e:
            print(f"✗ WebSocket error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _send_audio_loop(self, websocket):
        """
        Continuously send audio to OpenAI WebSocket.
        
        How it works:
        1. Get audio chunk from queue (captured by microphone callback)
        2. Convert to base64 string
        3. Send to OpenAI in required JSON format
        4. Repeat until recording stops
        """
        print("Audio send loop started")
        
        while self.is_recording:
            try:
                # Get audio from queue (non-blocking with timeout)
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # No audio available, check again
                    await asyncio.sleep(0.01)
                    continue
                
                # Convert audio bytes to base64 string
                # OpenAI Realtime API requires base64 encoding
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                # Send to OpenAI in required JSON format
                message = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                }
                
                await websocket.send(json.dumps(message))
                
            except Exception as e:
                print(f"Error sending audio: {e}")
                break
        
        print("Audio send loop stopped")
    
    async def _receive_events_loop(self, websocket):
        """
        Continuously receive and process events from OpenAI.
        
        OpenAI Realtime API Events:
        1. input_audio_buffer.speech_started - Speech detected
        2. input_audio_buffer.committed - Audio segment ready for transcription
        3. conversation.item.input_audio_transcription.completed - Transcript ready!
        4. input_audio_buffer.speech_stopped - Speech ended
        """
        print("Event receive loop started")
        
        while self.is_recording:
            try:
                # Wait for message from OpenAI
                message = await websocket.recv()
                
                # Parse JSON event
                event = json.loads(message)
                event_type = event.get("type")
                
                # Handle different event types
                if event_type == "input_audio_buffer.speech_started":
                    print("🎤 Speech detected...")
                
                elif event_type == "input_audio_buffer.committed":
                    # Audio segment committed for transcription
                    item_id = event.get("item_id")
                    print(f"📝 Transcribing segment {item_id}...")
                
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    # THIS IS THE IMPORTANT ONE - we got a transcript!
                    transcript = event.get("transcript", "")
                    
                    if transcript.strip():
                        print(f"✓ Transcribed: {transcript}")
                        
                        # Call callback to send to Streamlit
                        self.transcript_callback(transcript)
                
                elif event_type == "input_audio_buffer.speech_stopped":
                    print("⏸️ Speech ended")
                
                elif event_type == "error":
                    # Handle errors from OpenAI
                    error = event.get("error", {})
                    print(f"❌ OpenAI Error: {error}")
                
                else:
                    # Log other events for debugging
                    if event_type:
                        print(f"📨 Event: {event_type}")
            
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed")
                break
            except Exception as e:
                print(f"Error receiving event: {e}")
                break
        
        print("Event receive loop stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Called automatically by PyAudio for each audio chunk.
        
        This is a CALLBACK function - PyAudio calls it ~60 times per second.
        We just add the audio to a queue for the WebSocket thread to process.
        
        Args:
            in_data: Raw audio bytes from microphone (PCM16 format)
            frame_count: Number of audio frames captured
            time_info: Timing information
            status: Status flags (check for errors)
        
        Returns:
            Tuple: (None, pyaudio.paContinue) to keep recording
        """
        if status:
            print(f"Audio callback status: {status}")
        
        # Add audio to queue for WebSocket to send
        self.audio_queue.put(in_data)
        
        return (None, pyaudio.paContinue)
    
    def stop_recording(self):
        """
        Stop recording and clean up resources.
        
        Steps:
        1. Stop recording flag
        2. Close microphone stream
        3. Close WebSocket connection
        4. Wait for threads to finish
        """
        print("Stopping recording...")
        self.is_recording = False
        
        # Close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("✓ Audio stream closed")
        
        # Close WebSocket
        if self.websocket:
            try:
                # Schedule WebSocket close in its event loop
                if self.event_loop and self.event_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.close(),
                        self.event_loop
                    )
                print("✓ WebSocket closed")
            except:
                pass
        
        # Wait for WebSocket thread
        if self.ws_thread:
            self.ws_thread.join(timeout=3.0)
            print("✓ WebSocket thread stopped")
        
        print("✓ Recording stopped")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'audio'):
            self.audio.terminate()


# ============================================================================
# TESTING CODE - Run this file directly to test
# ============================================================================
from dotenv import load_dotenv
if __name__ == "__main__":
    """
    Test the audio processor independently.
    Run: python audio_processor.py
    
    Make sure to set OPENAI_API_KEY environment variable:
    export OPENAI_API_KEY="sk-your-key"  # Mac/Linux
    set OPENAI_API_KEY=sk-your-key       # Windows CMD
    """
    load_dotenv()

    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Get your API key from: https://platform.openai.com/api-keys")
        exit(1)
    
    def on_transcript(text):
        print(f"\n{'='*60}")
        print(f"[TRANSCRIPT] {text}")
        print('='*60 + "\n")
    
    processor = AudioProcessor(on_transcript, api_key)
    
    print("\n" + "="*60)
    print("AUDIO PROCESSOR TEST - OpenAI Realtime API")
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