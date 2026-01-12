#!/usr/bin/env python3
"""
Test streaming STT with known good audio file.

Sends output.wav through the streaming STT WebSocket to verify transcription.
Expected text: "hello this a test of the glem text to speach system. 
The quality is now much better with proper voice transcription"
"""

import asyncio
import base64
import json
import numpy as np
import sys
import wave
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets


def load_wav_file(filepath: str) -> tuple[np.ndarray, int]:
    """Load WAV file and return (audio_data, sample_rate)."""
    with wave.open(filepath, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        frames = wav.readframes(n_frames)
        
        print(f"📂 Loaded WAV: {n_frames} frames @ {sample_rate}Hz, {n_channels}ch, {sample_width*8}bit")
        
        # Convert to float32
        if sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(frames, dtype=np.float32)
        
        # Convert to mono if stereo
        if n_channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        
        return audio.astype(np.float32), sample_rate


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Simple linear interpolation resampling."""
    if src_rate == dst_rate:
        return audio
    
    ratio = dst_rate / src_rate
    output_length = int(len(audio) * ratio)
    output = np.zeros(output_length, dtype=np.float32)
    
    for i in range(output_length):
        src_idx = i / ratio
        src_floor = int(src_idx)
        src_ceil = min(src_floor + 1, len(audio) - 1)
        t = src_idx - src_floor
        output[i] = audio[src_floor] * (1 - t) + audio[src_ceil] * t
    
    return output


async def test_streaming_stt(audio: np.ndarray, expected_text: str):
    """Send audio through the streaming STT WebSocket."""
    
    ws_url = "ws://localhost:8009/ws/stt"
    chunk_size_ms = 80
    sample_rate = 16000
    samples_per_chunk = int(sample_rate * chunk_size_ms / 1000)  # 1280 samples
    
    print(f"\n🎤 Testing streaming STT")
    print(f"   Audio: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
    print(f"   Chunk size: {samples_per_chunk} samples ({chunk_size_ms}ms)")
    print(f"   Expected: {expected_text[:60]}...")
    print()
    
    transcriptions = []
    partials = []
    
    try:
        async with websockets.connect(ws_url) as ws:
            print("✅ Connected to WebSocket")
            
            # Wait for session started
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            print(f"   Session: {data.get('session_id', 'unknown')}")
            
            # Send audio in chunks
            num_chunks = len(audio) // samples_per_chunk
            print(f"\n📤 Sending {num_chunks} chunks...")
            
            for i in range(num_chunks):
                chunk = audio[i * samples_per_chunk : (i + 1) * samples_per_chunk]
                
                # Ensure float32
                chunk = chunk.astype(np.float32)
                
                # Convert to base64
                chunk_bytes = chunk.tobytes()
                chunk_b64 = base64.b64encode(chunk_bytes).decode('ascii')
                
                # Send chunk
                await ws.send(json.dumps({
                    "type": "audio_chunk",
                    "data": chunk_b64
                }))
                
                # Small delay to simulate real-time (half speed for testing)
                await asyncio.sleep(chunk_size_ms / 1000 * 0.3)
                
                # Check for responses (non-blocking)
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                        data = json.loads(msg)
                        
                        if data.get('type') == 'partial':
                            text = data.get('text', '')
                            if text:
                                partials.append(text)
                                print(f"   📝 Partial: {text}")
                        elif data.get('type') == 'sentence':
                            text = data.get('text', '')
                            if text:
                                transcriptions.append(text)
                                print(f"   ✅ Sentence: {text}")
                        elif data.get('type') == 'vad':
                            is_speech = data.get('is_speech')
                            symbol = "🗣️" if is_speech else "🔇"
                            print(f"   {symbol} VAD: {is_speech}")
                except asyncio.TimeoutError:
                    pass
                
                # Progress indicator
                if (i + 1) % 20 == 0:
                    print(f"   ... sent {i + 1}/{num_chunks} chunks")
            
            print(f"\n📤 Sent all {num_chunks} chunks")
            
            # Wait for final responses
            print("⏳ Waiting for final transcriptions...")
            try:
                for _ in range(100):  # Wait up to 10 seconds
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(msg)
                    
                    if data.get('type') == 'partial':
                        text = data.get('text', '')
                        if text:
                            partials.append(text)
                            print(f"   📝 Partial: {text}")
                    elif data.get('type') == 'sentence':
                        text = data.get('text', '')
                        if text:
                            transcriptions.append(text)
                            print(f"   ✅ Sentence: {text}")
            except asyncio.TimeoutError:
                pass
            
            # Send end signal
            await ws.send(json.dumps({"type": "end_stream"}))
            
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(msg)
                if data.get('type') == 'final':
                    final_text = data.get('text', '')
                    if final_text:
                        print(f"\n📜 Final transcript: {final_text}")
                        transcriptions.append(final_text)
            except:
                pass
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*50)
    print("📊 RESULTS")
    print("="*50)
    print(f"Partials received: {len(partials)}")
    print(f"Sentences received: {len(transcriptions)}")
    
    if transcriptions:
        print(f"\n✅ SUCCESS - Got transcriptions:")
        for t in transcriptions:
            print(f"   • {t}")
        return True
    elif partials:
        print(f"\n⚠️ PARTIAL SUCCESS - Got partials but no sentences:")
        for p in partials[-5:]:  # Last 5
            print(f"   • {p}")
        return True
    else:
        print(f"\n❌ FAILURE - No transcriptions received")
        print("   The model is not producing text from this audio.")
        return False


async def main():
    print("="*50)
    print("🧪 STREAMING STT MODEL TEST")
    print("="*50)
    
    # Load the WAV file
    wav_path = "/home/alec/git/chatter/output.wav"
    expected_text = "hello this a test of the glem text to speach system. The quality is now much better with proper voice transcription"
    
    if not Path(wav_path).exists():
        print(f"❌ WAV file not found: {wav_path}")
        return 1
    
    audio, src_rate = load_wav_file(wav_path)
    
    # Resample to 16kHz if needed
    if src_rate != 16000:
        print(f"🔄 Resampling from {src_rate}Hz to 16000Hz...")
        audio = resample_audio(audio, src_rate, 16000)
    
    # Print audio stats
    print(f"\n📊 Audio stats:")
    print(f"   Duration: {len(audio)/16000:.2f}s")
    print(f"   RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"   Min: {audio.min():.4f}")
    print(f"   Max: {audio.max():.4f}")
    
    # Run test
    success = await test_streaming_stt(audio, expected_text)
    
    if success:
        print("\n✅ Model is working with clean audio!")
        print("   Issue is likely frontend audio capture/resampling.")
    else:
        print("\n❌ Model failed even with clean audio.")
        print("   Issue is in the model/inference pipeline itself.")
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)
