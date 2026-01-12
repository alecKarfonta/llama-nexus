#!/usr/bin/env python3
"""
STT Health Check Script
Runs automatically on container startup to verify the streaming STT service is working.
"""

import asyncio
import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def health_check():
    """Quick health check with synthetic audio."""
    import websockets
    import json
    
    ws_url = os.getenv("STT_WS_URL", "ws://localhost:8009/ws/stream")
    
    try:
        # Generate 2 seconds of silence + tone (simple test pattern)
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Create a simple tone (440Hz) to test audio processing
        t = np.linspace(0, duration, samples, dtype=np.float32)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone at low volume
        
        # Connect and send chunks
        async with websockets.connect(ws_url, close_timeout=5) as ws:
            # Wait for session confirmation
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(response)
            
            if data.get("type") != "session_start":
                print(f"❌ Unexpected response: {data}")
                return False
            
            session_id = data.get("session_id")
            print(f"✅ Connected - Session: {session_id[:8]}...")
            
            # Send a few chunks
            chunk_size = int(sample_rate * 0.08)  # 80ms
            chunks_sent = 0
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                await ws.send(chunk.tobytes())
                chunks_sent += 1
                
                # Check for responses (non-blocking)
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.01)
                except asyncio.TimeoutError:
                    pass
            
            print(f"✅ Sent {chunks_sent} chunks successfully")
            
            # End session
            await ws.send(json.dumps({"type": "end"}))
            
            print("✅ STT Health Check PASSED")
            return True
            
    except Exception as e:
        print(f"❌ STT Health Check FAILED: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(health_check())
    sys.exit(0 if success else 1)
