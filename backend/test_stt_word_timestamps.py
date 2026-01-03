#!/usr/bin/env python3
"""
Test script for STT word-level timestamps functionality.
"""
import asyncio
import httpx
import os

def get_test_audio_file():
    """Use the provided output.wav file for testing."""
    return "/home/alec/git/llama-nexus/output.wav"

async def test_stt_word_timestamps():
    """Test STT transcription with word-level timestamps."""

    # Use the provided test audio file
    temp_filename = get_test_audio_file()

    try:
        # Check if file exists
        import os
        if not os.path.exists(temp_filename):
            print(f"Test audio file not found: {temp_filename}")
            return

        # Test endpoints
        base_url = "http://localhost:8700"

        # Check STT status first
        print("Checking STT service status...")
        async with httpx.AsyncClient(timeout=30.0) as client:
            status_response = await client.get(f"{base_url}/api/v1/stt/status")
            if status_response.status_code != 200:
                print(f"STT service not available: {status_response.status_code}")
                return

            status_data = status_response.json()
            if not status_data.get("running", False):
                print("STT service is not running. Please start it first.")
                return

            print("STT service is running.")

        # Test transcription without word timestamps
        print("\nTesting transcription without word timestamps...")
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(temp_filename, 'rb') as audio_file:
                files = {"file": ("test.wav", audio_file, "audio/wav")}
                data = {"model": "base", "response_format": "json"}

                response = await client.post(
                    f"{base_url}/api/v1/stt/transcribe",
                    files=files,
                    data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"Standard transcription result: {result}")
                else:
                    print(f"Standard transcription failed: {response.status_code} - {response.text}")

        # Test transcription with word timestamps
        print("\nTesting transcription with word timestamps...")
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(temp_filename, 'rb') as audio_file:
                files = {"file": ("test.wav", audio_file, "audio/wav")}
                data = {"model": "base", "word_timestamps": "true"}

                response = await client.post(
                    f"{base_url}/api/v1/stt/transcribe",
                    files=files,
                    data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"Word timestamps transcription result: {result}")

                    # Check if word timestamps are present
                    if "words" in result:
                        print(f"\nWord-level timestamps found! {len(result['words'])} words:")
                        for word_info in result["words"][:5]:  # Show first 5 words
                            print(f"  '{word_info.get('word', '')}': {word_info.get('start', 0):.2f}s - {word_info.get('end', 0):.2f}s")
                        if len(result["words"]) > 5:
                            print(f"  ... and {len(result['words']) - 5} more words")
                    else:
                        print("No word timestamps found in response")
                else:
                    print(f"Word timestamps transcription failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error during testing: {e}")

    # No cleanup needed - using existing file

if __name__ == "__main__":
    asyncio.run(test_stt_word_timestamps())

