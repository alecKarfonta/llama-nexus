"""
Nemotron Streaming Engine

Wraps NVIDIA Nemotron-Speech-Streaming-En-0.6b for cache-aware
streaming inference with sentence boundary detection.
"""

import os
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

# Thread pool for offloading blocking model operations from async event loop
# Note: Model inference is sequential on GPU - this pool just prevents blocking the event loop
_executor = ThreadPoolExecutor(max_workers=2)


@dataclass
class Sentence:
    """A detected sentence from the transcription."""
    text: str
    confidence: float = 1.0
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ChunkResult:
    """Result from processing a single audio chunk."""
    partial_text: str = ""
    sentences: List[Sentence] = field(default_factory=list)
    is_speech: bool = False
    vad_changed: bool = False


@dataclass
class FinalResult:
    """Final result when streaming session ends."""
    full_text: str = ""
    sentences: List[Sentence] = field(default_factory=list)
    duration_seconds: float = 0.0


class StreamingSession:
    """
    Per-connection streaming session with cache state.
    
    Maintains the FastConformer encoder cache for efficient
    streaming inference without redundant computation.
    """
    
    def __init__(
        self,
        session_id: str,
        model,
        chunk_size_ms: int = 80,
        sample_rate: int = 16000,
        sentence_end_silence_ms: int = 500,
        min_sentence_words: int = 2,
        vad_threshold: float = 0.01,
        model_buffer_ms: int = 896,
    ):
        self.session_id = session_id
        self.model = model
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.sentence_end_silence_ms = sentence_end_silence_ms
        self.min_sentence_words = min_sentence_words
        self.vad_threshold = vad_threshold
        
        # Cache state for streaming inference (old API)
        self.cache_state = None
        self.previous_hypotheses = None
        
        # Cache states for conformer_stream_step (new API)
        self.cache_last_channel = None
        self.cache_last_time = None
        self.cache_last_channel_len = None
        self.pred_out_stream = None
        self.cache_pre_encode = None
        
        # Initialize cache states from model if available
        if model is not None:
            try:
                self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = \
                    model.encoder.get_initial_cache_state(batch_size=1)
                
                # Initialize pre-encode cache with zeros (critical for first chunk)
                import torch
                pre_encode_cache_size = getattr(model.encoder.streaming_cfg, 'pre_encode_cache_size', [0, 0])
                if isinstance(pre_encode_cache_size, (list, tuple)) and len(pre_encode_cache_size) > 1:
                    cache_size = pre_encode_cache_size[1]
                else:
                    cache_size = 0
                    
                if cache_size > 0:
                    n_feat = 128  # Mel-spec features
                    self.cache_pre_encode = torch.zeros(1, n_feat, cache_size, device=model.device)
                    logger.info(f"Initialized cache_pre_encode: {self.cache_pre_encode.shape}")
                    
            except Exception as e:
                logger.warning(f"Could not initialize cache state: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # Transcription state
        self.full_transcript = ""
        self.pending_text = ""  # Text not yet committed to a sentence
        self.sentences: List[Sentence] = []
        
        # VAD state
        self.is_speech = False
        self.silence_start_time: Optional[float] = None
        self.speech_start_time: Optional[float] = None
        
        # Audio buffering (for VAD processing)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.samples_per_chunk = int(sample_rate * chunk_size_ms / 1000)
        self.total_samples_processed = 0
        
        # Model-level audio buffer for inference
        # NeMo expects ~105-112 mel frames per chunk (see streaming_cfg.chunk_size)
        # At 10ms hop, 112 frames = 1120ms of audio = 17920 samples at 16kHz
        # Configurable via model_buffer_ms parameter
        self.model_audio_buffer = np.array([], dtype=np.float32)
        self.model_samples_per_inference = int(sample_rate * model_buffer_ms / 1000)
        
        # Timing
        self.session_start_time = time.time()
        
        logger.info(f"Created streaming session {session_id} "
                   f"(chunk={chunk_size_ms}ms, samples_per_chunk={self.samples_per_chunk}, "
                   f"model_inference_size={self.model_samples_per_inference})")
    
    def update_config(self, chunk_size_ms: Optional[int] = None):
        """Update session configuration dynamically."""
        if chunk_size_ms:
            self.chunk_size_ms = chunk_size_ms
            self.samples_per_chunk = int(self.sample_rate * chunk_size_ms / 1000)
            logger.info(f"Session {self.session_id}: updated chunk_size to {chunk_size_ms}ms")
    
    def _reset_streaming_cache(self):
        """Reset all streaming cache states for a fresh utterance."""
        if self.model is not None:
            try:
                import torch
                # Reset encoder cache states
                self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = \
                    self.model.encoder.get_initial_cache_state(batch_size=1)
                
                # Reset decoder/prediction states
                self.previous_hypotheses = None
                self.pred_out_stream = None
                
                # Reset pre-encode cache to zeros
                pre_encode_cache_size = getattr(self.model.encoder.streaming_cfg, 'pre_encode_cache_size', [0, 0])
                if isinstance(pre_encode_cache_size, (list, tuple)) and len(pre_encode_cache_size) > 1:
                    cache_size = pre_encode_cache_size[1]
                else:
                    cache_size = 0
                    
                if cache_size > 0:
                    num_channels = self.model.cfg.preprocessor.features
                    device = next(self.model.parameters()).device
                    self.cache_pre_encode = torch.zeros(
                        (1, num_channels, cache_size), 
                        device=device
                    )
                
                # Also reset the model audio buffer
                self.model_audio_buffer = np.array([], dtype=np.float32)
                
                logger.debug(f"🔄 Reset streaming cache for new utterance")
            except Exception as e:
                logger.warning(f"Could not reset cache state: {e}")
    
    def _detect_sentence_boundary(self, text: str) -> Tuple[List[str], str]:
        """
        Detect sentence boundaries based on punctuation.
        
        Returns:
            Tuple of (complete_sentences, remaining_text)
        """
        sentences = []
        remaining = text
        
        # Sentence-ending punctuation
        end_markers = ['. ', '? ', '! ', '.\n', '?\n', '!\n']
        
        for marker in end_markers:
            while marker in remaining:
                idx = remaining.index(marker)
                sentence = remaining[:idx + 1].strip()
                
                # Only count as sentence if it has enough words
                word_count = len(sentence.split())
                if word_count >= self.min_sentence_words:
                    sentences.append(sentence)
                    remaining = remaining[idx + len(marker):].strip()
                else:
                    # Too short, keep accumulating
                    break
        
        return sentences, remaining
    
    async def process_chunk(self, audio_data: np.ndarray) -> ChunkResult:
        """
        Process an audio chunk through the streaming model.
        
        Args:
            audio_data: Float32 audio samples (16kHz mono)
            
        Returns:
            ChunkResult with partial text, sentences, and VAD state
        """
        result = ChunkResult()
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])
        
        # Process complete chunks
        while len(self.audio_buffer) >= self.samples_per_chunk:
            chunk = self.audio_buffer[:self.samples_per_chunk]
            self.audio_buffer = self.audio_buffer[self.samples_per_chunk:]
            
            # Run inference in thread pool to not block event loop
            chunk_result = await asyncio.get_event_loop().run_in_executor(
                _executor,
                self._process_chunk_sync,
                chunk
            )
            
            # Merge results
            if chunk_result.partial_text:
                result.partial_text = chunk_result.partial_text
            result.sentences.extend(chunk_result.sentences)
            if chunk_result.vad_changed:
                result.vad_changed = True
                result.is_speech = chunk_result.is_speech
        
        return result
    
    def _process_chunk_sync(self, chunk: np.ndarray) -> ChunkResult:
        """
        Synchronous chunk processing (runs in thread pool).
        
        This is where the actual NeMo inference happens.
        """
        result = ChunkResult()
        current_time = time.time()
        
        try:
            # Calculate RMS for simple VAD
            rms = np.sqrt(np.mean(chunk ** 2))
            
            # VAD with hysteresis to reduce jitter
            # Higher threshold to trigger speech, lower threshold to trigger silence
            if self.is_speech:
                # Currently speaking - need lower RMS to go to silence
                new_is_speech = bool(rms > self.vad_threshold * 0.6)
            else:
                # Currently silent - need higher RMS to trigger speech
                new_is_speech = bool(rms > self.vad_threshold)
            
            # Track prolonged silence for warning
            if not hasattr(self, '_silence_warning_logged'):
                self._silence_warning_logged = False
            
            # Track VAD state changes
            if new_is_speech != self.is_speech:
                result.vad_changed = True
                result.is_speech = new_is_speech
                self.is_speech = new_is_speech
                
                if new_is_speech:
                    self.speech_start_time = current_time
                    self.silence_start_time = None
                    self._silence_warning_logged = False
                    self._cache_reset_pending = False  # Cancel pending reset
                    logger.info(f"🗣️ Speech started (RMS={rms:.4f})")
                else:
                    self.silence_start_time = current_time
                    self._cache_reset_pending = True  # Schedule cache reset
                    logger.info(f"🔇 Silence detected (RMS={rms:.6f})")
            
            # TEMPORARILY DISABLED: Cache reset during streaming
            # Per NeMo reference, cache should persist across chunks for continuous transcription
            # Uncommenting this would reset cache after sentence_end_silence_ms of silence
            # if not hasattr(self, '_cache_reset_pending'):
            #     self._cache_reset_pending = False
            # if (self._cache_reset_pending and 
            #     self.silence_start_time and
            #     (current_time - self.silence_start_time) * 1000 >= self.sentence_end_silence_ms):
            #     self._reset_streaming_cache()
            #     self._cache_reset_pending = False
            
            # Warn about prolonged silence (>10 seconds)
            if (self.silence_start_time and 
                not self._silence_warning_logged and
                (current_time - self.silence_start_time) > 10.0):
                logger.warning(f"⚠️ Prolonged silence detected ({current_time - self.silence_start_time:.1f}s) - check audio input")
                self._silence_warning_logged = True
            
            # Run streaming transcription
            if self.model is not None:
                # Buffer audio until we have enough for model's expected chunk size
                # Model trained with streaming_cfg.chunk_size=[105,112] = ~1.12s per inference
                self.model_audio_buffer = np.concatenate([self.model_audio_buffer, chunk])
                
                transcription = ""
                while len(self.model_audio_buffer) >= self.model_samples_per_inference:
                    # Take the model's chunk size
                    model_chunk = self.model_audio_buffer[:self.model_samples_per_inference]
                    self.model_audio_buffer = self.model_audio_buffer[self.model_samples_per_inference:]
                    
                    # Prepare audio for NeMo (expects specific format)
                    audio_signal = model_chunk.reshape(1, -1)  # [batch, samples]
                    audio_length = np.array([len(model_chunk)])
                    
                    # Run cache-aware streaming inference
                    chunk_transcription, self.cache_state, self.previous_hypotheses = \
                        self._run_streaming_inference(audio_signal, audio_length)
                    
                    if chunk_transcription:
                        transcription = chunk_transcription
                
                if transcription:
                    # Update pending text
                    self.pending_text = transcription
                    result.partial_text = transcription
                    
                    # Check for complete sentences
                    sentences, self.pending_text = self._detect_sentence_boundary(transcription)
                    
                    for sent_text in sentences:
                        sentence = Sentence(
                            text=sent_text,
                            confidence=0.9,  # TODO: Get from model
                            start_time=self.speech_start_time or 0,
                            end_time=current_time,
                        )
                        result.sentences.append(sentence)
                        self.sentences.append(sentence)
                        self.full_transcript += sent_text + " "
            
            # Check for silence-based sentence end (INDEPENDENT of transcription)
            # This commits pending text as a sentence after sustained silence
            if (self.silence_start_time and 
                self.pending_text and
                (current_time - self.silence_start_time) * 1000 >= self.sentence_end_silence_ms):
                
                # Commit pending text as sentence after silence
                word_count = len(self.pending_text.split())
                if word_count >= self.min_sentence_words:
                    sentence = Sentence(
                        text=self.pending_text.strip(),
                        confidence=0.85,
                        start_time=self.speech_start_time or 0,
                        end_time=current_time,
                    )
                    result.sentences.append(sentence)
                    self.sentences.append(sentence)
                    self.full_transcript += self.pending_text.strip() + " "
                    self.pending_text = ""
            
            self.total_samples_processed += len(chunk)
            
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")
        
        return result
    
    def _run_streaming_inference(
        self,
        audio_signal: np.ndarray,
        audio_length: np.ndarray
    ) -> Tuple[str, any, any]:
        """
        Run NeMo cache-aware streaming inference.
        
        Uses the correct NeMo streaming API:
        1. Preprocess raw audio to mel-spectrogram (processed_signal)
        2. Call conformer_stream_step with processed signal
        3. Manage encoder cache states
        """
        try:
            import torch
            from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
            
            # Convert to torch tensor: (1, num_samples)
            audio_tensor = torch.from_numpy(audio_signal).float()
            audio_tensor = audio_tensor.flatten().unsqueeze(0)
            audio_len_tensor = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)
            
            device = next(self.model.parameters()).device
            audio_tensor = audio_tensor.to(device)
            audio_len_tensor = audio_len_tensor.to(device)
            
            with torch.no_grad():
                # Preprocess raw audio to mel-spectrogram
                processed_signal, processed_signal_length = self.model.preprocessor(
                    input_signal=audio_tensor, 
                    length=audio_len_tensor
                )
                
                # Prepend with pre-encode cache if we have one
                if self.cache_pre_encode is not None:
                    processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
                    processed_signal_length = processed_signal_length + self.cache_pre_encode.shape[2]
                
                # Save pre-encode cache for next chunk
                pre_encode_cache_size = getattr(self.model.encoder.streaming_cfg, 'pre_encode_cache_size', [0, 0])
                if isinstance(pre_encode_cache_size, (list, tuple)) and len(pre_encode_cache_size) > 1:
                    cache_size = pre_encode_cache_size[1]
                else:
                    cache_size = 0
                if cache_size > 0:
                    self.cache_pre_encode = processed_signal[:, :, -cache_size:]
                
                # Check which streaming API is available
                if hasattr(self.model, 'conformer_stream_step'):
                    # Track step number for drop_extra_pre_encoded calculation
                    if not hasattr(self, '_step_num'):
                        self._step_num = 0
                    
                    # Calculate drop_extra_pre_encoded per NeMo reference:
                    # For the first step, no need to drop tokens as no caching is used
                    if self._step_num == 0:
                        drop_extra = 0
                    else:
                        drop_extra = getattr(self.model.encoder.streaming_cfg, 'drop_extra_pre_encoded', 2)
                    
                    # Full cache-aware streaming API
                    (
                        pred_out,
                        transcribed_texts,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                    ) = self.model.conformer_stream_step(
                        processed_signal=processed_signal,
                        processed_signal_length=processed_signal_length,
                        cache_last_channel=self.cache_last_channel,
                        cache_last_time=self.cache_last_time,
                        cache_last_channel_len=self.cache_last_channel_len,
                        keep_all_outputs=False,  # Set to True at end of stream
                        previous_hypotheses=self.previous_hypotheses,
                        previous_pred_out=self.pred_out_stream,
                        drop_extra_pre_encoded=drop_extra,
                        return_transcription=True,
                    )
                    
                    self._step_num += 1
                    
                    # Update cache states
                    self.cache_last_channel = cache_last_channel
                    self.cache_last_time = cache_last_time
                    self.cache_last_channel_len = cache_last_channel_len
                    self.previous_hypotheses = previous_hypotheses
                    self.pred_out_stream = pred_out
                    
                    # Extract text
                    if transcribed_texts and len(transcribed_texts) > 0:
                        if isinstance(transcribed_texts[0], Hypothesis):
                            text = transcribed_texts[0].text
                        else:
                            text = str(transcribed_texts[0])
                        
                        # Only log when we get actual transcription
                        if text.strip():
                            logger.info(f"📝 Transcription: '{text}'")
                    else:
                        text = ""
                    
                    return text, None, None
                
                elif hasattr(self.model, 'transcribe'):
                    # Fallback to batch transcribe (non-streaming)
                    result = self.model.transcribe(
                        audio=audio_tensor,
                        batch_size=1,
                    )
                    text = result[0] if result else ""
                    return text, None, None
                    
                else:
                    logger.warning("Model doesn't have expected transcribe methods")
                    return "", None, None
                    
        except Exception as e:
            logger.error(f"Streaming inference error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "", self.cache_state, self.previous_hypotheses
    
    def get_final_result(self) -> FinalResult:
        """Get final transcription result when session ends."""
        # Commit any remaining pending text
        if self.pending_text.strip():
            sentence = Sentence(
                text=self.pending_text.strip(),
                confidence=0.8,
                start_time=self.speech_start_time or 0,
                end_time=time.time(),
            )
            self.sentences.append(sentence)
            self.full_transcript += self.pending_text.strip()
        
        duration = time.time() - self.session_start_time
        
        return FinalResult(
            full_text=self.full_transcript.strip(),
            sentences=self.sentences,
            duration_seconds=duration,
        )


class NemotronStreamingEngine:
    """
    Main engine for managing Nemotron streaming STT.
    
    Handles model loading, session management, and inference coordination.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b",
        chunk_size_ms: int = 80,
        sample_rate: int = 16000,
        sentence_end_silence_ms: int = 500,
        min_sentence_words: int = 2,
        vad_threshold: float = 0.01,
        vad_silence_threshold: float = 0.005,
        vad_hysteresis_ms: int = 100,
        model_buffer_ms: int = 896,
    ):
        self.model_name = model_name
        self.chunk_size_ms = chunk_size_ms
        self.sample_rate = sample_rate
        self.sentence_end_silence_ms = sentence_end_silence_ms
        self.min_sentence_words = min_sentence_words
        self.vad_threshold = vad_threshold
        self.vad_silence_threshold = vad_silence_threshold
        self.vad_hysteresis_ms = vad_hysteresis_ms
        self.model_buffer_ms = model_buffer_ms
        
        self.model = None
        self.is_loaded = False
        self.sessions: Dict[str, StreamingSession] = {}
        
        # Configure streaming context
        # att_context_size = [left_context, right_context] in 80ms frames
        # Lower right context = lower latency
        self._att_context_map = {
            80: [70, 0],    # 0.08s latency
            160: [70, 1],   # 0.16s latency
            560: [70, 6],   # 0.56s latency
            1120: [70, 13], # 1.12s latency
        }
    
    async def load_model(self):
        """Load the Nemotron model."""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Run model loading in thread pool
            self.model = await asyncio.get_event_loop().run_in_executor(
                _executor,
                self._load_model_sync
            )
            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
            raise
    
    def _load_model_sync(self):
        """Synchronous model loading (runs in thread pool)."""
        import nemo.collections.asr as nemo_asr
        
        # Load the pretrained streaming model
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name
        )
        
        # Configure for streaming
        att_context = self._att_context_map.get(self.chunk_size_ms, [70, 0])
        logger.info(f"Configuring streaming with att_context_size={att_context}, chunk_size={self.chunk_size_ms}ms")
        
        if hasattr(model, 'set_streaming_cfg'):
            # Set streaming configuration
            try:
                model.set_streaming_cfg({
                    'att_context_size': att_context,
                    'chunk_size': self.chunk_size_ms,
                })
                logger.info(f"✅ set_streaming_cfg applied successfully")
            except Exception as e:
                logger.warning(f"set_streaming_cfg failed: {e}")
        else:
            logger.warning("Model does not have set_streaming_cfg method")
            
        # Log encoder streaming config for debugging
        # NOTE: Do NOT modify streaming_cfg - model is trained with specific chunk sizes
        # Modifying causes empty transcriptions
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'streaming_cfg'):
            logger.info(f"Model streaming_cfg: {model.encoder.streaming_cfg}")
            # Store chunk size info for audio buffering
            self._model_chunk_frames = model.encoder.streaming_cfg.chunk_size
            self._model_shift_frames = model.encoder.streaming_cfg.shift_size
        
        # Move to GPU if available
        import torch
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"Model moved to GPU")
        
        model.eval()
        return model
    
    async def create_session(self, session_id: str) -> StreamingSession:
        """Create a new streaming session."""
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, cleaning up old session")
            await self.cleanup_session(session_id)
        
        session = StreamingSession(
            session_id=session_id,
            model=self.model,
            chunk_size_ms=self.chunk_size_ms,
            sample_rate=self.sample_rate,
            sentence_end_silence_ms=self.sentence_end_silence_ms,
            min_sentence_words=self.min_sentence_words,
            vad_threshold=self.vad_threshold,
            model_buffer_ms=self.model_buffer_ms,
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} (total active: {len(self.sessions)})")
        
        return session
    
    async def process_chunk(self, session_id: str, audio_data: np.ndarray) -> ChunkResult:
        """Process an audio chunk for a session."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return await session.process_chunk(audio_data)
    
    async def end_session(self, session_id: str) -> FinalResult:
        """End a streaming session and get final result."""
        session = self.sessions.get(session_id)
        if not session:
            return FinalResult()
        
        result = session.get_final_result()
        await self.cleanup_session(session_id)
        
        return result
    
    async def cleanup_session(self, session_id: str):
        """Clean up a streaming session."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            # Clear cache state to free GPU memory
            session.cache_state = None
            session.previous_hypotheses = None
            logger.info(f"Cleaned up session {session_id} (remaining: {len(self.sessions)})")
