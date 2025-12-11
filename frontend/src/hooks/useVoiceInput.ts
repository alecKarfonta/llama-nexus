/**
 * Voice Input Hook with Voice Activity Detection (VAD)
 * 
 * Provides continuous voice input with automatic silence detection
 * to determine when the user has finished speaking.
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { apiService } from '@/services/api'

export interface VoiceInputConfig {
  // VAD settings
  silenceThreshold: number       // Volume level below which is considered silence (0-1)
  silenceDuration: number        // How long silence must last before triggering (ms)
  minRecordingTime: number       // Minimum recording time before silence detection kicks in (ms)
  maxRecordingTime: number       // Maximum recording time before auto-stop (ms)
  
  // STT settings
  sttModel?: string              // Whisper model to use
  sttLanguage?: string           // Language code or 'auto'
  noSpeechThreshold: number      // Max no_speech_prob to accept (0-1). Higher = accept more uncertain speech
  
  // Callbacks
  onTranscription?: (text: string) => void
  onError?: (error: string) => void
  onSpeechStart?: () => void
  onSpeechEnd?: () => void
  onVolumeChange?: (volume: number) => void
}

export interface VoiceInputState {
  isListening: boolean           // Microphone is active and listening
  isRecording: boolean           // Currently capturing speech
  isTranscribing: boolean        // Sending audio to STT service
  isSpeaking: boolean            // User is currently speaking (above silence threshold)
  volume: number                 // Current audio volume level (0-1)
  recordingTime: number          // Current recording duration in ms
  silenceTime: number            // How long silence has been detected (ms)
  lastNoSpeechProb: number | null // Last transcription's no_speech_prob (for debugging)
  error: string | null
}

const defaultConfig: VoiceInputConfig = {
  silenceThreshold: 0.04,        // 4% of max volume - adjust based on your mic
  silenceDuration: 1500,         // 1.5 seconds of silence to end recording
  minRecordingTime: 500,         // At least 0.5 seconds of speech before silence detection
  maxRecordingTime: 60000,       // Max 60 seconds
  sttModel: 'base',
  sttLanguage: 'auto',
  noSpeechThreshold: 0.6,        // Reject transcriptions where no_speech_prob > 60%
}

export function useVoiceInput(userConfig: Partial<VoiceInputConfig> = {}) {
  // Store config in a ref to avoid stale closures
  const configRef = useRef({ ...defaultConfig, ...userConfig })
  
  // Update config ref when userConfig changes
  useEffect(() => {
    configRef.current = { ...defaultConfig, ...userConfig }
  }, [userConfig])
  
  const [state, setState] = useState<VoiceInputState>({
    isListening: false,
    isRecording: false,
    isTranscribing: false,
    isSpeaking: false,
    volume: 0,
    recordingTime: 0,
    silenceTime: 0,
    lastNoSpeechProb: null,
    error: null,
  })

  // Refs for audio processing
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  
  // Refs for VAD
  const silenceStartRef = useRef<number | null>(null)
  const recordingStartRef = useRef<number | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  
  // Refs to track state for the audio processing loop (avoids stale closure issues)
  const isListeningRef = useRef(false)
  const isRecordingRef = useRef(false)
  const isTranscribingRef = useRef(false)

  // Transcribe recorded audio
  const transcribeAudio = useCallback(async (audioBlob: Blob) => {
    if (audioBlob.size === 0) {
      console.warn('Empty audio blob, skipping transcription')
      return
    }
    
    console.log('Starting transcription, blob size:', audioBlob.size)
    isTranscribingRef.current = true
    setState(prev => ({ ...prev, isTranscribing: true, error: null, lastNoSpeechProb: null }))
    
    try {
      const result = await apiService.transcribeAudio(audioBlob, {
        model: configRef.current.sttModel,
        language: configRef.current.sttLanguage,
        response_format: 'verbose_json',  // Get segment probabilities
      })
      
      console.log('Transcription result:', result)
      
      // Check no_speech_prob from segments
      let avgNoSpeechProb: number | null = null
      if (result.segments && result.segments.length > 0) {
        const probs = result.segments
          .map(s => s.no_speech_prob)
          .filter((p): p is number => p !== undefined)
        
        if (probs.length > 0) {
          avgNoSpeechProb = probs.reduce((a, b) => a + b, 0) / probs.length
          console.log(`Average no_speech_prob: ${(avgNoSpeechProb * 100).toFixed(1)}% (threshold: ${(configRef.current.noSpeechThreshold * 100).toFixed(0)}%)`)
        }
      }
      
      // Update state with the probability
      setState(prev => ({ ...prev, lastNoSpeechProb: avgNoSpeechProb }))
      
      // Filter out low-confidence speech
      if (avgNoSpeechProb !== null && avgNoSpeechProb > configRef.current.noSpeechThreshold) {
        console.log(`Rejecting transcription - no_speech_prob ${(avgNoSpeechProb * 100).toFixed(1)}% exceeds threshold ${(configRef.current.noSpeechThreshold * 100).toFixed(0)}%`)
        return
      }
      
      if (result.text && result.text.trim()) {
        configRef.current.onTranscription?.(result.text.trim())
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Transcription failed'
      console.error('Transcription error:', errorMessage)
      setState(prev => ({ ...prev, error: errorMessage }))
      configRef.current.onError?.(errorMessage)
    } finally {
      isTranscribingRef.current = false
      setState(prev => ({ ...prev, isTranscribing: false }))
      
      // Reset for next recording
      silenceStartRef.current = null
      recordingStartRef.current = null
      
      console.log('Transcription complete, ready for next recording. isListening:', isListeningRef.current)
    }
  }, [])

  // Start recording
  const startRecordingInternal = useCallback(() => {
    if (!mediaRecorderRef.current || isRecordingRef.current) {
      console.log('Cannot start recording - recorder:', !!mediaRecorderRef.current, 'already recording:', isRecordingRef.current)
      return
    }
    
    console.log('Starting recording')
    chunksRef.current = []
    recordingStartRef.current = Date.now()
    silenceStartRef.current = null
    
    isRecordingRef.current = true
    mediaRecorderRef.current.start(100) // Capture in 100ms chunks
    
    setState(prev => ({ ...prev, isRecording: true, silenceTime: 0 }))
    configRef.current.onSpeechStart?.()
  }, [])

  // Stop recording
  const stopRecordingInternal = useCallback(() => {
    if (!mediaRecorderRef.current || !isRecordingRef.current) {
      return
    }
    
    console.log('Stopping recording')
    isRecordingRef.current = false
    setState(prev => ({ ...prev, isRecording: false }))
    configRef.current.onSpeechEnd?.()
    
    // Stop the recorder - this will trigger ondataavailable and onstop
    if (mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  // Process audio and detect voice activity - runs continuously via requestAnimationFrame
  const processAudio = useCallback(() => {
    // Always schedule next frame first to keep loop running
    if (isListeningRef.current) {
      animationFrameRef.current = requestAnimationFrame(processAudio)
    } else {
      return // Not listening, exit early
    }
    
    // Check if we have what we need
    if (!analyserRef.current) {
      console.warn('processAudio: No analyser')
      return
    }
    
    const config = configRef.current
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)
    
    // Calculate average volume (0-1)
    const average = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length
    const normalizedVolume = average / 255
    
    // Call volume change callback
    config.onVolumeChange?.(normalizedVolume)
    
    const now = Date.now()
    const recordingDuration = recordingStartRef.current ? now - recordingStartRef.current : 0
    
    // Check if user is speaking (above threshold)
    const isSpeaking = normalizedVolume > config.silenceThreshold
    
    // Calculate current silence time for display
    let currentSilenceTime = 0
    if (!isSpeaking && silenceStartRef.current) {
      currentSilenceTime = now - silenceStartRef.current
    }
    
    // Update state for UI
    setState(prev => ({
      ...prev,
      volume: normalizedVolume,
      isSpeaking,
      recordingTime: recordingDuration,
      silenceTime: isRecordingRef.current ? currentSilenceTime : 0,
    }))
    
    // Handle recording state changes based on voice activity
    if (isRecordingRef.current) {
      if (isSpeaking) {
        // User is speaking, reset silence timer
        silenceStartRef.current = null
      } else {
        // Silence detected
        if (silenceStartRef.current === null) {
          silenceStartRef.current = now
        } else {
          const silenceDuration = now - silenceStartRef.current
          
          // Check if we should stop recording
          if (
            silenceDuration >= config.silenceDuration &&
            recordingDuration >= config.minRecordingTime
          ) {
            console.log('Silence threshold reached, stopping recording. Duration:', recordingDuration, 'Silence:', silenceDuration)
            stopRecordingInternal()
            // Don't return - keep the loop running for next utterance
          }
        }
      }
      
      // Check max recording time
      if (recordingDuration >= config.maxRecordingTime) {
        console.log('Max recording time reached')
        stopRecordingInternal()
      }
    } else if (!isTranscribingRef.current) {
      // Not recording and not transcribing - check if we should start
      if (isSpeaking) {
        console.log('Voice detected, starting recording')
        startRecordingInternal()
      }
    }
  }, [startRecordingInternal, stopRecordingInternal])

  // Cleanup function
  const cleanup = useCallback(() => {
    console.log('Cleaning up voice input')
    
    // Reset refs first to stop the processing loop
    isListeningRef.current = false
    isRecordingRef.current = false
    isTranscribingRef.current = false
    
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    
    analyserRef.current = null
    mediaRecorderRef.current = null
    chunksRef.current = []
    silenceStartRef.current = null
    recordingStartRef.current = null
    
    setState(prev => ({
      ...prev,
      isListening: false,
      isRecording: false,
      isSpeaking: false,
      volume: 0,
      recordingTime: 0,
      silenceTime: 0,
    }))
  }, [])

  // Start listening (enables microphone and VAD)
  const startListening = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, error: null }))
      
      // Get microphone access
      console.log('Requesting microphone access...')
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      })
      streamRef.current = stream
      console.log('Microphone access granted')
      
      // Set up audio context for VAD
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      console.log('AudioContext created, state:', audioContext.state)
      
      // Resume audio context if suspended (required by some browsers)
      if (audioContext.state === 'suspended') {
        await audioContext.resume()
        console.log('AudioContext resumed')
      }
      
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.5 // Lower smoothing for faster response
      source.connect(analyser)
      analyserRef.current = analyser
      
      // Set up media recorder
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
      console.log('Using MIME type:', mimeType)
      const mediaRecorder = new MediaRecorder(stream, { mimeType })
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }
      
      mediaRecorder.onstop = async () => {
        console.log('MediaRecorder stopped, chunks:', chunksRef.current.length)
        const audioBlob = new Blob(chunksRef.current, { type: mimeType })
        chunksRef.current = []
        
        // Only transcribe if we have actual audio
        if (audioBlob.size > 0) {
          await transcribeAudio(audioBlob)
        }
      }
      
      mediaRecorderRef.current = mediaRecorder
      
      // Set refs BEFORE starting the loop
      isListeningRef.current = true
      isRecordingRef.current = false
      isTranscribingRef.current = false
      
      setState(prev => ({ ...prev, isListening: true }))
      
      // Start audio processing loop
      console.log('Starting continuous audio processing loop')
      animationFrameRef.current = requestAnimationFrame(processAudio)
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to access microphone'
      console.error('Failed to start listening:', errorMessage)
      setState(prev => ({ ...prev, error: errorMessage }))
      configRef.current.onError?.(errorMessage)
    }
  }, [processAudio, transcribeAudio])

  // Stop listening (disables microphone)
  const stopListening = useCallback(() => {
    cleanup()
  }, [cleanup])

  // Toggle listening state
  const toggleListening = useCallback(() => {
    if (state.isListening) {
      stopListening()
    } else {
      startListening()
    }
  }, [state.isListening, startListening, stopListening])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanup()
    }
  }, [cleanup])

  return {
    ...state,
    startListening,
    stopListening,
    toggleListening,
    config: configRef.current,
  }
}

export default useVoiceInput
