/**
 * Text-to-Speech Hook
 * 
 * Provides TTS functionality for reading assistant responses aloud.
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { apiService } from '@/services/api'

export interface TTSConfig {
  voice: string                  // Voice to use (alloy, echo, fable, onyx, nova, shimmer)
  model: string                  // TTS model (tts-1, tts-1-hd)
  speed: number                  // Speech speed (0.25 to 4.0)
  autoPlay: boolean              // Automatically play TTS for new responses
  format: 'mp3' | 'wav' | 'opus' | 'flac'
  
  // Callbacks
  onPlayStart?: () => void
  onPlayEnd?: () => void
  onError?: (error: string) => void
}

export interface TTSState {
  isPlaying: boolean
  isLoading: boolean
  isPaused: boolean
  currentText: string | null
  error: string | null
  progress: number              // 0-1 playback progress
}

const defaultConfig: TTSConfig = {
  voice: 'alloy',
  model: 'tts-1',
  speed: 1.0,
  autoPlay: false,
  format: 'mp3',
}

export function useTTS(userConfig: Partial<TTSConfig> = {}) {
  const config = { ...defaultConfig, ...userConfig }
  
  const [state, setState] = useState<TTSState>({
    isPlaying: false,
    isLoading: false,
    isPaused: false,
    currentText: null,
    error: null,
    progress: 0,
  })

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceRef = useRef<AudioBufferSourceNode | null>(null)
  const queueRef = useRef<string[]>([])
  const isProcessingRef = useRef(false)

  // Cleanup audio
  const cleanup = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = ''
      audioRef.current = null
    }
    
    if (sourceRef.current) {
      try {
        sourceRef.current.stop()
      } catch (e) {
        // Ignore errors if already stopped
      }
      sourceRef.current = null
    }
    
    setState(prev => ({
      ...prev,
      isPlaying: false,
      isPaused: false,
      currentText: null,
      progress: 0,
    }))
  }, [])

  // Stop playback
  const stop = useCallback(() => {
    queueRef.current = []
    isProcessingRef.current = false
    cleanup()
  }, [cleanup])

  // Pause playback
  const pause = useCallback(() => {
    if (audioRef.current && state.isPlaying) {
      audioRef.current.pause()
      setState(prev => ({ ...prev, isPlaying: false, isPaused: true }))
    }
  }, [state.isPlaying])

  // Resume playback
  const resume = useCallback(() => {
    if (audioRef.current && state.isPaused) {
      audioRef.current.play()
      setState(prev => ({ ...prev, isPlaying: true, isPaused: false }))
    }
  }, [state.isPaused])

  // Toggle pause/resume
  const togglePause = useCallback(() => {
    if (state.isPaused) {
      resume()
    } else if (state.isPlaying) {
      pause()
    }
  }, [state.isPaused, state.isPlaying, pause, resume])

  // Process the next item in the queue
  const processQueue = useCallback(async () => {
    if (isProcessingRef.current || queueRef.current.length === 0) {
      return
    }
    
    isProcessingRef.current = true
    const text = queueRef.current.shift()!
    
    setState(prev => ({
      ...prev,
      isLoading: true,
      currentText: text,
      error: null,
    }))
    
    try {
      // Get audio from TTS service
      const audioData = await apiService.synthesizeSpeech(text, {
        voice: config.voice,
        model: config.model,
        speed: config.speed,
        response_format: config.format,
      })
      
      // Create audio blob and URL
      const audioBlob = new Blob([audioData], { 
        type: config.format === 'mp3' ? 'audio/mpeg' : `audio/${config.format}` 
      })
      const audioUrl = URL.createObjectURL(audioBlob)
      
      // Create and configure audio element
      const audio = new Audio(audioUrl)
      audioRef.current = audio
      
      audio.onloadedmetadata = () => {
        setState(prev => ({ ...prev, isLoading: false }))
      }
      
      audio.onplay = () => {
        setState(prev => ({ ...prev, isPlaying: true, isPaused: false }))
        config.onPlayStart?.()
      }
      
      audio.ontimeupdate = () => {
        if (audio.duration) {
          setState(prev => ({ ...prev, progress: audio.currentTime / audio.duration }))
        }
      }
      
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl)
        setState(prev => ({
          ...prev,
          isPlaying: false,
          currentText: null,
          progress: 0,
        }))
        config.onPlayEnd?.()
        
        isProcessingRef.current = false
        
        // Process next item in queue
        if (queueRef.current.length > 0) {
          processQueue()
        }
      }
      
      audio.onerror = () => {
        URL.revokeObjectURL(audioUrl)
        const errorMessage = 'Failed to play audio'
        setState(prev => ({
          ...prev,
          isLoading: false,
          isPlaying: false,
          error: errorMessage,
        }))
        config.onError?.(errorMessage)
        
        isProcessingRef.current = false
        
        // Try next item in queue
        if (queueRef.current.length > 0) {
          processQueue()
        }
      }
      
      // Start playback
      await audio.play()
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'TTS failed'
      setState(prev => ({
        ...prev,
        isLoading: false,
        isPlaying: false,
        error: errorMessage,
      }))
      config.onError?.(errorMessage)
      
      isProcessingRef.current = false
      
      // Try next item in queue
      if (queueRef.current.length > 0) {
        processQueue()
      }
    }
  }, [config])

  // Speak text
  const speak = useCallback((text: string) => {
    if (!text || !text.trim()) return
    
    // Add to queue and process
    queueRef.current.push(text.trim())
    
    if (!isProcessingRef.current) {
      processQueue()
    }
  }, [processQueue])

  // Speak text immediately (clears queue)
  const speakNow = useCallback((text: string) => {
    if (!text || !text.trim()) return
    
    // Stop current playback and clear queue
    stop()
    
    // Speak immediately
    queueRef.current = [text.trim()]
    processQueue()
  }, [stop, processQueue])

  // Clear the queue
  const clearQueue = useCallback(() => {
    queueRef.current = []
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop()
    }
  }, [stop])

  return {
    ...state,
    speak,
    speakNow,
    stop,
    pause,
    resume,
    togglePause,
    clearQueue,
    queueLength: queueRef.current.length,
    config,
  }
}

export default useTTS
