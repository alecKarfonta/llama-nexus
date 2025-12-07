import React, { useState, useMemo } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  Chip,
  Tooltip,
  alpha,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material'
import {
  Compare as CompareIcon,
  Add as AddIcon,
  Remove as RemoveIcon,
  SwapHoriz as SwapIcon,
  ContentCopy as CopyIcon,
  Check as CheckIcon,
  Visibility as ViewIcon,
  VisibilityOff as HideIcon,
} from '@mui/icons-material'

interface SemanticDiffProps {
  original: string
  revised: string
  open: boolean
  onClose: () => void
}

interface DiffSegment {
  type: 'unchanged' | 'added' | 'removed' | 'modified'
  original?: string
  revised?: string
  content?: string
}

// Simple word-level diff algorithm
function computeWordDiff(original: string, revised: string): DiffSegment[] {
  const originalWords = original.split(/(\s+)/)
  const revisedWords = revised.split(/(\s+)/)
  
  const segments: DiffSegment[] = []
  let i = 0, j = 0
  
  while (i < originalWords.length || j < revisedWords.length) {
    if (i >= originalWords.length) {
      // Remaining revised words are additions
      segments.push({ type: 'added', content: revisedWords.slice(j).join('') })
      break
    }
    if (j >= revisedWords.length) {
      // Remaining original words are removals
      segments.push({ type: 'removed', content: originalWords.slice(i).join('') })
      break
    }
    
    if (originalWords[i] === revisedWords[j]) {
      // Words match
      let matchEnd = i
      let matchEndJ = j
      while (matchEnd < originalWords.length && matchEndJ < revisedWords.length && 
             originalWords[matchEnd] === revisedWords[matchEndJ]) {
        matchEnd++
        matchEndJ++
      }
      segments.push({ type: 'unchanged', content: originalWords.slice(i, matchEnd).join('') })
      i = matchEnd
      j = matchEndJ
    } else {
      // Words differ - look for next match
      let foundMatch = false
      const lookAhead = 10
      
      for (let k = 1; k <= lookAhead && !foundMatch; k++) {
        // Check if original word appears later in revised
        if (j + k < revisedWords.length && originalWords[i] === revisedWords[j + k]) {
          segments.push({ type: 'added', content: revisedWords.slice(j, j + k).join('') })
          j = j + k
          foundMatch = true
        }
        // Check if revised word appears later in original
        else if (i + k < originalWords.length && revisedWords[j] === originalWords[i + k]) {
          segments.push({ type: 'removed', content: originalWords.slice(i, i + k).join('') })
          i = i + k
          foundMatch = true
        }
      }
      
      if (!foundMatch) {
        // No match found, mark as modified
        segments.push({ type: 'removed', content: originalWords[i] })
        segments.push({ type: 'added', content: revisedWords[j] })
        i++
        j++
      }
    }
  }
  
  // Merge consecutive segments of same type
  const merged: DiffSegment[] = []
  for (const segment of segments) {
    if (merged.length > 0 && merged[merged.length - 1].type === segment.type) {
      merged[merged.length - 1].content += segment.content
    } else {
      merged.push(segment)
    }
  }
  
  return merged
}

// Compute similarity score (0-100)
function computeSimilarity(original: string, revised: string): number {
  const diff = computeWordDiff(original, revised)
  let unchangedChars = 0
  let totalChars = 0
  
  for (const segment of diff) {
    const len = (segment.content || '').length
    totalChars += len
    if (segment.type === 'unchanged') {
      unchangedChars += len
    }
  }
  
  return totalChars > 0 ? Math.round((unchangedChars / totalChars) * 100) : 100
}

// Extract key concepts (simplified)
function extractConcepts(text: string): string[] {
  // Simple keyword extraction - in production would use NLP
  const stopWords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not', 'only', 'own',
    'same', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'any', 'this',
    'that', 'these', 'those', 'it', 'its'])
  
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter(w => w.length > 3 && !stopWords.has(w))
  
  // Count frequency
  const freq: Record<string, number> = {}
  for (const word of words) {
    freq[word] = (freq[word] || 0) + 1
  }
  
  // Return top concepts
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([word]) => word)
}

// Diff Display Component
interface DiffDisplayProps {
  segments: DiffSegment[]
  viewMode: 'inline' | 'side-by-side'
}

const DiffDisplay: React.FC<DiffDisplayProps> = ({ segments, viewMode }) => {
  if (viewMode === 'inline') {
    return (
      <Box sx={{ p: 2, bgcolor: 'rgba(0, 0, 0, 0.2)', borderRadius: 2, lineHeight: 1.8 }}>
        {segments.map((segment, idx) => {
          if (segment.type === 'unchanged') {
            return <span key={idx}>{segment.content}</span>
          }
          if (segment.type === 'added') {
            return (
              <Box
                key={idx}
                component="span"
                sx={{
                  bgcolor: alpha('#10b981', 0.2),
                  color: '#34d399',
                  px: 0.5,
                  borderRadius: 0.5,
                  textDecoration: 'none',
                }}
              >
                {segment.content}
              </Box>
            )
          }
          if (segment.type === 'removed') {
            return (
              <Box
                key={idx}
                component="span"
                sx={{
                  bgcolor: alpha('#ef4444', 0.2),
                  color: '#f87171',
                  px: 0.5,
                  borderRadius: 0.5,
                  textDecoration: 'line-through',
                }}
              >
                {segment.content}
              </Box>
            )
          }
          return null
        })}
      </Box>
    )
  }
  
  // Side by side view
  const originalSegments = segments.filter(s => s.type !== 'added')
  const revisedSegments = segments.filter(s => s.type !== 'removed')
  
  return (
    <Box sx={{ display: 'flex', gap: 2 }}>
      <Box sx={{ flex: 1, p: 2, bgcolor: 'rgba(0, 0, 0, 0.2)', borderRadius: 2, lineHeight: 1.8 }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Original
        </Typography>
        {originalSegments.map((segment, idx) => {
          if (segment.type === 'unchanged') {
            return <span key={idx}>{segment.content}</span>
          }
          if (segment.type === 'removed') {
            return (
              <Box
                key={idx}
                component="span"
                sx={{
                  bgcolor: alpha('#ef4444', 0.2),
                  color: '#f87171',
                  px: 0.5,
                  borderRadius: 0.5,
                }}
              >
                {segment.content}
              </Box>
            )
          }
          return null
        })}
      </Box>
      <Box sx={{ flex: 1, p: 2, bgcolor: 'rgba(0, 0, 0, 0.2)', borderRadius: 2, lineHeight: 1.8 }}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Revised
        </Typography>
        {revisedSegments.map((segment, idx) => {
          if (segment.type === 'unchanged') {
            return <span key={idx}>{segment.content}</span>
          }
          if (segment.type === 'added') {
            return (
              <Box
                key={idx}
                component="span"
                sx={{
                  bgcolor: alpha('#10b981', 0.2),
                  color: '#34d399',
                  px: 0.5,
                  borderRadius: 0.5,
                }}
              >
                {segment.content}
              </Box>
            )
          }
          return null
        })}
      </Box>
    </Box>
  )
}

// Main Semantic Diff Component
export const SemanticDiff: React.FC<SemanticDiffProps> = ({ original, revised, open, onClose }) => {
  const [viewMode, setViewMode] = useState<'inline' | 'side-by-side'>('inline')
  const [showConcepts, setShowConcepts] = useState(true)
  const [copied, setCopied] = useState(false)

  // Compute diff
  const diff = useMemo(() => computeWordDiff(original, revised), [original, revised])
  
  // Compute similarity
  const similarity = useMemo(() => computeSimilarity(original, revised), [original, revised])
  
  // Extract concepts
  const originalConcepts = useMemo(() => extractConcepts(original), [original])
  const revisedConcepts = useMemo(() => extractConcepts(revised), [revised])
  
  // Find concept changes
  const addedConcepts = revisedConcepts.filter(c => !originalConcepts.includes(c))
  const removedConcepts = originalConcepts.filter(c => !revisedConcepts.includes(c))
  const sharedConcepts = originalConcepts.filter(c => revisedConcepts.includes(c))

  // Stats
  const stats = useMemo(() => {
    let added = 0, removed = 0, unchanged = 0
    for (const segment of diff) {
      const len = (segment.content || '').replace(/\s+/g, ' ').trim().split(' ').length
      if (segment.type === 'added') added += len
      else if (segment.type === 'removed') removed += len
      else if (segment.type === 'unchanged') unchanged += len
    }
    return { added, removed, unchanged }
  }, [diff])

  const handleCopy = () => {
    const report = `Semantic Diff Report
==================

Similarity: ${similarity}%

Statistics:
- Words Added: ${stats.added}
- Words Removed: ${stats.removed}
- Words Unchanged: ${stats.unchanged}

Concepts Added: ${addedConcepts.join(', ') || 'None'}
Concepts Removed: ${removedConcepts.join(', ') || 'None'}
Shared Concepts: ${sharedConcepts.join(', ') || 'None'}

Original:
${original}

Revised:
${revised}
`
    navigator.clipboard.writeText(report)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const getSimilarityColor = (score: number) => {
    if (score >= 80) return '#10b981'
    if (score >= 50) return '#f59e0b'
    return '#ef4444'
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <CompareIcon sx={{ color: '#6366f1' }} />
        Semantic Diff
        <Box sx={{ flex: 1 }} />
        <Tooltip title={copied ? 'Copied!' : 'Copy Report'}>
          <IconButton size="small" onClick={handleCopy}>
            {copied ? <CheckIcon color="success" /> : <CopyIcon />}
          </IconButton>
        </Tooltip>
      </DialogTitle>
      
      <DialogContent>
        {/* Stats Bar */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap', alignItems: 'center' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              px: 2,
              py: 1,
              borderRadius: 2,
              bgcolor: alpha(getSimilarityColor(similarity), 0.1),
              border: `1px solid ${alpha(getSimilarityColor(similarity), 0.3)}`,
            }}
          >
            <Typography variant="subtitle2" sx={{ color: getSimilarityColor(similarity) }}>
              {similarity}% Similar
            </Typography>
          </Box>
          
          <Chip
            icon={<AddIcon sx={{ fontSize: '14px !important' }} />}
            label={`+${stats.added} words`}
            size="small"
            sx={{
              bgcolor: alpha('#10b981', 0.1),
              color: '#34d399',
              '& .MuiChip-icon': { color: '#34d399' },
            }}
          />
          <Chip
            icon={<RemoveIcon sx={{ fontSize: '14px !important' }} />}
            label={`-${stats.removed} words`}
            size="small"
            sx={{
              bgcolor: alpha('#ef4444', 0.1),
              color: '#f87171',
              '& .MuiChip-icon': { color: '#f87171' },
            }}
          />
          
          <Box sx={{ flex: 1 }} />
          
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(_, v) => v && setViewMode(v)}
            size="small"
          >
            <ToggleButton value="inline">Inline</ToggleButton>
            <ToggleButton value="side-by-side">Side by Side</ToggleButton>
          </ToggleButtonGroup>
          
          <IconButton
            size="small"
            onClick={() => setShowConcepts(!showConcepts)}
            sx={{ bgcolor: showConcepts ? alpha('#6366f1', 0.1) : 'transparent' }}
          >
            {showConcepts ? <ViewIcon /> : <HideIcon />}
          </IconButton>
        </Box>

        {/* Concept Analysis */}
        {showConcepts && (
          <Card sx={{ mb: 3, bgcolor: 'rgba(0, 0, 0, 0.2)', border: '1px solid rgba(255, 255, 255, 0.06)' }}>
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Typography variant="subtitle2" sx={{ mb: 2 }}>
                Concept Analysis
              </Typography>
              <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                <Box>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Added Concepts
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {addedConcepts.length > 0 ? addedConcepts.map((c) => (
                      <Chip
                        key={c}
                        label={c}
                        size="small"
                        sx={{
                          height: 22,
                          bgcolor: alpha('#10b981', 0.1),
                          color: '#34d399',
                          fontSize: '0.6875rem',
                        }}
                      />
                    )) : (
                      <Typography variant="caption" color="text.secondary">None</Typography>
                    )}
                  </Box>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Removed Concepts
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {removedConcepts.length > 0 ? removedConcepts.map((c) => (
                      <Chip
                        key={c}
                        label={c}
                        size="small"
                        sx={{
                          height: 22,
                          bgcolor: alpha('#ef4444', 0.1),
                          color: '#f87171',
                          fontSize: '0.6875rem',
                        }}
                      />
                    )) : (
                      <Typography variant="caption" color="text.secondary">None</Typography>
                    )}
                  </Box>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    Shared Concepts
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {sharedConcepts.length > 0 ? sharedConcepts.map((c) => (
                      <Chip
                        key={c}
                        label={c}
                        size="small"
                        sx={{
                          height: 22,
                          bgcolor: alpha('#6366f1', 0.1),
                          color: '#818cf8',
                          fontSize: '0.6875rem',
                        }}
                      />
                    )) : (
                      <Typography variant="caption" color="text.secondary">None</Typography>
                    )}
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
        )}

        {/* Diff Display */}
        <DiffDisplay segments={diff} viewMode={viewMode} />
      </DialogContent>
      
      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  )
}

export default SemanticDiff
