import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  Box,
  IconButton,
  Tooltip,
  Typography,
  Collapse,
} from '@mui/material'
import {
  ContentCopy as CopyIcon,
  Check as CheckIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material'

interface MarkdownRendererProps {
  content: string
  reasoning_content?: string
}

/** Qwen / chat templates use <think>…</think>, not only <think>. */
const THINK_TAG_GROUP = 'redacted_thinking|think|thinking|thought'

/** Remove thinking blocks and stray tags from the main answer so </redacted_thinking> never shows as body text. */
function stripThinkingTagsFromDisplay(s: string): string {
  if (!s) return ''
  let out = s
  const blockRe = new RegExp(`<(${THINK_TAG_GROUP})>([\\s\\S]*?)<\\/\\1>`, 'gi')
  out = out.replace(blockRe, '')
  out = out.replace(/<\/?redacted_thinking>/gi, '')
  out = out.replace(/<\/?think>/gi, '')
  out = out.replace(/<\/?thinking>/gi, '')
  out = out.replace(/<\/?thought>/gi, '')
  return out.replace(/\n{3,}/g, '\n\n').trim()
}

/** Reasoning channel text sometimes includes stray tag fragments; keep panel clean. */
function stripOrphanThinkingTags(s: string): string {
  if (!s) return ''
  return s
    .replace(/<\/?redacted_thinking>/gi, '')
    .replace(/<\/?(?:think|thinking|thought)>/gi, '')
    .trim()
}

interface CodeBlockProps {
  language: string
  value: string
}

const CodeBlock: React.FC<CodeBlockProps> = ({ language, value }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(value)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Box sx={{ position: 'relative', my: 2 }}>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.3)',
          borderRadius: '4px 4px 0 0',
          px: 2,
          py: 0.5,
        }}
      >
        <Typography
          variant="caption"
          sx={{ color: 'text.secondary', fontFamily: 'monospace' }}
        >
          {language || 'code'}
        </Typography>
        <Tooltip title={copied ? 'Copied!' : 'Copy code'}>
          <IconButton size="small" onClick={handleCopy} sx={{ color: 'text.secondary' }}>
            {copied ? <CheckIcon fontSize="small" /> : <CopyIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
      </Box>
      <SyntaxHighlighter
        language={language || 'text'}
        style={oneDark}
        customStyle={{
          margin: 0,
          borderRadius: '0 0 4px 4px',
          fontSize: '0.85rem',
        }}
        showLineNumbers={value.split('\n').length > 3}
      >
        {value}
      </SyntaxHighlighter>
    </Box>
  )
}

// Component to render thinking/reasoning content with collapse
interface ThinkingBlockProps {
  content: string
}

const ThinkingBlock: React.FC<ThinkingBlockProps> = ({ content }) => {
  // Default expanded: models often stream only reasoning_content first (or only); collapsed looked like "no reply"
  const [expanded, setExpanded] = useState(true)

  if (!content) return null

  return (
    <Box
      sx={{
        mb: 2,
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        overflow: 'hidden',
      }}
    >
      <Box
        onClick={() => setExpanded(!expanded)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          py: 1,
          backgroundColor: 'rgba(156, 39, 176, 0.1)',
          cursor: 'pointer',
          '&:hover': {
            backgroundColor: 'rgba(156, 39, 176, 0.15)',
          },
        }}
      >
        <Typography
          variant="body2"
          sx={{
            fontWeight: 500,
            color: 'secondary.main',
            display: 'flex',
            alignItems: 'center',
            gap: 1,
          }}
        >
          Thinking Process
          <Typography variant="caption" color="text.secondary">
            ({content.length} characters)
          </Typography>
        </Typography>
        <IconButton size="small">
          {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        </IconButton>
      </Box>
      <Collapse in={expanded}>
        <Box
          sx={{
            p: 2,
            backgroundColor: 'rgba(0, 0, 0, 0.2)',
            maxHeight: 400,
            overflow: 'auto',
          }}
        >
          <Typography
            variant="body2"
            sx={{
              whiteSpace: 'pre-wrap',
              fontFamily: 'monospace',
              fontSize: '0.8rem',
              color: 'text.secondary',
              lineHeight: 1.6,
            }}
          >
            {content}
          </Typography>
        </Box>
      </Collapse>
    </Box>
  )
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  reasoning_content,
}) => {
  const raw = typeof content === 'string' ? content : ''

  let thinking = ''
  let displayContent = ''

  // 1) Extract all *completed* thinking blocks (<think>...</think>) anywhere in raw.
  const closedBlockRe = new RegExp(`<(${THINK_TAG_GROUP})>([\\s\\S]*?)<\\/\\1>`, 'gi')
  const closedThinking: string[] = []
  const rawWithoutClosed = raw.replace(closedBlockRe, (_m, _tag, inner) => {
    closedThinking.push(inner)
    return ''
  })

  // 2) Handle any *still-streaming* thinking block (open tag with no matching close yet).
  //    This can appear anywhere in the text (often after a leading newline), so do NOT anchor to ^.
  const unclosedRe = new RegExp(`<(${THINK_TAG_GROUP})>([\\s\\S]*)$`, 'i')
  const unclosedMatch = rawWithoutClosed.match(unclosedRe)
  let beforeUnclosed = rawWithoutClosed
  let streamingThinking = ''
  if (unclosedMatch && unclosedMatch.index !== undefined) {
    beforeUnclosed = rawWithoutClosed.slice(0, unclosedMatch.index)
    streamingThinking = unclosedMatch[2]
  }

  const combinedThinking = [...closedThinking, streamingThinking]
    .map((s) => s.trim())
    .filter(Boolean)
    .join('\n\n')

  thinking = combinedThinking
  displayContent = beforeUnclosed

  // reasoning_content (from OpenAI-style split stream) always wins for the panel.
  if (reasoning_content) {
    thinking = reasoning_content
  }

  displayContent = stripThinkingTagsFromDisplay(displayContent)
  const thinkingForPanel = stripOrphanThinkingTags(thinking)

  const hasMainText = Boolean(displayContent && displayContent.trim())
  const hasAnyText = Boolean(raw && raw.trim())

  return (
    <Box sx={{ width: '100%' }}>
      {thinkingForPanel && <ThinkingBlock content={thinkingForPanel} />}
      {!hasMainText && thinkingForPanel && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          The model returned thinking/reasoning above. The final answer appears below when the model emits visible text.
        </Typography>
      )}
      {/* Diagnostic: the stream finished but server sent no content AND no reasoning.
          Surfacing this so an empty server response is never confused with a render bug. */}
      {!hasAnyText && !thinkingForPanel && (
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            mb: 1,
            px: 1,
            py: 0.5,
            color: 'warning.main',
            border: '1px dashed',
            borderColor: 'warning.main',
            borderRadius: 1,
            fontFamily: 'monospace',
            fontSize: '0.75rem',
          }}
        >
          [empty assistant response — the server emitted no content and no reasoning. This is a llama-server/model issue (typically predicted_n=1 with EOS as the first token). Retry the prompt.]
        </Typography>
      )}
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          // Code blocks with syntax highlighting
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '')
            const value = String(children).replace(/\n$/, '')

            if (!inline && (match || value.includes('\n'))) {
              return <CodeBlock language={match?.[1] || ''} value={value} />
            }

            // Inline code
            return (
              <Box
                component="code"
                sx={{
                  backgroundColor: 'rgba(0, 0, 0, 0.3)',
                  px: 0.75,
                  py: 0.25,
                  borderRadius: 0.5,
                  fontFamily: 'monospace',
                  fontSize: '0.85em',
                }}
                {...props}
              >
                {children}
              </Box>
            )
          },
          // Paragraphs
          p({ children }) {
            return (
              <Typography
                variant="body1"
                sx={{ mb: 1.5, lineHeight: 1.7, '&:last-child': { mb: 0 } }}
              >
                {children}
              </Typography>
            )
          },
          // Headers
          h1({ children }) {
            return (
              <Typography variant="h5" sx={{ mt: 3, mb: 1.5, fontWeight: 600 }}>
                {children}
              </Typography>
            )
          },
          h2({ children }) {
            return (
              <Typography variant="h6" sx={{ mt: 2.5, mb: 1, fontWeight: 600 }}>
                {children}
              </Typography>
            )
          },
          h3({ children }) {
            return (
              <Typography
                variant="subtitle1"
                sx={{ mt: 2, mb: 1, fontWeight: 600 }}
              >
                {children}
              </Typography>
            )
          },
          // Lists
          ul({ children }) {
            return (
              <Box
                component="ul"
                sx={{ pl: 2.5, my: 1.5, '& li': { mb: 0.5 } }}
              >
                {children}
              </Box>
            )
          },
          ol({ children }) {
            return (
              <Box
                component="ol"
                sx={{ pl: 2.5, my: 1.5, '& li': { mb: 0.5 } }}
              >
                {children}
              </Box>
            )
          },
          li({ children }) {
            return (
              <Typography component="li" variant="body1" sx={{ lineHeight: 1.7 }}>
                {children}
              </Typography>
            )
          },
          // Blockquotes
          blockquote({ children }) {
            return (
              <Box
                sx={{
                  borderLeft: 3,
                  borderColor: 'primary.main',
                  pl: 2,
                  py: 0.5,
                  my: 2,
                  backgroundColor: 'rgba(0, 0, 0, 0.1)',
                  borderRadius: '0 4px 4px 0',
                }}
              >
                {children}
              </Box>
            )
          },
          // Tables
          table({ children }) {
            return (
              <Box
                sx={{
                  overflowX: 'auto',
                  my: 2,
                  '& table': {
                    borderCollapse: 'collapse',
                    width: '100%',
                    minWidth: 400,
                  },
                }}
              >
                <table>{children}</table>
              </Box>
            )
          },
          th({ children }) {
            return (
              <Box
                component="th"
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  p: 1,
                  backgroundColor: 'rgba(0, 0, 0, 0.2)',
                  fontWeight: 600,
                  textAlign: 'left',
                }}
              >
                {children}
              </Box>
            )
          },
          td({ children }) {
            return (
              <Box
                component="td"
                sx={{
                  border: '1px solid',
                  borderColor: 'divider',
                  p: 1,
                }}
              >
                {children}
              </Box>
            )
          },
          // Links
          a({ href, children }) {
            return (
              <Box
                component="a"
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  color: 'primary.main',
                  textDecoration: 'none',
                  '&:hover': {
                    textDecoration: 'underline',
                  },
                }}
              >
                {children}
              </Box>
            )
          },
          // Horizontal rule
          hr() {
            return (
              <Box
                component="hr"
                sx={{
                  border: 'none',
                  borderTop: '1px solid',
                  borderColor: 'divider',
                  my: 2,
                }}
              />
            )
          },
        }}
      >
        {displayContent}
      </ReactMarkdown>
    </Box>
  )
}
