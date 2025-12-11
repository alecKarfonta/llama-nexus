import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
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
  const [expanded, setExpanded] = useState(false)

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
  // Check for thinking tags in content and extract them
  // Support multiple tag formats: <think>, <thinking>, <thought>
  // Use greedy matching to capture everything between tags
  const thinkingRegex = /<(think|thinking|thought)>([\s\S]*?)<\/\1>/gi
  let thinking = ''
  let displayContent = content
  
  // Extract all thinking blocks
  let match
  const thinkingParts: string[] = []
  while ((match = thinkingRegex.exec(content)) !== null) {
    thinkingParts.push(match[2].trim())
  }
  
  if (thinkingParts.length > 0) {
    thinking = thinkingParts.join('\n\n')
    displayContent = content.replace(/<(think|thinking|thought)>[\s\S]*?<\/\1>/gi, '').trim()
  }
  
  // Handle unclosed thinking tags - if content starts with <think> but no closing tag,
  // treat everything as thinking (the response is still being generated)
  const unclosedThinkMatch = content.match(/^<(think|thinking|thought)>([\s\S]*)$/i)
  if (unclosedThinkMatch && !thinking) {
    thinking = unclosedThinkMatch[2].trim()
    displayContent = '' // Response hasn't started yet
  }
  
  // Use reasoning_content from API if available (takes priority)
  if (reasoning_content) {
    thinking = reasoning_content
  }

  return (
    <Box sx={{ width: '100%' }}>
      {thinking && <ThinkingBlock content={thinking} />}
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
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
