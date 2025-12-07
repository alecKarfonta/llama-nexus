/**
 * MarkdownContent Component
 * Renders markdown content with syntax highlighting for code blocks
 */

import React, { memo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Box, Typography, IconButton, Tooltip, Paper } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

interface MarkdownContentProps {
  content: string;
  isUserMessage?: boolean;
}

// Code block with copy button
const CodeBlock: React.FC<{
  language: string;
  value: string;
}> = memo(({ language, value }) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Box sx={{ position: 'relative', my: 2 }}>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: 'rgba(0, 0, 0, 0.3)',
          px: 2,
          py: 0.5,
          borderTopLeftRadius: 4,
          borderTopRightRadius: 4,
        }}
      >
        <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
          {language || 'code'}
        </Typography>
        <Tooltip title={copied ? 'Copied!' : 'Copy code'}>
          <IconButton size="small" onClick={handleCopy} sx={{ color: 'text.secondary' }}>
            <ContentCopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      <SyntaxHighlighter
        style={oneDark}
        language={language || 'text'}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderTopLeftRadius: 0,
          borderTopRightRadius: 0,
          borderBottomLeftRadius: 4,
          borderBottomRightRadius: 4,
          fontSize: '0.85rem',
        }}
      >
        {value}
      </SyntaxHighlighter>
    </Box>
  );
});

CodeBlock.displayName = 'CodeBlock';

// Inline code styling
const InlineCode: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <Box
    component="code"
    sx={{
      backgroundColor: 'rgba(255, 255, 255, 0.1)',
      px: 0.75,
      py: 0.25,
      borderRadius: 0.5,
      fontFamily: 'monospace',
      fontSize: '0.9em',
    }}
  >
    {children}
  </Box>
);

export const MarkdownContent: React.FC<MarkdownContentProps> = memo(({ content, isUserMessage = false }) => {
  return (
    <Box
      sx={{
        '& p': {
          my: 1,
          '&:first-of-type': { mt: 0 },
          '&:last-of-type': { mb: 0 },
        },
        '& ul, & ol': {
          pl: 3,
          my: 1,
        },
        '& li': {
          my: 0.5,
        },
        '& blockquote': {
          borderLeft: '3px solid',
          borderColor: 'primary.main',
          pl: 2,
          ml: 0,
          my: 2,
          color: 'text.secondary',
          fontStyle: 'italic',
        },
        '& h1, & h2, & h3, & h4, & h5, & h6': {
          mt: 2,
          mb: 1,
          fontWeight: 600,
        },
        '& h1': { fontSize: '1.5rem' },
        '& h2': { fontSize: '1.3rem' },
        '& h3': { fontSize: '1.15rem' },
        '& h4': { fontSize: '1rem' },
        '& table': {
          borderCollapse: 'collapse',
          width: '100%',
          my: 2,
        },
        '& th, & td': {
          border: '1px solid',
          borderColor: 'divider',
          px: 2,
          py: 1,
          textAlign: 'left',
        },
        '& th': {
          backgroundColor: 'rgba(255, 255, 255, 0.05)',
          fontWeight: 600,
        },
        '& a': {
          color: 'primary.main',
          textDecoration: 'none',
          '&:hover': {
            textDecoration: 'underline',
          },
        },
        '& hr': {
          border: 'none',
          borderTop: '1px solid',
          borderColor: 'divider',
          my: 2,
        },
        '& img': {
          maxWidth: '100%',
          borderRadius: 1,
        },
      }}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const isInline = !match && !className;
            
            if (isInline) {
              return <InlineCode>{children}</InlineCode>;
            }
            
            return (
              <CodeBlock
                language={match ? match[1] : ''}
                value={String(children).replace(/\n$/, '')}
              />
            );
          },
          p({ children }) {
            return <Typography variant="body1" component="p">{children}</Typography>;
          },
          li({ children }) {
            return <Typography variant="body1" component="li">{children}</Typography>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </Box>
  );
});

MarkdownContent.displayName = 'MarkdownContent';

export default MarkdownContent;
