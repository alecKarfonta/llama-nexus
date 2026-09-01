import React from 'react'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import type { SyntaxHighlighterProps } from 'react-syntax-highlighter'
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash'
import c from 'react-syntax-highlighter/dist/esm/languages/prism/c'
import cpp from 'react-syntax-highlighter/dist/esm/languages/prism/cpp'
import csharp from 'react-syntax-highlighter/dist/esm/languages/prism/csharp'
import css from 'react-syntax-highlighter/dist/esm/languages/prism/css'
import diff from 'react-syntax-highlighter/dist/esm/languages/prism/diff'
import docker from 'react-syntax-highlighter/dist/esm/languages/prism/docker'
import go from 'react-syntax-highlighter/dist/esm/languages/prism/go'
import ini from 'react-syntax-highlighter/dist/esm/languages/prism/ini'
import java from 'react-syntax-highlighter/dist/esm/languages/prism/java'
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript'
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import jsx from 'react-syntax-highlighter/dist/esm/languages/prism/jsx'
import markdown from 'react-syntax-highlighter/dist/esm/languages/prism/markdown'
import php from 'react-syntax-highlighter/dist/esm/languages/prism/php'
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import ruby from 'react-syntax-highlighter/dist/esm/languages/prism/ruby'
import rust from 'react-syntax-highlighter/dist/esm/languages/prism/rust'
import shellSession from 'react-syntax-highlighter/dist/esm/languages/prism/shell-session'
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql'
import swift from 'react-syntax-highlighter/dist/esm/languages/prism/swift'
import toml from 'react-syntax-highlighter/dist/esm/languages/prism/toml'
import tsx from 'react-syntax-highlighter/dist/esm/languages/prism/tsx'
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript'
import yaml from 'react-syntax-highlighter/dist/esm/languages/prism/yaml'
import oneDark from 'react-syntax-highlighter/dist/esm/styles/prism/one-dark'
import vscDarkPlus from 'react-syntax-highlighter/dist/esm/styles/prism/vsc-dark-plus'

export type CodeHighlighterVariant = 'oneDark' | 'vscDarkPlus'

interface CodeHighlighterProps extends Omit<SyntaxHighlighterProps, 'children' | 'language' | 'style'> {
  code: string
  language?: string
  variant?: CodeHighlighterVariant
}

const syntaxStyles = {
  oneDark,
  vscDarkPlus,
}

const registeredLanguages = {
  bash,
  c,
  cpp,
  csharp,
  css,
  diff,
  docker,
  go,
  ini,
  java,
  javascript,
  json,
  jsx,
  markdown,
  php,
  python,
  ruby,
  rust,
  'shell-session': shellSession,
  sql,
  swift,
  toml,
  tsx,
  typescript,
  yaml,
}

Object.entries(registeredLanguages).forEach(([name, languageDefinition]) => {
  SyntaxHighlighter.registerLanguage(name, languageDefinition)
})

const languageAliases: Record<string, string> = {
  'c++': 'cpp',
  cs: 'csharp',
  dockerfile: 'docker',
  js: 'javascript',
  md: 'markdown',
  py: 'python',
  rb: 'ruby',
  sh: 'bash',
  shell: 'bash',
  ts: 'typescript',
  yml: 'yaml',
  zsh: 'bash',
}

const normalizeLanguage = (language: string) => {
  const normalized = language.trim().toLowerCase()
  return languageAliases[normalized] || normalized || 'text'
}

export const CodeHighlighter: React.FC<CodeHighlighterProps> = ({
  code,
  language = 'text',
  variant = 'oneDark',
  customStyle,
  ...props
}) => (
  <SyntaxHighlighter
    language={normalizeLanguage(language)}
    style={syntaxStyles[variant]}
    customStyle={{
      margin: 0,
      ...customStyle,
    }}
    {...props}
  >
    {code}
  </SyntaxHighlighter>
)
