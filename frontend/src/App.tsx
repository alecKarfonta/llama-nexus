import { Suspense, useState, useEffect, useCallback, useLayoutEffect, useRef } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { Box, useMediaQuery } from '@mui/material'
import { useTheme } from '@mui/material/styles'
import { Header } from '@/components/layout/Header'
import { Sidebar } from '@/components/layout/Sidebar'
import { CommandPalette } from '@/components/common/CommandPalette'
import { PageSkeleton } from '@/components/common/PageSkeleton'
import { ToastProvider } from '@/contexts/ToastContext'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { appPages, appRedirects } from '@/config/navigation'
import { useLocalPreference } from '@/hooks/useLocalPreferences'

const HASH_SCROLL_RETRY_DELAYS_MS = [75, 150, 300, 600, 1000, 1500]

const getHashElementId = (hash: string) => {
  if (!hash) return ''

  try {
    return decodeURIComponent(hash.slice(1))
  } catch {
    return hash.slice(1)
  }
}

function App() {
  const location = useLocation()
  const theme = useTheme()
  const isNarrowShell = useMediaQuery(theme.breakpoints.down('md'))
  const [sidebarOpen, setSidebarOpen] = useState(() => (
    typeof window === 'undefined' ? true : !window.matchMedia('(max-width:899.95px)').matches
  ))
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const contentScrollRef = useRef<HTMLDivElement | null>(null)
  const { value: compactControlSurfaces } = useLocalPreference('compactControlSurfaces')
  const density = compactControlSurfaces ? 'compact' : 'comfortable'
  const isGraphFocusRoute = location.pathname === '/knowledge-graph' || location.pathname.startsWith('/knowledge-graph/')

  const handleSidebarToggle = () => {
    setSidebarOpen((open) => !open)
  }

  useEffect(() => {
    setSidebarOpen(!isNarrowShell && !isGraphFocusRoute)
  }, [isGraphFocusRoute, isNarrowShell])

  // Global keyboard shortcut for Command Palette (Cmd+K / Ctrl+K)
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault()
      setCommandPaletteOpen((prev: boolean) => !prev)
    }
  }, [])

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  useEffect(() => {
    document.documentElement.dataset.density = density

    return () => {
      delete document.documentElement.dataset.density
    }
  }, [density])

  useLayoutEffect(() => {
    const scroller = contentScrollRef.current
    const hashElementId = getHashElementId(location.hash)
    let hashTargetFound = false

    const scrollToTop = () => {
      if (scroller) {
        scroller.scrollTop = 0
        scroller.scrollLeft = 0
      }

      window.scrollTo(0, 0)
    }

    const scrollToHashTarget = () => {
      if (!hashElementId) return false

      const target = document.getElementById(hashElementId)

      if (!target) {
        return false
      }

      target.scrollIntoView({ block: 'start', inline: 'nearest' })
      hashTargetFound = true
      return true
    }

    if (!scrollToHashTarget()) {
      scrollToTop()
    }

    if (!hashElementId) {
      return undefined
    }

    const frame = requestAnimationFrame(() => {
      if (!hashTargetFound) {
        scrollToHashTarget()
      }
    })
    const timeouts = HASH_SCROLL_RETRY_DELAYS_MS.map((delay) =>
      window.setTimeout(() => {
        if (!hashTargetFound) {
          scrollToHashTarget()
        }
      }, delay)
    )

    return () => {
      cancelAnimationFrame(frame)
      timeouts.forEach((timeout) => window.clearTimeout(timeout))
    }
  }, [location.hash, location.pathname, location.search])

  return (
    <ErrorBoundary resetKey={location.key}>
      <ToastProvider>
        <Box
          data-density={density}
          data-graph-focus-shell={isGraphFocusRoute ? 'true' : 'false'}
          sx={{
            display: 'flex',
            height: '100dvh',
            bgcolor: 'background.default',
            overflow: 'hidden',
          }}
        >
          <Header onMenuClick={handleSidebarToggle} onCommandPalette={() => setCommandPaletteOpen(true)} />
          <Sidebar open={sidebarOpen} onToggle={handleSidebarToggle} temporary={isNarrowShell} />

          <Box
            component="main"
            sx={{
              flexGrow: 1,
              flexBasis: 0,
              minWidth: 0,
              pt: '56px',
              boxSizing: 'border-box',
              transition: 'margin-left 0.3s ease-in-out',
              display: 'flex',
              flexDirection: 'column',
              height: '100dvh',
              minHeight: 0,
              width: '100%',
              maxWidth: '100%',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            <Box ref={contentScrollRef} sx={{
              flex: 1,
              minHeight: 0,
              position: 'relative',
              zIndex: 1,
              overflow: 'auto',
              scrollPaddingTop: 12,
            }}>
              <Suspense fallback={<PageSkeleton />}>
                <Routes>
                  {appPages.map((page) => (
                    <Route key={page.id} path={page.path} element={page.element} />
                  ))}
                  {appRedirects.map((redirect) => (
                    <Route key={redirect.path} path={redirect.path} element={redirect.element} />
                  ))}
                </Routes>
              </Suspense>
            </Box>
          </Box>
        </Box>

        {/* Command Palette */}
        <CommandPalette
          open={commandPaletteOpen}
          onClose={() => setCommandPaletteOpen(false)}
        />
      </ToastProvider>
    </ErrorBoundary>
  )
}

export default App
