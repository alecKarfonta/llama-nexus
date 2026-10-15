import React, { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Box,
  alpha,
  Collapse,
} from '@mui/material'
import {
  ChevronRight as ChevronRightIcon,
  Tune as TuneIcon,
  ViewCompact as SimpleIcon,
} from '@mui/icons-material'
import { navigationSections, getNavigationSections } from '@/config/navigation'
import type { NavigationItem } from '@/types'
import { preferenceStorageKeys, readStoredJson, writeStoredJson, getLocalPreference, setLocalPreference, localPreferenceChangeEvent } from '@/utils/preferences'

interface SidebarProps {
  open: boolean
  onToggle: () => void
  temporary?: boolean
}

// Load collapsed sections from localStorage
const loadCollapsedSections = (): Record<string, boolean> => {
  return readStoredJson<Record<string, boolean>>(preferenceStorageKeys.sidebarCollapsedSections, {})
}

export const Sidebar: React.FC<SidebarProps> = ({ open, onToggle, temporary = false }) => {
  const location = useLocation()
  const navigate = useNavigate()
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>(loadCollapsedSections())
  // [2026-10-15] | INFO | SIDEBAR.SIMPLE_MODE | Simple (default) shows core pages;
  // Advanced shows the full navigation. Synced across components via the
  // local-preference change event.
  const [simpleMode, setSimpleMode] = useState<boolean>(() => getLocalPreference('simpleNavigationMode'))

  useEffect(() => {
    const onPreferenceChange = () => setSimpleMode(getLocalPreference('simpleNavigationMode'))
    window.addEventListener(localPreferenceChangeEvent, onPreferenceChange)
    return () => window.removeEventListener(localPreferenceChangeEvent, onPreferenceChange)
  }, [])

  const toggleSimpleMode = () => {
    setLocalPreference('simpleNavigationMode', !simpleMode)
    setSimpleMode(!simpleMode)
  }

  const visibleSections = simpleMode ? getNavigationSections(true) : navigationSections

  // Save collapsed state to localStorage
  useEffect(() => {
    writeStoredJson(preferenceStorageKeys.sidebarCollapsedSections, collapsedSections)
  }, [collapsedSections])

  const toggleSection = (sectionId: string, currentlyCollapsed: boolean) => {
    setCollapsedSections(prev => ({
      ...prev,
      [sectionId]: !currentlyCollapsed
    }))
  }

  const handleNavigation = (path: string) => {
    navigate(path)
    if (temporary && open) {
      onToggle()
    }
  }

  const isExactOrChildPath = (path: string) => (
    location.pathname === path || location.pathname.startsWith(`${path}/`)
  )

  const isItemActive = (item: NavigationItem) => (
    isExactOrChildPath(item.path) ||
    item.subItems?.some((subItem) => isExactOrChildPath(subItem.path)) ||
    false
  )

  const renderNavigationButton = (item: NavigationItem, nested = false) => {
    const isSelected = isItemActive(item)
    const itemColor = item.color || '#6366f1'
    const Icon = item.icon

    return (
      <ListItem key={item.id} disablePadding sx={{ mb: 0.25 }}>
        <ListItemButton
          selected={isSelected}
          onClick={() => handleNavigation(item.path)}
          sx={{
            py: nested ? 0.5 : 0.625,
            pl: nested ? 4 : 1.25,
            pr: 1.25,
            borderRadius: 1.5,
            minHeight: nested ? 30 : 36,
            position: 'relative',
            overflow: 'hidden',
            transition: 'background-color 0.15s ease, color 0.15s ease',
            bgcolor: 'transparent',
            '&:hover': {
              bgcolor: alpha(itemColor, 0.08),
              '& .nav-icon': {
                color: itemColor,
              },
              '& .nav-label': {
                color: 'text.primary',
              },
            },
            '&.Mui-selected': {
              bgcolor: alpha(itemColor, nested ? 0.08 : 0.12),
              '&::before': {
                content: '""',
                position: 'absolute',
                left: nested ? 18 : 0,
                top: '50%',
                transform: 'translateY(-50%)',
                width: nested ? 2 : 3,
                height: nested ? '44%' : '60%',
                bgcolor: itemColor,
                borderRadius: '0 3px 3px 0',
              },
              '& .nav-icon': {
                color: itemColor,
              },
              '& .nav-label': {
                color: 'text.primary',
                fontWeight: 600,
              },
              '&:hover': {
                bgcolor: alpha(itemColor, 0.16),
              },
            },
          }}
        >
          <ListItemIcon
            className="nav-icon"
            sx={{
              minWidth: nested ? 28 : 36,
              color: isSelected ? itemColor : 'text.secondary',
              transition: 'color 0.15s ease',
              '& .MuiSvgIcon-root': {
                fontSize: nested ? '1rem' : '1.25rem',
              },
            }}
          >
            {Icon && <Icon />}
          </ListItemIcon>
          <ListItemText
            className="nav-label"
            primary={item.label}
            primaryTypographyProps={{
              fontSize: nested ? '0.8125rem' : '0.875rem',
              fontWeight: isSelected ? 600 : 500,
              color: isSelected ? 'text.primary' : 'text.secondary',
              lineHeight: 1.3,
              noWrap: true,
              sx: {
                transition: 'all 0.2s ease-in-out',
              },
            }}
          />
        </ListItemButton>
      </ListItem>
    )
  }

  return (
    <Drawer
      data-app-chrome="sidebar"
      variant={temporary ? 'temporary' : 'persistent'}
      anchor="left"
      open={open}
      onClose={temporary ? onToggle : undefined}
      ModalProps={temporary ? { keepMounted: true } : undefined}
      sx={{
        width: temporary ? 0 : open ? 232 : 0,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 232,
          boxSizing: 'border-box',
          bgcolor: 'rgba(15, 15, 35, 0.98)',
          backdropFilter: 'blur(12px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.06)',
          boxShadow: 'none',
        },
      }}
    >
      <Toolbar sx={{ minHeight: '56px !important' }} />

      {/* Sidebar Content */}
      <Box sx={{
        flex: 1,
        overflowY: 'auto',
        overflowX: 'hidden',
        py: 1,
        '&::-webkit-scrollbar': {
          width: 4,
        },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: 2,
        },
      }}>
        {visibleSections.map((section) => {
          const hasActiveItem = section.items.some(isItemActive)
          const hasStoredPreference = Object.prototype.hasOwnProperty.call(collapsedSections, section.id)
          const isCollapsed = hasStoredPreference ? collapsedSections[section.id] : !hasActiveItem
          const isExpanded = !isCollapsed || hasActiveItem

          return (
            <React.Fragment key={section.id}>
              <Box
                onClick={() => toggleSection(section.id, isCollapsed)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  mx: 1,
                  mt: 1.25,
                  mb: 0.5,
                  px: 1,
                  py: 0.25,
                  borderRadius: 1,
                  transition: 'all 0.15s ease',
                  '&:hover': {
                    bgcolor: 'rgba(255, 255, 255, 0.03)',
                  },
                }}
              >
                <Typography
                  sx={{
                    fontSize: '0.6875rem',
                    textTransform: 'uppercase',
                    color: hasActiveItem ? alpha('#fff', 0.8) : alpha('#fff', 0.4),
                    letterSpacing: 0,
                    fontWeight: 600,
                    transition: 'color 0.15s ease',
                  }}
                >
                  {section.title}
                </Typography>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    color: alpha('#fff', 0.3),
                    transition: 'transform 0.2s ease',
                    transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                  }}
                >
                  <ChevronRightIcon sx={{ fontSize: 14 }} />
                </Box>
              </Box>
              <Collapse in={isExpanded} timeout={200}>
                <List sx={{ py: 0, px: 1 }}>
                  {section.items.map((item) => (
                    <React.Fragment key={item.id}>
                      {renderNavigationButton(item)}
                      {item.subItems && (
                        <Collapse in={isItemActive(item)} timeout={180}>
                          <List disablePadding sx={{ mt: 0.25, mb: 0.5 }}>
                            {item.subItems.map((subItem) => renderNavigationButton(subItem, true))}
                          </List>
                        </Collapse>
                      )}
                    </React.Fragment>
                  ))}
                </List>
              </Collapse>
            </React.Fragment>
          )
        })}

      </Box>

      {/* Footer */}
      <Box sx={{
        p: 1.25,
        borderTop: '1px solid rgba(255, 255, 255, 0.06)',
      }}>
        {/* [2026-10-15] | INFO | SIDEBAR.MODE_TOGGLE | Simple/Advanced navigation switch */}
        <ListItemButton
          onClick={toggleSimpleMode}
          sx={{
            borderRadius: 2,
            mb: 1,
            px: 1.5,
            py: 0.75,
            bgcolor: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(255, 255, 255, 0.04)',
            '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.05)' },
          }}
        >
          <ListItemIcon sx={{ minWidth: 32, color: 'text.secondary' }}>
            {simpleMode ? <TuneIcon sx={{ fontSize: 18 }} /> : <SimpleIcon sx={{ fontSize: 18 }} />}
          </ListItemIcon>
          <ListItemText
            primary={simpleMode ? 'Advanced tools' : 'Simple mode'}
            primaryTypographyProps={{
              fontSize: '0.75rem',
              fontWeight: 600,
              color: 'text.secondary',
            }}
          />
        </ListItemButton>
        <Box sx={{
          p: 1,
          borderRadius: 2,
          bgcolor: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid rgba(255, 255, 255, 0.04)',
        }}>
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              display: 'block',
              fontSize: '0.6875rem',
              letterSpacing: 0,
            }}
          >
            Llama Nexus v1.0
          </Typography>
          <Typography
            variant="caption"
            sx={{
              color: alpha('#64748b', 0.6),
              display: 'block',
              fontSize: '0.625rem',
            }}
          >
            ML Model Management
          </Typography>
        </Box>
      </Box>
    </Drawer>
  )
}
