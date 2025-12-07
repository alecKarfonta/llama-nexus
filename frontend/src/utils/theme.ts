/**
 * Material-UI Theme Configuration
 * Beautiful modern dark theme with gradients and glass effects
 */

import { createTheme, alpha } from '@mui/material/styles';

// Custom color palette with rich, beautiful colors
const colors = {
  // Primary - Vibrant blue-violet gradient base
  primary: {
    main: '#6366f1',
    light: '#818cf8',
    dark: '#4f46e5',
    50: 'rgba(99, 102, 241, 0.08)',
    100: 'rgba(99, 102, 241, 0.16)',
    200: 'rgba(99, 102, 241, 0.24)',
    gradient: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
  },
  // Secondary - Cyan accent
  secondary: {
    main: '#06b6d4',
    light: '#22d3ee',
    dark: '#0891b2',
    gradient: 'linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%)',
  },
  // Success - Beautiful emerald
  success: {
    main: '#10b981',
    light: '#34d399',
    dark: '#059669',
    50: 'rgba(16, 185, 129, 0.08)',
    100: 'rgba(16, 185, 129, 0.16)',
    200: 'rgba(16, 185, 129, 0.24)',
    gradient: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
  },
  // Warning - Warm amber
  warning: {
    main: '#f59e0b',
    light: '#fbbf24',
    dark: '#d97706',
    50: 'rgba(245, 158, 11, 0.08)',
    100: 'rgba(245, 158, 11, 0.16)',
    200: 'rgba(245, 158, 11, 0.24)',
    gradient: 'linear-gradient(135deg, #f59e0b 0%, #f97316 100%)',
  },
  // Error - Vibrant rose
  error: {
    main: '#ef4444',
    light: '#f87171',
    dark: '#dc2626',
    50: 'rgba(239, 68, 68, 0.08)',
    100: 'rgba(239, 68, 68, 0.16)',
    200: 'rgba(239, 68, 68, 0.24)',
    gradient: 'linear-gradient(135deg, #ef4444 0%, #f43f5e 100%)',
  },
  // Info - Sky blue
  info: {
    main: '#0ea5e9',
    light: '#38bdf8',
    dark: '#0284c7',
    50: 'rgba(14, 165, 233, 0.08)',
    100: 'rgba(14, 165, 233, 0.16)',
    200: 'rgba(14, 165, 233, 0.24)',
    gradient: 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%)',
  },
  // Background colors - Rich dark tones
  background: {
    default: '#0f0f23',
    paper: '#1a1a2e',
    elevated: '#242444',
    glass: 'rgba(30, 30, 60, 0.6)',
    gradient: 'linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%)',
  },
  // Surface colors for cards and containers
  surface: {
    main: '#1e1e3f',
    light: '#2a2a4a',
    dark: '#16162f',
    glass: 'rgba(30, 30, 70, 0.4)',
    border: 'rgba(255, 255, 255, 0.06)',
    borderLight: 'rgba(255, 255, 255, 0.12)',
  },
  // Text colors
  text: {
    primary: '#f1f5f9',
    secondary: '#94a3b8',
    muted: '#64748b',
    disabled: '#475569',
  },
  // Accent colors for highlights
  accent: {
    purple: '#a855f7',
    pink: '#ec4899',
    rose: '#f43f5e',
    orange: '#f97316',
    yellow: '#eab308',
    lime: '#84cc16',
    teal: '#14b8a6',
  },
};

// Glass-morphism styles
export const glassStyles = {
  light: {
    background: 'rgba(255, 255, 255, 0.03)',
    backdropFilter: 'blur(12px)',
    border: '1px solid rgba(255, 255, 255, 0.06)',
  },
  medium: {
    background: 'rgba(255, 255, 255, 0.05)',
    backdropFilter: 'blur(16px)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
  },
  strong: {
    background: 'rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(255, 255, 255, 0.12)',
  },
};

// Gradient presets
export const gradients = {
  primary: colors.primary.gradient,
  secondary: colors.secondary.gradient,
  success: colors.success.gradient,
  warning: colors.warning.gradient,
  error: colors.error.gradient,
  info: colors.info.gradient,
  purple: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
  cyan: 'linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%)',
  emerald: 'linear-gradient(135deg, #10b981 0%, #14b8a6 100%)',
  amber: 'linear-gradient(135deg, #f59e0b 0%, #f97316 100%)',
  rose: 'linear-gradient(135deg, #f43f5e 0%, #ec4899 100%)',
  sunset: 'linear-gradient(135deg, #f97316 0%, #ec4899 50%, #8b5cf6 100%)',
  aurora: 'linear-gradient(135deg, #06b6d4 0%, #10b981 50%, #8b5cf6 100%)',
  midnight: 'linear-gradient(135deg, #1e1e3f 0%, #0f0f23 100%)',
  card: 'linear-gradient(145deg, rgba(30, 30, 63, 0.8) 0%, rgba(26, 26, 46, 0.9) 100%)',
  cardHover: 'linear-gradient(145deg, rgba(42, 42, 74, 0.9) 0%, rgba(30, 30, 63, 0.95) 100%)',
  header: 'linear-gradient(90deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 46, 0.95) 100%)',
  sidebar: 'linear-gradient(180deg, rgba(26, 26, 46, 0.98) 0%, rgba(15, 15, 35, 0.98) 100%)',
};

// Shadow presets
export const shadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.3)',
  glow: (color: string) => `0 0 20px ${alpha(color, 0.3)}, 0 0 40px ${alpha(color, 0.1)}`,
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.3)',
  card: '0 4px 20px rgba(0, 0, 0, 0.3), 0 0 1px rgba(255, 255, 255, 0.05)',
  cardHover: '0 8px 30px rgba(0, 0, 0, 0.4), 0 0 1px rgba(255, 255, 255, 0.1)',
  elevated: '0 12px 40px rgba(0, 0, 0, 0.5), 0 0 1px rgba(255, 255, 255, 0.08)',
};

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: colors.primary.main,
      light: colors.primary.light,
      dark: colors.primary.dark,
      '50': colors.primary[50] as string,
      '100': colors.primary[100] as string,
      '200': colors.primary[200] as string,
    },
    secondary: {
      main: colors.secondary.main,
      light: colors.secondary.light,
      dark: colors.secondary.dark,
    },
    success: {
      main: colors.success.main,
      light: colors.success.light,
      dark: colors.success.dark,
      '50': colors.success[50] as string,
      '100': colors.success[100] as string,
      '200': colors.success[200] as string,
    },
    warning: {
      main: colors.warning.main,
      light: colors.warning.light,
      dark: colors.warning.dark,
      '50': colors.warning[50] as string,
      '100': colors.warning[100] as string,
      '200': colors.warning[200] as string,
    },
    error: {
      main: colors.error.main,
      light: colors.error.light,
      dark: colors.error.dark,
      '50': colors.error[50] as string,
      '100': colors.error[100] as string,
      '200': colors.error[200] as string,
    },
    info: {
      main: colors.info.main,
      light: colors.info.light,
      dark: colors.info.dark,
      '50': colors.info[50] as string,
      '100': colors.info[100] as string,
      '200': colors.info[200] as string,
    },
    background: {
      default: colors.background.default,
      paper: colors.background.paper,
    },
    grey: {
      50: '#f8fafc',
      100: '#f1f5f9',
      200: '#e2e8f0',
      300: '#cbd5e1',
      400: '#94a3b8',
      500: '#64748b',
      600: '#475569',
      700: '#334155',
      800: '#1e293b',
      900: '#0f172a',
    },
    text: {
      primary: colors.text.primary,
      secondary: colors.text.secondary,
      disabled: colors.text.disabled,
    },
    divider: colors.surface.border,
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    h1: {
      fontSize: '2rem',
      fontWeight: 700,
      letterSpacing: '-0.025em',
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '-0.02em',
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.25rem',
      fontWeight: 600,
      letterSpacing: '-0.015em',
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.125rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1rem',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '0.875rem',
      fontWeight: 600,
      lineHeight: 1.5,
    },
    subtitle1: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.5,
      letterSpacing: '0.01em',
    },
    subtitle2: {
      fontSize: '0.875rem',
      fontWeight: 500,
      lineHeight: 1.5,
      letterSpacing: '0.01em',
    },
    body1: {
      fontSize: '0.9375rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.5,
      letterSpacing: '0.02em',
    },
    overline: {
      fontSize: '0.6875rem',
      fontWeight: 600,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      lineHeight: 1.5,
    },
    button: {
      fontSize: '0.875rem',
      fontWeight: 600,
      letterSpacing: '0.02em',
      textTransform: 'none',
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    shadows.sm,
    shadows.sm,
    shadows.md,
    shadows.md,
    shadows.md,
    shadows.lg,
    shadows.lg,
    shadows.lg,
    shadows.lg,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.xl,
    shadows.elevated,
  ],
  transitions: {
    duration: {
      shortest: 150,
      shorter: 200,
      short: 250,
      standard: 300,
      complex: 375,
      enteringScreen: 225,
      leavingScreen: 195,
    },
    easing: {
      easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
      easeOut: 'cubic-bezier(0.0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
      sharp: 'cubic-bezier(0.4, 0, 0.6, 1)',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        '*': {
          boxSizing: 'border-box',
        },
        html: {
          scrollBehavior: 'smooth',
        },
        body: {
          background: `${colors.background.default}`,
          backgroundImage: `
            radial-gradient(ellipse at 20% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 0% 50%, rgba(6, 182, 212, 0.04) 0%, transparent 50%)
          `,
          backgroundAttachment: 'fixed',
          minHeight: '100vh',
          fontSmoothing: 'antialiased',
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale',
        },
        '::-webkit-scrollbar': {
          width: '8px',
          height: '8px',
        },
        '::-webkit-scrollbar-track': {
          background: 'rgba(0, 0, 0, 0.1)',
          borderRadius: '4px',
        },
        '::-webkit-scrollbar-thumb': {
          background: 'rgba(255, 255, 255, 0.1)',
          borderRadius: '4px',
          '&:hover': {
            background: 'rgba(255, 255, 255, 0.15)',
          },
        },
        '::selection': {
          background: alpha(colors.primary.main, 0.3),
          color: colors.text.primary,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: colors.surface.main,
          border: `1px solid ${colors.surface.border}`,
          transition: 'all 0.2s ease-in-out',
        },
        elevation1: {
          boxShadow: shadows.card,
        },
        elevation2: {
          boxShadow: shadows.card,
        },
        elevation3: {
          boxShadow: shadows.cardHover,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: gradients.card,
          backdropFilter: 'blur(12px)',
          border: `1px solid ${colors.surface.border}`,
          boxShadow: shadows.card,
          borderRadius: 16,
          transition: 'all 0.25s ease-in-out',
          '&:hover': {
            background: gradients.cardHover,
            boxShadow: shadows.cardHover,
            borderColor: colors.surface.borderLight,
            transform: 'translateY(-2px)',
          },
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: {
          padding: 20,
          '&:last-child': {
            paddingBottom: 20,
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 10,
          fontWeight: 600,
          fontSize: '0.875rem',
          padding: '10px 20px',
          transition: 'all 0.2s ease-in-out',
        },
        contained: {
          background: gradients.primary,
          boxShadow: `0 4px 14px ${alpha(colors.primary.main, 0.4)}`,
          '&:hover': {
            background: gradients.primary,
            boxShadow: `0 6px 20px ${alpha(colors.primary.main, 0.5)}`,
            transform: 'translateY(-1px)',
          },
          '&:active': {
            transform: 'translateY(0)',
          },
        },
        containedSecondary: {
          background: gradients.secondary,
          boxShadow: `0 4px 14px ${alpha(colors.secondary.main, 0.4)}`,
          '&:hover': {
            background: gradients.secondary,
            boxShadow: `0 6px 20px ${alpha(colors.secondary.main, 0.5)}`,
          },
        },
        containedSuccess: {
          background: gradients.success,
          boxShadow: `0 4px 14px ${alpha(colors.success.main, 0.4)}`,
          '&:hover': {
            background: gradients.success,
            boxShadow: `0 6px 20px ${alpha(colors.success.main, 0.5)}`,
          },
        },
        containedError: {
          background: gradients.error,
          boxShadow: `0 4px 14px ${alpha(colors.error.main, 0.4)}`,
          '&:hover': {
            background: gradients.error,
            boxShadow: `0 6px 20px ${alpha(colors.error.main, 0.5)}`,
          },
        },
        containedWarning: {
          background: gradients.warning,
          boxShadow: `0 4px 14px ${alpha(colors.warning.main, 0.4)}`,
          '&:hover': {
            background: gradients.warning,
            boxShadow: `0 6px 20px ${alpha(colors.warning.main, 0.5)}`,
          },
        },
        outlined: {
          borderColor: colors.surface.borderLight,
          backgroundColor: 'rgba(255, 255, 255, 0.02)',
          '&:hover': {
            borderColor: colors.primary.main,
            backgroundColor: alpha(colors.primary.main, 0.08),
          },
        },
        text: {
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.08),
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.1),
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
          fontSize: '0.75rem',
          height: 26,
          backgroundColor: alpha(colors.primary.main, 0.1),
          border: `1px solid ${alpha(colors.primary.main, 0.2)}`,
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.15),
          },
        },
        colorSuccess: {
          backgroundColor: alpha(colors.success.main, 0.1),
          border: `1px solid ${alpha(colors.success.main, 0.2)}`,
          color: colors.success.light,
        },
        colorWarning: {
          backgroundColor: alpha(colors.warning.main, 0.1),
          border: `1px solid ${alpha(colors.warning.main, 0.2)}`,
          color: colors.warning.light,
        },
        colorError: {
          backgroundColor: alpha(colors.error.main, 0.1),
          border: `1px solid ${alpha(colors.error.main, 0.2)}`,
          color: colors.error.light,
        },
        colorInfo: {
          backgroundColor: alpha(colors.info.main, 0.1),
          border: `1px solid ${alpha(colors.info.main, 0.2)}`,
          color: colors.info.light,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          background: gradients.sidebar,
          backdropFilter: 'blur(20px)',
          borderRight: `1px solid ${colors.surface.border}`,
          boxShadow: shadows.lg,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: gradients.header,
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${colors.surface.border}`,
          boxShadow: shadows.md,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 10,
            backgroundColor: alpha(colors.surface.main, 0.5),
            transition: 'all 0.2s ease-in-out',
            '& fieldset': {
              borderColor: colors.surface.border,
              transition: 'all 0.2s ease-in-out',
            },
            '&:hover fieldset': {
              borderColor: colors.surface.borderLight,
            },
            '&.Mui-focused fieldset': {
              borderColor: colors.primary.main,
              boxShadow: `0 0 0 3px ${alpha(colors.primary.main, 0.1)}`,
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 10,
        },
      },
    },
    MuiMenu: {
      styleOverrides: {
        paper: {
          background: gradients.card,
          backdropFilter: 'blur(16px)',
          border: `1px solid ${colors.surface.border}`,
          borderRadius: 12,
          boxShadow: shadows.elevated,
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          fontSize: '0.875rem',
          padding: '10px 16px',
          borderRadius: 8,
          margin: '2px 4px',
          transition: 'all 0.15s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.1),
          },
          '&.Mui-selected': {
            backgroundColor: alpha(colors.primary.main, 0.15),
            '&:hover': {
              backgroundColor: alpha(colors.primary.main, 0.2),
            },
          },
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          background: gradients.card,
          backdropFilter: 'blur(20px)',
          border: `1px solid ${colors.surface.border}`,
          borderRadius: 16,
          boxShadow: shadows.elevated,
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          minHeight: 44,
        },
        indicator: {
          height: 3,
          borderRadius: '3px 3px 0 0',
          background: gradients.primary,
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          fontSize: '0.875rem',
          minHeight: 44,
          padding: '10px 20px',
          transition: 'all 0.2s ease-in-out',
          '&.Mui-selected': {
            color: colors.primary.light,
          },
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.08),
          },
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: colors.surface.light,
          border: `1px solid ${colors.surface.border}`,
          borderRadius: 8,
          fontSize: '0.75rem',
          padding: '8px 12px',
          boxShadow: shadows.md,
        },
        arrow: {
          color: colors.surface.light,
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          height: 6,
          borderRadius: 3,
          backgroundColor: alpha(colors.primary.main, 0.1),
        },
        bar: {
          borderRadius: 3,
          background: gradients.primary,
        },
      },
    },
    MuiCircularProgress: {
      styleOverrides: {
        circle: {
          strokeLinecap: 'round',
        },
      },
    },
    MuiDivider: {
      styleOverrides: {
        root: {
          borderColor: colors.surface.border,
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          margin: '2px 8px',
          padding: '10px 12px',
          transition: 'all 0.15s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.08),
          },
          '&.Mui-selected': {
            backgroundColor: alpha(colors.primary.main, 0.12),
            '&:hover': {
              backgroundColor: alpha(colors.primary.main, 0.16),
            },
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: '1px solid',
        },
        standardSuccess: {
          backgroundColor: alpha(colors.success.main, 0.1),
          borderColor: alpha(colors.success.main, 0.2),
          color: colors.success.light,
        },
        standardError: {
          backgroundColor: alpha(colors.error.main, 0.1),
          borderColor: alpha(colors.error.main, 0.2),
          color: colors.error.light,
        },
        standardWarning: {
          backgroundColor: alpha(colors.warning.main, 0.1),
          borderColor: alpha(colors.warning.main, 0.2),
          color: colors.warning.light,
        },
        standardInfo: {
          backgroundColor: alpha(colors.info.main, 0.1),
          borderColor: alpha(colors.info.main, 0.2),
          color: colors.info.light,
        },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        root: {
          width: 42,
          height: 24,
          padding: 0,
        },
        switchBase: {
          padding: 2,
          '&.Mui-checked': {
            transform: 'translateX(18px)',
            '& + .MuiSwitch-track': {
              background: gradients.primary,
              opacity: 1,
            },
          },
        },
        thumb: {
          width: 20,
          height: 20,
          boxShadow: shadows.sm,
        },
        track: {
          borderRadius: 12,
          backgroundColor: colors.surface.light,
          opacity: 1,
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        root: {
          height: 6,
        },
        track: {
          background: gradients.primary,
          border: 'none',
        },
        rail: {
          backgroundColor: alpha(colors.primary.main, 0.2),
        },
        thumb: {
          width: 18,
          height: 18,
          backgroundColor: colors.text.primary,
          boxShadow: `0 0 0 4px ${alpha(colors.primary.main, 0.2)}`,
          '&:hover, &.Mui-focusVisible': {
            boxShadow: `0 0 0 6px ${alpha(colors.primary.main, 0.3)}`,
          },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderColor: colors.surface.border,
          padding: '14px 16px',
        },
        head: {
          backgroundColor: alpha(colors.surface.main, 0.5),
          fontWeight: 600,
          fontSize: '0.8125rem',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: colors.text.secondary,
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          transition: 'background-color 0.15s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.04),
          },
        },
      },
    },
    MuiBadge: {
      styleOverrides: {
        badge: {
          fontWeight: 600,
          fontSize: '0.625rem',
        },
        colorPrimary: {
          background: gradients.primary,
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          background: 'transparent',
          border: `1px solid ${colors.surface.border}`,
          borderRadius: '12px !important',
          '&:before': {
            display: 'none',
          },
          '&.Mui-expanded': {
            margin: 0,
          },
        },
      },
    },
    MuiAccordionSummary: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          '&:hover': {
            backgroundColor: alpha(colors.primary.main, 0.04),
          },
        },
      },
    },
  },
});

// Export colors for use in custom components
export { colors };
