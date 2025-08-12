/**
 * Material-UI Theme Configuration
 */

import { createTheme } from '@mui/material/styles';

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#007acc',
      light: '#4db8ff',
      dark: '#005599',
      '50': '#e6f3ff',
      '100': '#cce7ff',
      '200': '#99cfff',
    },
    secondary: {
      main: '#569cd6',
      light: '#7fb3e0',
      dark: '#3a7bc8',
    },
    success: {
      main: '#4ec9b0',
      light: '#6ed4c0',
      dark: '#2db398',
      '50': '#e6f9f5',
      '100': '#ccf2eb',
      '200': '#99e5d7',
    },
    warning: {
      main: '#dcdcaa',
      light: '#e6e6bb',
      dark: '#c8c888',
      '50': '#fefef5',
      '100': '#fcfceb',
      '200': '#f9f9d7',
    },
    error: {
      main: '#f44747',
      light: '#f66969',
      dark: '#d73a49',
      '50': '#fef2f2',
      '100': '#fee2e2',
      '200': '#fecaca',
    },
    info: {
      main: '#4fc1ff',
      light: '#72d1ff',
      dark: '#2baeff',
      '50': '#f0f9ff',
      '100': '#e0f2fe',
      '200': '#bae6fd',
    },
    background: {
      default: '#1e1e1e',
      paper: '#252526',
    },
    grey: {
      50: '#f9fafb',
      100: '#f3f4f6',
      200: '#e5e7eb',
      300: '#d1d5db',
      400: '#9ca3af',
      500: '#6b7280',
      600: '#4b5563',
      700: '#374151',
      800: '#2d2d30',
      900: '#1e1e1e',
    },
    text: {
      primary: '#cccccc',
      secondary: '#9ca3af',
    },
  },
  typography: {
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h1: {
      fontSize: '1.875rem',
      fontWeight: 700,
      letterSpacing: '-0.025em',
    },
    h2: {
      fontSize: '1.5rem',
      fontWeight: 600,
      letterSpacing: '-0.02em',
    },
    h3: {
      fontSize: '1.25rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h4: {
      fontSize: '1.125rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h5: {
      fontSize: '1rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '0.875rem',
      fontWeight: 600,
      color: '#9ca3af',
    },
    body1: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
    body2: {
      fontSize: '0.8125rem',
      lineHeight: 1.5,
    },
    caption: {
      fontSize: '0.75rem',
      lineHeight: 1.4,
    },
    button: {
      fontSize: '0.8125rem',
      fontWeight: 500,
      letterSpacing: '0.01em',
    },
  },
  shape: {
    borderRadius: 2,
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#1e1e1e',
          minHeight: '100vh',
          fontSmoothing: 'antialiased',
          WebkitFontSmoothing: 'antialiased',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
          borderRadius: 3,
          border: '1px solid rgba(255, 255, 255, 0.1)',
          backgroundColor: '#252526',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 3,
          fontWeight: 500,
          fontSize: '0.8125rem',
          padding: '6px 12px',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
          },
        },
        outlined: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          fontWeight: 500,
          fontSize: '0.6875rem',
          height: 20,
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#252526',
          color: '#cccccc',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            fontSize: '0.875rem',
            '& input': {
              padding: '8px 12px',
            },
          },
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          fontSize: '0.875rem',
          padding: '6px 12px',
        },
      },
    },
  },
});
