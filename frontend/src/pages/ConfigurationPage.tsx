import React, { useEffect } from 'react';
import {
  Typography,
  Card,
  CardContent,
  Box,
  Button,
  Alert,
} from '@mui/material';
import {
  ArrowForward as ArrowForwardIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';

export const ConfigurationPage: React.FC = () => {
  useEffect(() => {
    // Auto-redirect to Deploy page after a short delay to show the message
    const timer = setTimeout(() => {
      window.location.href = '/deploy';
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  const handleRedirectNow = () => {
    window.location.href = '/deploy';
  };

  return (
    <Box sx={{ 
      p: { xs: 2, sm: 3, md: 4 },
      maxWidth: '100%',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '60vh'
    }}>
      <Card sx={{ 
        maxWidth: 600,
        width: '100%',
        borderRadius: 2, 
        boxShadow: '0 4px 6px 0 rgba(0, 0, 0, 0.1)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        bgcolor: 'background.paper',
        textAlign: 'center'
      }}>
        <CardContent sx={{ p: 4 }}>
          <Box sx={{ mb: 3 }}>
            <SettingsIcon sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography 
              variant="h4" 
              sx={{ 
                fontWeight: 700, 
                color: 'text.primary',
                mb: 1
              }}
            >
              Configuration Moved
            </Typography>
            <Typography 
              variant="h6" 
              color="text.secondary" 
              sx={{ mb: 3 }}
            >
              All configuration options are now available in the Deploy page
            </Typography>
          </Box>

          <Alert 
            severity="info" 
            sx={{ 
              mb: 3,
              textAlign: 'left',
              borderRadius: 1,
              '& .MuiAlert-message': {
                width: '100%'
              }
            }}
          >
            <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
              The Deploy page now includes:
            </Typography>
            <Box component="ul" sx={{ m: 0, pl: 2 }}>
              <li>API Key configuration (Server tab)</li>
              <li>Model selection and parameters</li>
              <li>Sampling configuration</li>
              <li>Performance settings</li>
              <li>Context extension options</li>
              <li>LlamaCPP version management</li>
              <li>Real-time command preview</li>
            </Box>
          </Alert>

          <Button
            variant="contained"
            size="large"
            startIcon={<ArrowForwardIcon />}
            onClick={handleRedirectNow}
            sx={{ 
              borderRadius: 1,
              fontWeight: 600,
              fontSize: '1rem',
              textTransform: 'none',
              px: 4,
              py: 1.5,
              boxShadow: 'none',
              '&:hover': { 
                boxShadow: '0 2px 4px 0 rgba(0, 0, 0, 0.1)' 
              }
            }}
          >
            Go to Deploy Page
          </Button>

          <Typography 
            variant="body2" 
            color="text.secondary" 
            sx={{ mt: 2 }}
          >
            You will be redirected automatically in 3 seconds...
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};