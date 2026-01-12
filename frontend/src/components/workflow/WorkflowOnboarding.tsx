/**
 * WorkflowOnboarding - First-time user wizard for workflow creation
 */
import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Paper,
  alpha,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Alert,
  Chip,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Psychology as LlmIcon,
  AccountTree as WorkflowIcon,
  Lightbulb as TipIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { apiService } from '@/services/api';
import type { ModelInfo } from '@/types/api';

interface WorkflowOnboardingProps {
  open: boolean;
  onClose: () => void;
  onTemplateSelect: (templateId: string) => void;
  onSkip: () => void;
}

const steps = ['Check Models', 'Choose Template', 'Get Started'];

export const WorkflowOnboarding: React.FC<WorkflowOnboardingProps> = ({
  open,
  onClose,
  onTemplateSelect,
  onSkip,
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      loadModels();
    }
  }, [open]);

  const loadModels = async () => {
    try {
      setLoading(true);
      setError(null);
      const allModels = await apiService.getModels();
      const availableModels = allModels.filter(
        (m) => m.status === 'available' || m.status === 'deployed'
      );
      setModels(availableModels);
    } catch (err: any) {
      console.error('Failed to load models:', err);
      setError(err.message || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleSelectTemplate = (templateId: string) => {
    onTemplateSelect(templateId);
    onClose();
  };

  const handleSkip = () => {
    onSkip();
    onClose();
  };

  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Checking Your Deployed Models
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Let's make sure you have models deployed before creating workflows.
            </Typography>

            {loading ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, py: 3 }}>
                <CircularProgress size={24} />
                <Typography variant="body2" color="text.secondary">
                  Scanning for deployed models...
                </Typography>
              </Box>
            ) : error ? (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            ) : models.length === 0 ? (
              <Alert severity="warning" sx={{ mb: 2 }}>
                No models are currently deployed. You'll need to deploy a model before you can use LLM nodes in your workflows.
              </Alert>
            ) : (
              <Box>
                <Paper
                  elevation={0}
                  sx={{
                    p: 2,
                    bgcolor: alpha('#10b981', 0.1),
                    border: '1px solid',
                    borderColor: alpha('#10b981', 0.2),
                    mb: 2,
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <CheckIcon sx={{ color: '#10b981' }} />
                    <Typography variant="subtitle2" sx={{ color: '#10b981' }}>
                      Great! Found {models.length} deployed {models.length === 1 ? 'model' : 'models'}
                    </Typography>
                  </Box>
                  <List dense>
                    {models.map((model) => (
                      <ListItem key={model.name} sx={{ py: 0.5 }}>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <LlmIcon fontSize="small" sx={{ color: '#6366f1' }} />
                        </ListItemIcon>
                        <ListItemText
                          primary={model.name}
                          secondary={`${model.variant} | ${(model.size / (1024 ** 3)).toFixed(1)}GB`}
                          primaryTypographyProps={{ variant: 'body2' }}
                          secondaryTypographyProps={{ variant: 'caption' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Paper>

                <Paper
                  elevation={0}
                  sx={{
                    p: 2,
                    bgcolor: alpha('#6366f1', 0.05),
                    border: '1px solid',
                    borderColor: alpha('#6366f1', 0.1),
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                    <TipIcon fontSize="small" sx={{ color: '#6366f1', mt: 0.5 }} />
                    <Box>
                      <Typography variant="subtitle2" gutterBottom>
                        Pro Tip
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        These models can be used in LLM Chat nodes to power your workflows. You can always change which model a workflow uses later.
                      </Typography>
                    </Box>
                  </Box>
                </Paper>
              </Box>
            )}
          </Box>
        );

      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Choose a Template to Get Started
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Select a template that matches what you want to build. You can customize it later.
            </Typography>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Paper
                elevation={0}
                sx={{
                  p: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: '#6366f1',
                    bgcolor: alpha('#6366f1', 0.05),
                  },
                }}
                onClick={() => handleSelectTemplate('simple-chat')}
              >
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                  <LlmIcon sx={{ color: '#6366f1', fontSize: 32 }} />
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                      Chat with Your Model
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Simple chat interface with your deployed LLM. Perfect for testing and basic conversations.
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Chip label="Beginner Friendly" size="small" sx={{ bgcolor: alpha('#10b981', 0.1), color: '#10b981' }} />
                      <Chip label="4 nodes" size="small" variant="outlined" />
                    </Box>
                  </Box>
                </Box>
              </Paper>

              <Paper
                elevation={0}
                sx={{
                  p: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: '#8b5cf6',
                    bgcolor: alpha('#8b5cf6', 0.05),
                  },
                }}
                onClick={() => handleSelectTemplate('rag-local-llm')}
              >
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                  <WorkflowIcon sx={{ color: '#8b5cf6', fontSize: 32 }} />
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                      Document Q&A with RAG
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Query documents using semantic search and generate answers with your model. Includes context retrieval.
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Chip label="Recommended" size="small" sx={{ bgcolor: alpha('#f59e0b', 0.1), color: '#f59e0b' }} />
                      <Chip label="6 nodes" size="small" variant="outlined" />
                    </Box>
                  </Box>
                </Box>
              </Paper>

              <Paper
                elevation={0}
                sx={{
                  p: 2,
                  border: '1px solid',
                  borderColor: 'divider',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  '&:hover': {
                    borderColor: '#ec4899',
                    bgcolor: alpha('#ec4899', 0.05),
                  },
                }}
                onClick={() => handleSelectTemplate('multi-model-compare')}
              >
                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                  <WorkflowIcon sx={{ color: '#ec4899', fontSize: 32 }} />
                  <Box sx={{ flex: 1 }}>
                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                      Compare Multiple Models
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Send the same prompt to different models and compare their responses side-by-side.
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <Chip label="Advanced" size="small" sx={{ bgcolor: alpha('#6366f1', 0.1), color: '#6366f1' }} />
                      <Chip label="7 nodes" size="small" variant="outlined" />
                    </Box>
                  </Box>
                </Box>
              </Paper>

              <Button
                variant="text"
                onClick={handleSkip}
                sx={{ alignSelf: 'center', mt: 1 }}
              >
                Start from scratch instead
              </Button>
            </Box>
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              You're All Set!
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Here are some tips to help you build great workflows:
            </Typography>

            <List>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon sx={{ color: '#10b981' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Drag nodes from the left palette onto the canvas"
                  secondary="Click and drag any node type to add it to your workflow"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon sx={{ color: '#10b981' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Connect nodes by dragging from outputs to inputs"
                  secondary="Data flows through your workflow following these connections"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon sx={{ color: '#10b981' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Configure nodes by clicking them"
                  secondary="Use the property panel on the right to set options and select models"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon sx={{ color: '#10b981' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Test your LLM nodes before running"
                  secondary="Use the 'Test Node' button to verify model connections"
                />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckIcon sx={{ color: '#10b981' }} />
                </ListItemIcon>
                <ListItemText
                  primary="Save your work regularly"
                  secondary="Click the save button in the top toolbar"
                />
              </ListItem>
            </List>

            <Paper
              elevation={0}
              sx={{
                p: 2,
                mt: 2,
                bgcolor: alpha('#6366f1', 0.05),
                border: '1px solid',
                borderColor: alpha('#6366f1', 0.1),
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                <TipIcon fontSize="small" sx={{ color: '#6366f1', mt: 0.5 }} />
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Need Help?
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Hover over any node in the palette to see its description and required inputs. The execution panel shows real-time progress when you run your workflow.
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 2,
        },
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <WorkflowIcon sx={{ color: '#6366f1' }} />
          <Typography variant="h6" fontWeight={600}>
            Welcome to Workflow Builder
          </Typography>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {renderStepContent()}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        {activeStep === 0 ? (
          <>
            <Button onClick={onClose}>Skip Tutorial</Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={loading || (models.length === 0 && !error)}
            >
              Continue
            </Button>
          </>
        ) : activeStep === 1 ? (
          <>
            <Button onClick={handleBack}>Back</Button>
            <Button onClick={onClose}>Close</Button>
          </>
        ) : (
          <>
            <Button onClick={handleBack}>Back</Button>
            <Button variant="contained" onClick={onClose}>
              Start Building
            </Button>
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default WorkflowOnboarding;
