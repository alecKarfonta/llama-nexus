import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  CircularProgress,
  Chip,
  Tabs,
  Tab,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Snackbar,
  Tooltip,
  IconButton,
  TextField,
  Divider,
  Paper,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Build as BuildIcon,
  CheckCircle as CheckIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { apiService } from '@/services/api';
import type { LlamaCppCommit, LlamaCppCommitsResponse } from '@/types/api';

interface LlamaCppCommitSelectorProps {
  onCommitChanged?: (commit: string) => void;
}

export const LlamaCppCommitSelector: React.FC<LlamaCppCommitSelectorProps> = ({
  onCommitChanged
}) => {
  const [commits, setCommits] = useState<LlamaCppCommitsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedCommit, setSelectedCommit] = useState<string>('');
  const [tabValue, setTabValue] = useState(0);
  const [applying, setApplying] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean;
    commit: LlamaCppCommit | null;
  }>({ open: false, commit: null });
  const [customCommit, setCustomCommit] = useState<string>('');
  const [validatingCustom, setValidatingCustom] = useState(false);
  const [customCommitInfo, setCustomCommitInfo] = useState<any>(null);
  const [customCommitError, setCustomCommitError] = useState<string | null>(null);

  useEffect(() => {
    fetchCommits();
  }, []);

  const fetchCommits = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getLlamaCppCommits();
      setCommits(data);
      setSelectedCommit(data.current_commit);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch commits');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyCommit = async (commit: LlamaCppCommit) => {
    setConfirmDialog({ open: true, commit });
  };

  const confirmApplyCommit = async () => {
    const commit = confirmDialog.commit;
    if (!commit) return;

    try {
      setApplying(true);
      setError(null);
      const result = await apiService.applyLlamaCppCommit(commit.tag);
      setSuccess(`${result.message}. ${result.requires_rebuild ? 'Rebuild required.' : ''}`);
      setSelectedCommit(commit.tag);
      onCommitChanged?.(commit.tag);
      
      // Refresh commits to update current status
      await fetchCommits();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply commit');
    } finally {
      setApplying(false);
      setConfirmDialog({ open: false, commit: null });
    }
  };

  const handleRebuild = async () => {
    try {
      setRebuilding(true);
      setError(null);
      const result = await apiService.rebuildLlamaCpp();
      setSuccess('Containers rebuilt successfully');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rebuild containers');
    } finally {
      setRebuilding(false);
    }
  };

  const validateCustomCommit = async (commitId: string) => {
    if (!commitId.trim()) {
      setCustomCommitInfo(null);
      setCustomCommitError(null);
      return;
    }

    try {
      setValidatingCustom(true);
      setCustomCommitError(null);
      const result = await apiService.validateLlamaCppCommit(commitId.trim());
      
      if (result.valid) {
        setCustomCommitInfo(result.commit);
        setCustomCommitError(null);
      } else {
        setCustomCommitInfo(null);
        setCustomCommitError(result.error || 'Invalid commit');
      }
    } catch (err) {
      setCustomCommitInfo(null);
      setCustomCommitError(err instanceof Error ? err.message : 'Validation failed');
    } finally {
      setValidatingCustom(false);
    }
  };

  const handleCustomCommitChange = (value: string) => {
    setCustomCommit(value);
    
    // Debounce validation
    const timeoutId = setTimeout(() => {
      validateCustomCommit(value);
    }, 500);

    return () => clearTimeout(timeoutId);
  };

  const handleApplyCustomCommit = async () => {
    if (!customCommit.trim() || !customCommitInfo) return;

    try {
      setApplying(true);
      setError(null);
      const result = await apiService.applyLlamaCppCommit(customCommit.trim());
      setSuccess(`${result.message}. ${result.requires_rebuild ? 'Rebuild required.' : ''}`);
      setSelectedCommit(customCommit.trim());
      onCommitChanged?.(customCommit.trim());
      
      // Clear custom input
      setCustomCommit('');
      setCustomCommitInfo(null);
      setCustomCommitError(null);
      
      // Refresh commits to update current status
      await fetchCommits();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply commit');
    } finally {
      setApplying(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const renderCommitList = (commitList: LlamaCppCommit[], title: string) => (
    <Box>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      {commitList.map((commit) => (
        <Card 
          key={commit.tag} 
          variant="outlined" 
          sx={{ 
            mb: 2, 
            border: commit.is_current ? '2px solid #4caf50' : undefined,
            position: 'relative'
          }}
        >
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="flex-start">
              <Box flex={1}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Typography variant="subtitle1" fontWeight="bold">
                    {commit.name}
                  </Typography>
                  {commit.is_current && (
                    <Chip 
                      label="Current" 
                      color="success" 
                      size="small" 
                      icon={<CheckIcon />}
                    />
                  )}
                </Box>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Tag: {commit.tag} • Published: {formatDate(commit.published_at)}
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  {commit.body}
                </Typography>
              </Box>
              <Box display="flex" gap={1} ml={2}>
                <Tooltip title="View details">
                  <IconButton size="small">
                    <InfoIcon />
                  </IconButton>
                </Tooltip>
                {!commit.is_current && (
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => handleApplyCommit(commit)}
                    disabled={applying}
                  >
                    Apply
                  </Button>
                )}
              </Box>
            </Box>
          </CardContent>
        </Card>
      ))}
    </Box>
  );

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={200}>
            <CircularProgress />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (!commits) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">Failed to load commits</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      <Card>
        <CardHeader
          title="LlamaCPP Version Management"
          subheader={`Current version: ${commits.current_commit}`}
          action={
            <Box display="flex" gap={1}>
              <Button
                variant="outlined"
                startIcon={<RefreshIcon />}
                onClick={fetchCommits}
                disabled={loading}
              >
                Refresh
              </Button>
              <Button
                variant="contained"
                startIcon={<BuildIcon />}
                onClick={handleRebuild}
                disabled={rebuilding}
                color="warning"
              >
                {rebuilding ? 'Rebuilding...' : 'Rebuild'}
              </Button>
            </Box>
          }
        />
        <CardContent>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {(applying || rebuilding) && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {applying ? 'Applying commit...' : 'Rebuilding containers...'}
              </Typography>
            </Box>
          )}

          {/* Custom Commit Input */}
          <Paper sx={{ p: 3, mb: 3, bgcolor: 'grey.50' }}>
            <Typography variant="h6" gutterBottom>
              Apply Custom Commit
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Enter a specific commit SHA, tag, or branch name to use a custom version.
            </Typography>
            
            <Box display="flex" gap={2} alignItems="flex-start">
              <TextField
                label="Commit ID / Tag / Branch"
                value={customCommit}
                onChange={(e) => handleCustomCommitChange(e.target.value)}
                placeholder="e.g., b6181, main, or full SHA"
                fullWidth
                error={!!customCommitError}
                helperText={customCommitError}
                disabled={applying || validatingCustom}
                InputProps={{
                  endAdornment: validatingCustom ? <CircularProgress size={20} /> : null
                }}
              />
              <Button
                variant="contained"
                onClick={handleApplyCustomCommit}
                disabled={!customCommitInfo || applying || validatingCustom}
                sx={{ minWidth: 120, height: 56 }}
              >
                {applying ? 'Applying...' : 'Apply'}
              </Button>
            </Box>

            {customCommitInfo && (
              <Box sx={{ mt: 2, p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
                <Typography variant="subtitle2" gutterBottom>
                  ✓ Valid Commit Found
                </Typography>
                <Typography variant="body2">
                  <strong>SHA:</strong> {customCommitInfo.short_sha} ({customCommitInfo.sha})
                </Typography>
                <Typography variant="body2">
                  <strong>Author:</strong> {customCommitInfo.author}
                </Typography>
                <Typography variant="body2">
                  <strong>Date:</strong> {formatDate(customCommitInfo.date)}
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  <strong>Message:</strong> {customCommitInfo.message}
                </Typography>
              </Box>
            )}
          </Paper>

          <Divider sx={{ mb: 3 }} />

          <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)} sx={{ mb: 3 }}>
            <Tab label={`Releases (${commits.releases.length})`} />
            <Tab label={`Recent Commits (${commits.recent_commits.length})`} />
          </Tabs>

          {tabValue === 0 && renderCommitList(commits.releases, 'Official Releases')}
          {tabValue === 1 && renderCommitList(commits.recent_commits, 'Recent Commits')}
        </CardContent>
      </Card>

      {/* Confirmation Dialog */}
      <Dialog
        open={confirmDialog.open}
        onClose={() => setConfirmDialog({ open: false, commit: null })}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Confirm Version Change</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Are you sure you want to switch to this version?
          </Typography>
          {confirmDialog.commit && (
            <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                {confirmDialog.commit.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Tag: {confirmDialog.commit.tag}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                {confirmDialog.commit.body}
              </Typography>
            </Box>
          )}
          <Alert severity="warning" sx={{ mt: 2 }}>
            This will update the Dockerfile and require a container rebuild to take effect.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setConfirmDialog({ open: false, commit: null })}
            disabled={applying}
          >
            Cancel
          </Button>
          <Button 
            onClick={confirmApplyCommit} 
            variant="contained"
            disabled={applying}
          >
            {applying ? 'Applying...' : 'Apply'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Success Snackbar */}
      <Snackbar
        open={!!success}
        autoHideDuration={6000}
        onClose={() => setSuccess(null)}
      >
        <Alert onClose={() => setSuccess(null)} severity="success">
          {success}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default LlamaCppCommitSelector;
