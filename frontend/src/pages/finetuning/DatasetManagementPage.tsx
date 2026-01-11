import React, { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  Chip,
  Grid,
  Alert,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  alpha,
} from "@mui/material";
import {
  CloudUpload as UploadIcon,
  Storage as DatasetIcon,
  Check as CheckIcon,
  Delete as DeleteIcon,
  Visibility as PreviewIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  ArrowBack as BackIcon,
  Description as FileIcon,
  DataObject as JsonIcon,
  FormatListBulleted as ListIcon,
  CheckCircle as ValidIcon,
  Error as ErrorIcon,
  BarChart as ChartIcon,
  TextFields as TokenIcon,
  Straighten as LengthIcon,
  Category as FormatIcon,
  Assessment as AnalyticsIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

// Types
type Dataset = {
  id: string;
  name: string;
  format: string;
  status: string;
  num_records?: number;
  file_size_bytes?: number;
  created_at?: string;
  validation_errors?: string[];
};

type DatasetPreview = {
  records: Array<Record<string, any>>;
  total: number;
};

type FieldStat = {
  name: string;
  present_count: number;
  total_count: number;
  completeness: number;
  avg_length: number;
  min_length: number;
  max_length: number;
  empty_count: number;
};

type DatasetStats = {
  dataset_id: string;
  total_records: number;
  detected_format: string;
  format_confidence: number;
  format_indicators: Record<string, boolean>;
  tokens: {
    total: number;
    avg_per_record: number;
    min: number;
    max: number;
    distribution: Record<string, number>;
  };
  sequences: {
    avg_length: number;
    min_length: number;
    max_length: number;
    distribution: Record<string, number>;
  };
  fields: FieldStat[];
  validation: {
    is_valid: boolean;
    errors: string[];
  };
};

// Accent colors
const accentColors = {
  primary: "#6366f1",
  success: "#10b981",
  warning: "#f59e0b",
  info: "#06b6d4",
  purple: "#8b5cf6",
  rose: "#f43f5e",
};

// Section Card Component
interface SectionCardProps {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  accentColor?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}

const SectionCard: React.FC<SectionCardProps> = ({
  title,
  subtitle,
  icon,
  accentColor = accentColors.primary,
  children,
  action,
}) => (
  <Card
    sx={{
      position: "relative",
      overflow: "hidden",
      background: "linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)",
      backdropFilter: "blur(12px)",
      border: "1px solid rgba(255, 255, 255, 0.06)",
      borderRadius: 3,
      transition: "all 0.3s ease-in-out",
      "&:hover": {
        borderColor: alpha(accentColor, 0.2),
        boxShadow: `0 8px 32px ${alpha(accentColor, 0.15)}`,
      },
    }}
  >
    <Box
      sx={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        height: 3,
        background: `linear-gradient(90deg, ${accentColor} 0%, ${alpha(accentColor, 0.3)} 100%)`,
      }}
    />
    <CardContent sx={{ p: 2.5 }}>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          {icon && (
            <Box
              sx={{
                width: 40,
                height: 40,
                borderRadius: 2,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                bgcolor: alpha(accentColor, 0.1),
                color: accentColor,
                "& .MuiSvgIcon-root": { fontSize: 22 },
              }}
            >
              {icon}
            </Box>
          )}
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: "1rem", lineHeight: 1.3 }}>
              {title}
            </Typography>
            {subtitle && (
              <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.75rem" }}>
                {subtitle}
              </Typography>
            )}
          </Box>
        </Box>
        {action}
      </Box>
      <Box sx={{ maxWidth: "100%", overflow: "auto" }}>{children}</Box>
    </CardContent>
  </Card>
);

// Stat Box Component
interface StatBoxProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}

const StatBox: React.FC<StatBoxProps> = ({ label, value, icon, color }) => (
  <Box
    sx={{
      p: 2,
      borderRadius: 2,
      bgcolor: alpha(color, 0.08),
      border: `1px solid ${alpha(color, 0.15)}`,
      textAlign: "center",
      transition: "all 0.2s",
      "&:hover": { bgcolor: alpha(color, 0.12), transform: "translateY(-2px)" },
    }}
  >
    <Box sx={{ mb: 1, color, "& .MuiSvgIcon-root": { fontSize: 28 } }}>{icon}</Box>
    <Typography sx={{ fontWeight: 700, fontSize: "1.5rem", color: "text.primary", lineHeight: 1 }}>
      {value}
    </Typography>
    <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.7rem" }}>
      {label}
    </Typography>
  </Box>
);

// Status Chip
const StatusChip: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { color: string; label: string }> = {
    ready: { color: accentColors.success, label: "Ready" },
    processing: { color: accentColors.info, label: "Processing" },
    error: { color: accentColors.rose, label: "Error" },
    pending: { color: accentColors.warning, label: "Pending" },
  };
  const { color, label } = config[status] || { color: "#64748b", label: status };

  return (
    <Chip
      label={label}
      size="small"
      sx={{
        bgcolor: alpha(color, 0.1),
        color: color,
        border: `1px solid ${alpha(color, 0.3)}`,
        fontWeight: 600,
        fontSize: "0.7rem",
        height: 24,
      }}
    />
  );
};

// Histogram Component
interface HistogramProps {
  data: Record<string, number>;
  color: string;
  label: string;
  height?: number;
}

const Histogram: React.FC<HistogramProps> = ({ data, color, label, height = 120 }) => {
  const entries = Object.entries(data);
  if (entries.length === 0) {
    return (
      <Box sx={{ height, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Typography variant="caption" color="text.secondary">No data available</Typography>
      </Box>
    );
  }

  const maxValue = Math.max(...entries.map(([, v]) => v));
  const total = entries.reduce((sum, [, v]) => sum + v, 0);

  return (
    <Box>
      <Typography variant="caption" sx={{ color: "text.secondary", mb: 1, display: "block" }}>
        {label}
      </Typography>
      <Box sx={{ display: "flex", alignItems: "flex-end", gap: 0.5, height }}>
        {entries.map(([bucket, count], idx) => {
          const barHeight = maxValue > 0 ? (count / maxValue) * 100 : 0;
          const percentage = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
          return (
            <Tooltip
              key={bucket}
              title={`${bucket}: ${count} records (${percentage}%)`}
              arrow
            >
              <Box
                sx={{
                  flex: 1,
                  height: `${barHeight}%`,
                  minHeight: 4,
                  bgcolor: alpha(color, 0.6 + (idx % 2) * 0.2),
                  borderRadius: "4px 4px 0 0",
                  transition: "all 0.2s",
                  cursor: "pointer",
                  "&:hover": {
                    bgcolor: color,
                    transform: "scaleY(1.05)",
                  },
                }}
              />
            </Tooltip>
          );
        })}
      </Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", mt: 0.5 }}>
        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.6rem" }}>
          {entries[0]?.[0]?.split("-")[0] || "0"}
        </Typography>
        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.6rem" }}>
          {entries[entries.length - 1]?.[0]?.split("-")[1] || "max"}
        </Typography>
      </Box>
    </Box>
  );
};

// Field Completeness Bar
interface FieldCompletenessProps {
  fields: FieldStat[];
}

const FieldCompleteness: React.FC<FieldCompletenessProps> = ({ fields }) => {
  if (fields.length === 0) {
    return (
      <Box sx={{ py: 2, textAlign: "center" }}>
        <Typography variant="caption" color="text.secondary">No field data</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      {fields.map((field) => {
        const color = field.completeness >= 90 ? accentColors.success :
          field.completeness >= 70 ? accentColors.warning : accentColors.rose;
        return (
          <Box key={field.name}>
            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
              <Typography variant="caption" sx={{ fontWeight: 600, fontFamily: "monospace" }}>
                {field.name}
              </Typography>
              <Typography variant="caption" sx={{ color, fontWeight: 700 }}>
                {field.completeness.toFixed(1)}%
              </Typography>
            </Box>
            <LinearProgress
              variant="determinate"
              value={field.completeness}
              sx={{
                height: 6,
                borderRadius: 3,
                bgcolor: "rgba(255, 255, 255, 0.1)",
                "& .MuiLinearProgress-bar": {
                  borderRadius: 3,
                  bgcolor: color,
                },
              }}
            />
            <Box sx={{ display: "flex", justifyContent: "space-between", mt: 0.5 }}>
              <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.6rem" }}>
                {field.present_count - field.empty_count} / {field.total_count} records
              </Typography>
              <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.6rem" }}>
                avg: {field.avg_length.toFixed(0)} chars
              </Typography>
            </Box>
          </Box>
        );
      })}
    </Box>
  );
};

// Format Detection Badge
interface FormatDetectionProps {
  format: string;
  confidence: number;
  indicators: Record<string, boolean>;
}

const FormatDetection: React.FC<FormatDetectionProps> = ({ format, confidence, indicators }) => {
  const confidenceColor = confidence >= 0.9 ? accentColors.success :
    confidence >= 0.7 ? accentColors.warning : accentColors.rose;

  const presentIndicators = Object.entries(indicators)
    .filter(([, present]) => present)
    .map(([key]) => key.replace("has_", ""));

  return (
    <Box>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
        <Box
          sx={{
            width: 48,
            height: 48,
            borderRadius: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: `linear-gradient(135deg, ${accentColors.purple} 0%, ${alpha(accentColors.purple, 0.6)} 100%)`,
            boxShadow: `0 4px 14px ${alpha(accentColors.purple, 0.4)}`,
          }}
        >
          <FormatIcon sx={{ color: "#fff", fontSize: 24 }} />
        </Box>
        <Box>
          <Typography sx={{ fontWeight: 700, fontSize: "1.1rem", textTransform: "uppercase" }}>
            {format}
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <LinearProgress
              variant="determinate"
              value={confidence * 100}
              sx={{
                width: 60,
                height: 4,
                borderRadius: 2,
                bgcolor: "rgba(255, 255, 255, 0.1)",
                "& .MuiLinearProgress-bar": { bgcolor: confidenceColor },
              }}
            />
            <Typography variant="caption" sx={{ color: confidenceColor, fontWeight: 700 }}>
              {(confidence * 100).toFixed(0)}% confidence
            </Typography>
          </Box>
        </Box>
      </Box>

      <Typography variant="caption" sx={{ color: "text.secondary", mb: 1, display: "block" }}>
        Detected Fields
      </Typography>
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
        {presentIndicators.map((indicator) => (
          <Chip
            key={indicator}
            label={indicator}
            size="small"
            icon={<ValidIcon sx={{ fontSize: "14px !important" }} />}
            sx={{
              height: 22,
              fontSize: "0.65rem",
              bgcolor: alpha(accentColors.success, 0.1),
              color: accentColors.success,
              border: `1px solid ${alpha(accentColors.success, 0.2)}`,
              "& .MuiChip-icon": { color: accentColors.success },
            }}
          />
        ))}
      </Box>
    </Box>
  );
};

export const DatasetManagementPage: React.FC = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loadingStats, setLoadingStats] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<Dataset | null>(null);
  const [showStats, setShowStats] = useState(false);

  const fetchDatasets = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/datasets");
      if (res.ok) {
        const data = await res.json();
        setDatasets(data);
      }
    } catch {
      setError("Failed to load datasets");
    }
  }, []);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  useEffect(() => {
    if (!selectedDataset) {
      setPreview(null);
      setStats(null);
      setShowStats(false);
      return;
    }
    fetch(`/api/v1/finetune/datasets/${selectedDataset.id}/preview?limit=10`)
      .then((res) => res.json())
      .then((data) => setPreview({ records: data.preview || [], total: selectedDataset.num_records || data.preview?.length || 0 }))
      .catch(() => setPreview(null));
  }, [selectedDataset]);

  const fetchStats = useCallback(async () => {
    if (!selectedDataset) return;
    setLoadingStats(true);
    try {
      const res = await fetch(`/api/v1/finetune/datasets/${selectedDataset.id}/stats`);
      if (res.ok) {
        const data = await res.json();
        setStats(data);
        setShowStats(true);
      }
    } catch {
      setError("Failed to load dataset statistics");
    } finally {
      setLoadingStats(false);
    }
  }, [selectedDataset]);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setError(null);
    setActionStatus("Uploading dataset...");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", file.name.replace(/\.[^/.]+$/, ""));

    try {
      const res = await fetch("/api/v1/finetune/datasets", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        setActionStatus("Dataset uploaded successfully!");
        fetchDatasets();
      } else {
        const data = await res.json();
        setError(data.detail || "Upload failed");
        setActionStatus(null);
      }
    } catch {
      setError("Upload failed");
      setActionStatus(null);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith(".json") || file.name.endsWith(".jsonl"))) {
      handleUpload(file);
    } else {
      setError("Please upload a JSON or JSONL file");
    }
  }, []);

  const handleValidate = async (datasetId: string) => {
    setActionStatus("Validating dataset...");
    try {
      const res = await fetch(`/api/v1/finetune/datasets/${datasetId}/validate`, { method: "POST" });
      const data = await res.json();
      if (data.valid) {
        setActionStatus("Validation passed!");
      } else {
        setActionStatus(`Validation failed: ${data.errors?.length || 0} errors found`);
      }
      fetchDatasets();
    } catch {
      setActionStatus("Validation failed");
    }
  };

  const handleDelete = async () => {
    if (!datasetToDelete) return;
    try {
      await fetch(`/api/v1/finetune/datasets/${datasetToDelete.id}`, { method: "DELETE" });
      setSelectedDataset(null);
      setDeleteDialogOpen(false);
      setDatasetToDelete(null);
      fetchDatasets();
      setActionStatus("Dataset deleted");
    } catch {
      setError("Delete failed");
    }
  };

  const formatSize = (bytes?: number) => {
    if (!bytes) return "-";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "-";
    return new Date(dateStr).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <Box
      sx={{
        width: "100%",
        maxWidth: "100vw",
        overflow: "hidden",
        px: { xs: 2, sm: 3, md: 4 },
        py: 3,
        boxSizing: "border-box",
        minHeight: "100%",
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          flexDirection: { xs: "column", sm: "row" },
          justifyContent: "space-between",
          alignItems: { xs: "flex-start", sm: "center" },
          mb: 4,
          gap: 2,
        }}
      >
        <Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
            <Typography
              variant="h1"
              sx={{
                fontWeight: 700,
                fontSize: { xs: "1.5rem", sm: "1.75rem", md: "2rem" },
                lineHeight: 1,
                background: "linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              Dataset Management
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.875rem", maxWidth: 500 }}>
            Upload, validate, and manage your training datasets
          </Typography>
        </Box>

        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<BackIcon />}
            onClick={() => navigate("/finetuning")}
            sx={{
              borderColor: "rgba(255, 255, 255, 0.1)",
              color: "text.secondary",
              "&:hover": { borderColor: accentColors.primary, color: accentColors.primary },
            }}
          >
            Back to Fine-Tuning
          </Button>
          <Tooltip title="Refresh">
            <IconButton
              size="small"
              onClick={fetchDatasets}
              sx={{
                width: 38,
                height: 38,
                borderRadius: 2,
                border: "1px solid rgba(255, 255, 255, 0.08)",
                bgcolor: "rgba(255, 255, 255, 0.03)",
                "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" },
              }}
            >
              <RefreshIcon sx={{ fontSize: 18 }} />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {actionStatus && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setActionStatus(null)}>
          {actionStatus}
        </Alert>
      )}

      {/* Upload Section */}
      <Box sx={{ mb: 4 }}>
        <SectionCard title="Upload Dataset" subtitle="Drag and drop or browse files" icon={<UploadIcon />} accentColor={accentColors.info}>
          <Box
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            sx={{
              border: `2px dashed ${dragOver ? accentColors.success : "rgba(255, 255, 255, 0.15)"}`,
              borderRadius: 3,
              p: 5,
              textAlign: "center",
              bgcolor: dragOver ? alpha(accentColors.success, 0.05) : "rgba(0, 0, 0, 0.2)",
              transition: "all 0.3s ease-in-out",
              cursor: "pointer",
              "&:hover": {
                borderColor: accentColors.info,
                bgcolor: alpha(accentColors.info, 0.05),
              },
            }}
          >
            {uploading ? (
              <Box>
                <LinearProgress sx={{ mb: 2, borderRadius: 1 }} />
                <Typography variant="body1" color="text.secondary">
                  Uploading...
                </Typography>
              </Box>
            ) : (
              <>
                <Box
                  sx={{
                    width: 64,
                    height: 64,
                    borderRadius: 3,
                    bgcolor: alpha(accentColors.info, 0.1),
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    mx: "auto",
                    mb: 2,
                  }}
                >
                  <UploadIcon sx={{ fontSize: 32, color: accentColors.info }} />
                </Box>
                <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>
                  Drop your dataset here
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  or click to browse files
                </Typography>
                <input
                  type="file"
                  accept=".json,.jsonl"
                  style={{ display: "none" }}
                  id="dataset-upload"
                  onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
                />
                <label htmlFor="dataset-upload">
                  <Button variant="outlined" component="span" startIcon={<FileIcon />}>
                    Browse Files
                  </Button>
                </label>
                <Box sx={{ mt: 2, display: "flex", gap: 1, justifyContent: "center", flexWrap: "wrap" }}>
                  {["Alpaca", "ShareGPT", "ChatML", "Completion"].map((format) => (
                    <Chip
                      key={format}
                      label={format}
                      size="small"
                      sx={{
                        bgcolor: "rgba(255, 255, 255, 0.05)",
                        border: "1px solid rgba(255, 255, 255, 0.1)",
                        fontSize: "0.7rem",
                      }}
                    />
                  ))}
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: "block" }}>
                  Supported formats: JSON, JSONL
                </Typography>
              </>
            )}
          </Box>
        </SectionCard>
      </Box>

      {/* Datasets Grid */}
      <Grid container spacing={3}>
        {/* Dataset List */}
        <Grid item xs={12} md={5}>
          <SectionCard
            title="Datasets"
            subtitle={`${datasets.length} datasets`}
            icon={<DatasetIcon />}
            accentColor={accentColors.purple}
          >
            {datasets.length === 0 ? (
              <Box sx={{ p: 4, textAlign: "center" }}>
                <DatasetIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                <Typography color="text.secondary">No datasets uploaded yet</Typography>
              </Box>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                {datasets.map((ds) => (
                  <Box
                    key={ds.id}
                    onClick={() => setSelectedDataset(ds)}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      bgcolor: selectedDataset?.id === ds.id ? alpha(accentColors.success, 0.1) : "rgba(255, 255, 255, 0.02)",
                      border: `1px solid ${selectedDataset?.id === ds.id ? accentColors.success : "rgba(255, 255, 255, 0.06)"}`,
                      cursor: "pointer",
                      transition: "all 0.2s",
                      "&:hover": {
                        bgcolor: alpha(accentColors.primary, 0.08),
                        borderColor: alpha(accentColors.primary, 0.3),
                      },
                    }}
                  >
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {ds.name}
                      </Typography>
                      <StatusChip status={ds.status} />
                    </Box>
                    <Box sx={{ display: "flex", gap: 2, color: "text.secondary", fontSize: "0.75rem" }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <JsonIcon sx={{ fontSize: 14 }} />
                        {ds.format}
                      </Box>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <ListIcon sx={{ fontSize: 14 }} />
                        {ds.num_records ?? "?"} records
                      </Box>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <FileIcon sx={{ fontSize: 14 }} />
                        {formatSize(ds.file_size_bytes)}
                      </Box>
                    </Box>
                  </Box>
                ))}
              </Box>
            )}
          </SectionCard>
        </Grid>

        {/* Dataset Details */}
        <Grid item xs={12} md={7}>
          <SectionCard
            title={selectedDataset ? selectedDataset.name : "Dataset Details"}
            subtitle={selectedDataset ? `Format: ${selectedDataset.format}` : "Select a dataset to view details"}
            icon={<PreviewIcon />}
            accentColor={accentColors.info}
            action={
              selectedDataset && (
                <Box sx={{ display: "flex", gap: 1 }}>
                  <Tooltip title="Validate Dataset">
                    <IconButton
                      size="small"
                      onClick={() => handleValidate(selectedDataset.id)}
                      sx={{
                        bgcolor: alpha(accentColors.success, 0.1),
                        color: accentColors.success,
                        "&:hover": { bgcolor: alpha(accentColors.success, 0.2) },
                      }}
                    >
                      <CheckIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Delete Dataset">
                    <IconButton
                      size="small"
                      onClick={() => {
                        setDatasetToDelete(selectedDataset);
                        setDeleteDialogOpen(true);
                      }}
                      sx={{
                        bgcolor: alpha(accentColors.rose, 0.1),
                        color: accentColors.rose,
                        "&:hover": { bgcolor: alpha(accentColors.rose, 0.2) },
                      }}
                    >
                      <DeleteIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                </Box>
              )
            }
          >
            {selectedDataset ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                {/* Stats */}
                <Grid container spacing={2}>
                  <Grid item xs={4}>
                    <StatBox
                      label="Records"
                      value={selectedDataset.num_records ?? "-"}
                      icon={<ListIcon />}
                      color={accentColors.primary}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <StatBox
                      label="File Size"
                      value={formatSize(selectedDataset.file_size_bytes)}
                      icon={<FileIcon />}
                      color={accentColors.info}
                    />
                  </Grid>
                  <Grid item xs={4}>
                    <StatBox
                      label="Format"
                      value={selectedDataset.format}
                      icon={<JsonIcon />}
                      color={accentColors.purple}
                    />
                  </Grid>
                </Grid>

                {/* Analyze Dataset Button */}
                <Button
                  variant={showStats ? "outlined" : "contained"}
                  startIcon={loadingStats ? <RefreshIcon className="rotating" /> : <AnalyticsIcon />}
                  onClick={fetchStats}
                  disabled={loadingStats}
                  sx={{
                    background: showStats ? "transparent" : `linear-gradient(135deg, ${accentColors.info} 0%, ${accentColors.purple} 100%)`,
                    borderColor: showStats ? accentColors.info : undefined,
                    "&:hover": {
                      background: showStats ? alpha(accentColors.info, 0.1) : undefined,
                    },
                    "@keyframes spin": {
                      "0%": { transform: "rotate(0deg)" },
                      "100%": { transform: "rotate(360deg)" },
                    },
                    "& .rotating": {
                      animation: "spin 1s linear infinite",
                    },
                  }}
                >
                  {loadingStats ? "Analyzing..." : showStats ? "Refresh Statistics" : "Analyze Dataset"}
                </Button>

                {/* Dataset Statistics Visualization */}
                {showStats && stats && (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    {/* Format Detection */}
                    <Box
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        bgcolor: "rgba(0, 0, 0, 0.2)",
                        border: "1px solid rgba(255, 255, 255, 0.06)",
                      }}
                    >
                      <FormatDetection
                        format={stats.detected_format}
                        confidence={stats.format_confidence}
                        indicators={stats.format_indicators}
                      />
                    </Box>

                    {/* Token Stats Summary */}
                    <Grid container spacing={2}>
                      <Grid item xs={3}>
                        <Box
                          sx={{
                            p: 1.5,
                            borderRadius: 2,
                            bgcolor: alpha(accentColors.info, 0.08),
                            border: `1px solid ${alpha(accentColors.info, 0.15)}`,
                            textAlign: "center",
                          }}
                        >
                          <TokenIcon sx={{ fontSize: 20, color: accentColors.info, mb: 0.5 }} />
                          <Typography sx={{ fontWeight: 700, fontSize: "1rem" }}>
                            {(stats.tokens.total / 1000).toFixed(1)}K
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.6rem" }}>
                            Total Tokens
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={3}>
                        <Box
                          sx={{
                            p: 1.5,
                            borderRadius: 2,
                            bgcolor: alpha(accentColors.success, 0.08),
                            border: `1px solid ${alpha(accentColors.success, 0.15)}`,
                            textAlign: "center",
                          }}
                        >
                          <Typography sx={{ fontWeight: 700, fontSize: "1rem", color: accentColors.success }}>
                            {stats.tokens.avg_per_record.toFixed(0)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.6rem" }}>
                            Avg Tokens/Record
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={3}>
                        <Box
                          sx={{
                            p: 1.5,
                            borderRadius: 2,
                            bgcolor: alpha(accentColors.warning, 0.08),
                            border: `1px solid ${alpha(accentColors.warning, 0.15)}`,
                            textAlign: "center",
                          }}
                        >
                          <Typography sx={{ fontWeight: 700, fontSize: "1rem", color: accentColors.warning }}>
                            {stats.tokens.min}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.6rem" }}>
                            Min Tokens
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={3}>
                        <Box
                          sx={{
                            p: 1.5,
                            borderRadius: 2,
                            bgcolor: alpha(accentColors.purple, 0.08),
                            border: `1px solid ${alpha(accentColors.purple, 0.15)}`,
                            textAlign: "center",
                          }}
                        >
                          <Typography sx={{ fontWeight: 700, fontSize: "1rem", color: accentColors.purple }}>
                            {stats.tokens.max}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.6rem" }}>
                            Max Tokens
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>

                    {/* Histograms */}
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Box
                          sx={{
                            p: 2,
                            borderRadius: 2,
                            bgcolor: "rgba(0, 0, 0, 0.2)",
                            border: "1px solid rgba(255, 255, 255, 0.06)",
                          }}
                        >
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                            <TokenIcon sx={{ fontSize: 18, color: accentColors.info }} />
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                              Token Distribution
                            </Typography>
                          </Box>
                          <Histogram
                            data={stats.tokens.distribution}
                            color={accentColors.info}
                            label="Tokens per record"
                            height={100}
                          />
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Box
                          sx={{
                            p: 2,
                            borderRadius: 2,
                            bgcolor: "rgba(0, 0, 0, 0.2)",
                            border: "1px solid rgba(255, 255, 255, 0.06)",
                          }}
                        >
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                            <LengthIcon sx={{ fontSize: 18, color: accentColors.purple }} />
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                              Sequence Length Distribution
                            </Typography>
                          </Box>
                          <Histogram
                            data={stats.sequences.distribution}
                            color={accentColors.purple}
                            label="Characters per record"
                            height={100}
                          />
                        </Box>
                      </Grid>
                    </Grid>

                    {/* Field Completeness */}
                    <Box
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        bgcolor: "rgba(0, 0, 0, 0.2)",
                        border: "1px solid rgba(255, 255, 255, 0.06)",
                      }}
                    >
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                        <ChartIcon sx={{ fontSize: 18, color: accentColors.success }} />
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          Field Completeness
                        </Typography>
                      </Box>
                      <FieldCompleteness fields={stats.fields} />
                    </Box>
                  </Box>
                )}

                {/* Validation Errors */}
                {selectedDataset.validation_errors && selectedDataset.validation_errors.length > 0 && (
                  <Alert
                    severity="error"
                    icon={<ErrorIcon />}
                    sx={{
                      bgcolor: alpha(accentColors.rose, 0.1),
                      border: `1px solid ${alpha(accentColors.rose, 0.3)}`,
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                      Validation Errors ({selectedDataset.validation_errors.length})
                    </Typography>
                    <Box component="ul" sx={{ m: 0, pl: 2, fontSize: "0.8rem" }}>
                      {selectedDataset.validation_errors.slice(0, 5).map((err, i) => (
                        <li key={i}>{err}</li>
                      ))}
                      {selectedDataset.validation_errors.length > 5 && (
                        <li>...and {selectedDataset.validation_errors.length - 5} more</li>
                      )}
                    </Box>
                  </Alert>
                )}

                <Divider sx={{ borderColor: "rgba(255, 255, 255, 0.06)" }} />

                {/* Preview */}
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
                    Data Preview
                  </Typography>
                  {preview && preview.records.length > 0 ? (
                    <Box sx={{ maxHeight: 350, overflow: "auto" }}>
                      {preview.records.map((record, i) => (
                        <Box
                          key={i}
                          sx={{
                            p: 2,
                            mb: 1.5,
                            bgcolor: "rgba(0, 0, 0, 0.3)",
                            borderRadius: 2,
                            border: "1px solid rgba(255, 255, 255, 0.06)",
                          }}
                        >
                          <Typography
                            component="pre"
                            sx={{
                              m: 0,
                              fontSize: "0.75rem",
                              fontFamily: "monospace",
                              whiteSpace: "pre-wrap",
                              wordBreak: "break-word",
                              color: accentColors.success,
                            }}
                          >
                            {JSON.stringify(record, null, 2)}
                          </Typography>
                        </Box>
                      ))}
                      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                        Showing {preview.records.length} of {preview.total} records
                      </Typography>
                    </Box>
                  ) : (
                    <Box sx={{ p: 3, textAlign: "center", color: "text.secondary" }}>
                      <PreviewIcon sx={{ fontSize: 40, opacity: 0.3, mb: 1 }} />
                      <Typography variant="body2">No preview available</Typography>
                    </Box>
                  )}
                </Box>
              </Box>
            ) : (
              <Box sx={{ p: 4, textAlign: "center" }}>
                <PreviewIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                <Typography color="text.secondary">Select a dataset to view details</Typography>
              </Box>
            )}
          </SectionCard>
        </Grid>
      </Grid>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        PaperProps={{
          sx: {
            bgcolor: "rgba(26, 26, 46, 0.98)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            borderRadius: 3,
          },
        }}
      >
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Box
            sx={{
              width: 40,
              height: 40,
              borderRadius: 2,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              bgcolor: alpha(accentColors.rose, 0.1),
              color: accentColors.rose,
            }}
          >
            <WarningIcon />
          </Box>
          Delete Dataset
        </DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete <strong>{datasetToDelete?.name}</strong>? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions sx={{ p: 2.5 }}>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleDelete}
            sx={{
              bgcolor: accentColors.rose,
              "&:hover": { bgcolor: alpha(accentColors.rose, 0.8) },
            }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DatasetManagementPage;
