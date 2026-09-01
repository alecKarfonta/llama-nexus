import React, { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  Button,
  IconButton,
  Chip,
  Grid,
  Alert,
  Tooltip,
  Collapse,
  LinearProgress,
  CircularProgress,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useMediaQuery,
  alpha,
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import {
  CloudUpload as UploadIcon,
  Storage as DatasetIcon,
  Delete as DeleteIcon,
  Visibility as PreviewIcon,
  Refresh as RefreshIcon,
  Warning as WarningIcon,
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
  RocketLaunch as TrainIcon,
  ExpandLess as ExpandLessIcon,
  ExpandMore as ExpandMoreIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { trainingDatasetHandoffFromLibraryDataset } from "@/types/trainingHandoff";
import { getRoutePath } from "@/config/navigation";
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from "@/components/common";
import { accentColors } from "@/utils/theme";

// Types
type Dataset = {
  id: string;
  name: string;
  format: string;
  status: string;
  num_examples?: number;
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

const DATASET_STATUS_CONFIG: StatusChipConfigMap = {
  actions: { color: accentColors.info, label: "Actions" },
  inspecting: { color: accentColors.info, label: "Inspecting" },
  library: { color: accentColors.purple, label: "Dataset library" },
  selected: { color: accentColors.success, label: "Selected" },
  selecting: { color: accentColors.warning, label: "Upload or select" },
  uploading: { color: accentColors.warning, label: "Uploading" },
};

const datasetPanelSx = {
  borderRadius: "8px",
  bgcolor: "background.paper",
  border: "1px solid",
  borderColor: "divider",
} as const;

const datasetActionButtonSx = {
  width: 32,
  height: 32,
  borderRadius: "8px",
  border: "1px solid",
  borderColor: "divider",
  bgcolor: "action.hover",
  "&:hover": {
    bgcolor: "action.selected",
  },
} as const;

const previewValueToText = (value: any) => {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "string") return value.replace(/\s+/g, " ").trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === "string" ? item : JSON.stringify(item)))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
  }
  return JSON.stringify(value).replace(/\s+/g, " ").trim();
};

const clampPreviewText = (value: string, limit = 180) => (
  value.length > limit ? `${value.slice(0, limit).trim()}...` : value
);

const getPreviewCell = (
  record: Record<string, any>,
  fields: string[],
  limit: number,
  excludedField?: string
) => {
  for (const field of fields) {
    if (field === excludedField || !Object.prototype.hasOwnProperty.call(record, field)) continue;
    const value = previewValueToText(record[field]);
    if (value !== "-") return { field, value: clampPreviewText(value, limit) };
  }
  return { field: "", value: "-" };
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
                  bgcolor: "action.hover",
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
                  bgcolor: "action.hover",
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
  const theme = useTheme();
  const compactPreview = useMediaQuery(theme.breakpoints.down("sm"));
  const hasInlineDatasetDetails = useMediaQuery(theme.breakpoints.up("md"));
  const trainingPath = getRoutePath("finetuning");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loadingStats, setLoadingStats] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [datasetToDelete, setDatasetToDelete] = useState<Dataset | null>(null);
  const [showStats, setShowStats] = useState(false);
  const [mobilePreviewOpen, setMobilePreviewOpen] = useState(false);

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
    if (datasets.length === 0) {
      if (selectedDataset) setSelectedDataset(null);
      return;
    }

    if (!selectedDataset || !datasets.some((dataset) => dataset.id === selectedDataset.id)) {
      setSelectedDataset(datasets[0]);
    }
  }, [datasets, selectedDataset]);

  useEffect(() => {
    if (!selectedDataset) {
      setPreview(null);
      setStats(null);
      setShowStats(false);
      return;
    }
    fetch(`/api/v1/finetune/datasets/${selectedDataset.id}/preview?limit=10`)
      .then((res) => res.json())
      .then((data) => setPreview({ records: data.preview || [], total: selectedDataset.num_examples || data.preview?.length || 0 }))
      .catch(() => setPreview(null));
    setMobilePreviewOpen(false);
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

  const handleTrainDataset = (dataset: Dataset) => {
    navigate(trainingPath, {
      state: {
        dataset: trainingDatasetHandoffFromLibraryDataset(dataset),
      },
    });
  };

  const formatSize = (bytes?: number) => {
    if (!bytes) return "-";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  };

  const totalRecords = datasets.reduce((total, dataset) => total + (dataset.num_examples || 0), 0);
  const totalBytes = datasets.reduce((total, dataset) => total + (dataset.file_size_bytes || 0), 0);
  const readyDatasets = datasets.filter((dataset) => dataset.status === "ready").length;
  const invalidDatasets = datasets.filter((dataset) => (dataset.validation_errors?.length || 0) > 0 || dataset.status === "error").length;
  const uniqueFormats = new Set(datasets.map((dataset) => dataset.format).filter(Boolean)).size;
  const datasetSummaryParts = [
    totalRecords > 0 ? `${totalRecords.toLocaleString()} records` : null,
    uniqueFormats > 0 ? `${uniqueFormats} format${uniqueFormats === 1 ? "" : "s"}` : null,
    totalBytes > 0 ? formatSize(totalBytes) : null,
  ].filter(Boolean);
  const selectedDatasetNeedsValidation = Boolean(
    selectedDataset && (
      selectedDataset.status !== "ready" ||
      (selectedDataset.validation_errors?.length || 0) > 0
    )
  );
  const previewRecordLimit = compactPreview ? 3 : 10;
  const previewRecords = preview?.records.slice(0, previewRecordLimit) ?? [];
  const previewExpanded = !compactPreview || mobilePreviewOpen;
  const selectedDatasetMetrics = selectedDataset ? [
    { label: "Records", value: selectedDataset.num_examples?.toLocaleString() ?? "-", icon: <ListIcon sx={{ fontSize: 15, color: accentColors.primary }} /> },
    { label: "Size", value: formatSize(selectedDataset.file_size_bytes), icon: <FileIcon sx={{ fontSize: 15, color: accentColors.info }} /> },
    { label: "Format", value: selectedDataset.format, icon: <JsonIcon sx={{ fontSize: 15, color: accentColors.purple }} /> },
  ] : [];

  return (
    <PageShell sx={{ py: 1 }}>
      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {actionStatus && (
        <Alert severity="success" sx={{ mb: 1.25 }} onClose={() => setActionStatus(null)}>
          {actionStatus}
        </Alert>
      )}

      {/* Datasets Grid */}
      <Grid container spacing={1.25}>
        {/* Dataset List */}
        <Grid item xs={12} md={3}>
          <SectionCard
            title="Datasets"
            icon={<DatasetIcon />}
            accentColor={accentColors.purple}
            action={(
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.35 }}>
                <Tooltip title={uploading ? "Uploading dataset" : "Upload dataset"}>
                  <IconButton
                    aria-label={uploading ? "Uploading dataset" : "Upload dataset"}
                    size="small"
                    component="label"
                    disabled={uploading}
                    sx={{
                      ...datasetActionButtonSx,
                      bgcolor: "primary.main",
                      color: "primary.contrastText",
                      borderColor: "primary.main",
                      "&:hover": {
                        bgcolor: "primary.dark",
                      },
                    }}
                  >
                    {uploading ? <CircularProgress size={15} color="inherit" /> : <UploadIcon sx={{ fontSize: 18 }} />}
                    <input
                      type="file"
                      accept=".json,.jsonl"
                      hidden
                      onChange={(e) => e.target.files?.[0] && handleUpload(e.target.files[0])}
                    />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Refresh datasets">
                  <IconButton
                    aria-label="Refresh datasets"
                    size="small"
                    onClick={fetchDatasets}
                    sx={datasetActionButtonSx}
                  >
                    <RefreshIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            )}
          >
            {(datasets.length > 0 || datasetSummaryParts.length > 0) && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.6, flexWrap: "wrap", mb: 1 }}>
                {datasets.length > 0 && (
                  <StatusChip status="library" label={`${datasets.length} total`} statusConfig={DATASET_STATUS_CONFIG} height={22} />
                )}
                {readyDatasets > 0 && (
                  <StatusChip status="selected" label={`${readyDatasets} ready`} statusConfig={DATASET_STATUS_CONFIG} height={22} />
                )}
                {invalidDatasets > 0 && (
                  <StatusChip
                    status="uploading"
                    label={`${invalidDatasets} issue${invalidDatasets === 1 ? "" : "s"}`}
                    statusConfig={DATASET_STATUS_CONFIG}
                    height={22}
                  />
                )}
                {datasetSummaryParts.length > 0 && (
                  <Typography variant="caption" color="text.secondary" sx={{ minWidth: 0 }}>
                    {datasetSummaryParts.join(" / ")}
                  </Typography>
                )}
              </Box>
            )}
            {datasets.length === 0 ? (
              <Box sx={{ p: 2, textAlign: "center" }}>
                <Typography color="text.secondary">No datasets uploaded yet</Typography>
              </Box>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                {datasets.map((ds) => (
                  <Box
                    key={ds.id}
                    role="option"
                    aria-selected={selectedDataset?.id === ds.id}
                    tabIndex={0}
                    onClick={() => setSelectedDataset(ds)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setSelectedDataset(ds);
                      }
                    }}
                      sx={{
                        p: 1.25,
                        borderRadius: "8px",
                        bgcolor: selectedDataset?.id === ds.id ? alpha(accentColors.success, 0.1) : "background.paper",
                        border: "1px solid",
                        borderColor: selectedDataset?.id === ds.id ? accentColors.success : "divider",
                        cursor: "pointer",
                        transition: "all 0.2s",
                        "&:hover": {
                          bgcolor: alpha(accentColors.primary, 0.08),
                          borderColor: selectedDataset?.id === ds.id ? accentColors.success : alpha(accentColors.primary, 0.3),
                        },
                        "&:hover .dataset-delete-action, &:focus-within .dataset-delete-action": {
                          opacity: 1,
                          pointerEvents: "auto",
                        },
                        "&:focus-visible": {
                          outline: "2px solid",
                          outlineColor: alpha(accentColors.success, 0.7),
                          outlineOffset: 1,
                        },
                      }}
                  >
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 1, mb: 0.75 }}>
                      <Typography variant="subtitle2" noWrap sx={{ fontWeight: 600, minWidth: 0 }}>
                        {ds.name}
                      </Typography>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexShrink: 0 }}>
                        <StatusChip status={ds.status} />
                        {selectedDataset?.id === ds.id && !hasInlineDatasetDetails && (
                          <>
                            <Tooltip title="Train with dataset">
                              <IconButton
                                aria-label={`Train with ${ds.name}`}
                                size="small"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleTrainDataset(ds);
                                }}
                                sx={{ ...datasetActionButtonSx, width: 28, height: 28 }}
                              >
                                <TrainIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title={showStats ? "Refresh dataset analysis" : "Analyze dataset"}>
                              <span>
                                <IconButton
                                  aria-label={showStats ? `Refresh analysis for ${ds.name}` : `Analyze ${ds.name}`}
                                  size="small"
                                  disabled={loadingStats}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    fetchStats();
                                  }}
                                  sx={{ ...datasetActionButtonSx, width: 28, height: 28 }}
                                >
                                  {loadingStats ? <CircularProgress size={14} /> : <AnalyticsIcon sx={{ fontSize: 16 }} />}
                                </IconButton>
                              </span>
                            </Tooltip>
                            {selectedDatasetNeedsValidation && (
                              <Tooltip title="Validate dataset">
                                <IconButton
                                  aria-label={`Validate ${ds.name}`}
                                  size="small"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    handleValidate(ds.id);
                                  }}
                                  sx={{ ...datasetActionButtonSx, width: 28, height: 28 }}
                                >
                                  <ValidIcon sx={{ fontSize: 16 }} />
                                </IconButton>
                              </Tooltip>
                            )}
                          </>
                        )}
                        <Tooltip title="Delete dataset">
                          <IconButton
                            aria-label={`Delete ${ds.name}`}
                            className="dataset-delete-action"
                            size="small"
                            onClick={(event) => {
                              event.stopPropagation();
                              setDatasetToDelete(ds);
                              setDeleteDialogOpen(true);
                            }}
                            sx={{
                              ...datasetActionButtonSx,
                              width: 24,
                              height: 24,
                              opacity: 0,
                              pointerEvents: "none",
                              color: accentColors.rose,
                              "&:hover": {
                                bgcolor: alpha(accentColors.rose, 0.12),
                                borderColor: alpha(accentColors.rose, 0.35),
                              },
                            }}
                          >
                            <DeleteIcon sx={{ fontSize: 15 }} />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1.25, color: "text.secondary", fontSize: "0.75rem" }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <JsonIcon sx={{ fontSize: 14 }} />
                        {ds.format}
                      </Box>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <ListIcon sx={{ fontSize: 14 }} />
                        {ds.num_examples ?? "?"} records
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
        <Grid item xs={12} md={9} sx={{ display: { xs: "none", md: "block" } }}>
          <SectionCard
            title={selectedDataset ? selectedDataset.name : "Dataset Details"}
            subtitle={selectedDataset ? undefined : "Select a dataset to view details"}
            icon={<PreviewIcon />}
            accentColor={accentColors.info}
            action={selectedDataset ? (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <Tooltip title="Train with selected dataset">
                  <IconButton
                    aria-label="Train with selected dataset"
                    size="small"
                    onClick={() => handleTrainDataset(selectedDataset)}
                    sx={datasetActionButtonSx}
                  >
                    <TrainIcon sx={{ fontSize: 18 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title={showStats ? "Refresh dataset analysis" : "Analyze dataset"}>
                  <span>
                    <IconButton
                      aria-label={showStats ? "Refresh dataset analysis" : "Analyze dataset"}
                      size="small"
                      onClick={fetchStats}
                      disabled={loadingStats}
                      sx={datasetActionButtonSx}
                    >
                      {loadingStats ? <CircularProgress size={15} /> : <AnalyticsIcon sx={{ fontSize: 18 }} />}
                    </IconButton>
                  </span>
                </Tooltip>
                {selectedDatasetNeedsValidation && (
                  <Tooltip title="Validate dataset">
                    <IconButton
                      aria-label="Validate dataset"
                      size="small"
                      onClick={() => handleValidate(selectedDataset.id)}
                      sx={datasetActionButtonSx}
                    >
                      <ValidIcon sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
            ) : undefined}
          >
            {selectedDataset ? (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25 }}>
                <Box
                  sx={{
                    ...datasetPanelSx,
                    display: "grid",
                    gridTemplateColumns: { xs: "repeat(3, minmax(0, 1fr))", sm: "repeat(3, minmax(0, 1fr))" },
                    gap: 0.75,
                    px: 1,
                    py: 0.85,
                  }}
                >
                  {selectedDatasetMetrics.map((metric) => (
                    <Box key={metric.label} sx={{ minWidth: 0, display: "flex", alignItems: "center", gap: 0.6 }}>
                      {metric.icon}
                      <Box sx={{ minWidth: 0 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ display: "block", lineHeight: 1.2 }}>
                          {metric.label}
                        </Typography>
                        <Typography variant="body2" noWrap title={String(metric.value)} sx={{ fontWeight: 800 }}>
                          {metric.value}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Box>

                {/* Dataset Statistics Visualization */}
                {showStats && stats && (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    {/* Format Detection */}
                    <Box sx={{ p: 2, ...datasetPanelSx }}>
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
                            borderRadius: "8px",
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
                            borderRadius: "8px",
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
                            borderRadius: "8px",
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
                            borderRadius: "8px",
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
                        <Box sx={{ p: 2, ...datasetPanelSx }}>
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
                        <Box sx={{ p: 2, ...datasetPanelSx }}>
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
                    <Box sx={{ p: 2, ...datasetPanelSx }}>
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

                <Divider sx={{ borderColor: "divider", my: -0.5 }} />

                {/* Preview */}
                <Box>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1, mb: previewExpanded ? 1 : 0 }}>
                    <Box sx={{ display: "flex", alignItems: "baseline", gap: 1, minWidth: 0 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                        Data Preview
                      </Typography>
                      {preview && preview.records.length > 0 && (
                        <Typography variant="caption" color="text.secondary" noWrap>
                          {previewRecords.length} of {preview.total}
                        </Typography>
                      )}
                    </Box>
                    {compactPreview && preview && preview.records.length > 0 && (
                      <Tooltip title={mobilePreviewOpen ? "Hide preview" : "Show preview"}>
                        <IconButton
                          aria-label={mobilePreviewOpen ? "Hide preview" : "Show preview"}
                          size="small"
                          onClick={() => setMobilePreviewOpen((value) => !value)}
                          sx={{ ...datasetActionButtonSx, width: 28, height: 28 }}
                        >
                          {mobilePreviewOpen ? <ExpandLessIcon sx={{ fontSize: 18 }} /> : <ExpandMoreIcon sx={{ fontSize: 18 }} />}
                        </IconButton>
                      </Tooltip>
                    )}
                  </Box>
                  <Collapse in={previewExpanded} unmountOnExit={compactPreview}>
                    {preview && previewRecords.length > 0 ? (
                      <Box
                        sx={{
                          ...datasetPanelSx,
                          overflow: "hidden",
                        }}
                      >
                        <Box
                          sx={{
                            display: { xs: "none", md: "grid" },
                            gridTemplateColumns: "48px minmax(0, 0.95fr) minmax(0, 0.85fr) minmax(0, 1.2fr)",
                            gap: 1,
                            px: 1.25,
                            py: 0.6,
                            borderBottom: "1px solid",
                            borderColor: "divider",
                            color: "text.secondary",
                          }}
                        >
                          {["#", "Prompt", "Input", "Output"].map((label) => (
                            <Typography key={label} variant="caption" sx={{ fontWeight: 800 }}>
                              {label}
                            </Typography>
                          ))}
                        </Box>
                        <Box sx={{ display: "grid" }}>
                          {previewRecords.map((record, i) => {
                            const promptCell = getPreviewCell(record, ["instruction", "prompt", "question", "text", "input"], 150);
                            const inputCell = getPreviewCell(record, ["input", "context"], 130, promptCell.field);
                            const outputCell = getPreviewCell(record, ["output", "response", "completion", "answer", "text"], 190, promptCell.field);
                            const previewCells = [promptCell, inputCell, outputCell];

                            return (
                              <Box
                                key={i}
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: { xs: "1fr", md: "48px minmax(0, 0.95fr) minmax(0, 0.85fr) minmax(0, 1.2fr)" },
                                  gap: { xs: 0.35, md: 1 },
                                  px: 1.25,
                                  py: { xs: 0.65, md: 0.6 },
                                  minWidth: 0,
                                  borderTop: i === 0 ? 0 : "1px solid",
                                  borderColor: "divider",
                                }}
                              >
                                <Typography
                                  variant="caption"
                                  sx={{
                                    fontWeight: 800,
                                    color: accentColors.info,
                                    fontFamily: "monospace",
                                  }}
                                >
                                  #{i + 1}
                                </Typography>
                                {previewCells.map((cell, cellIndex) => (
                                  <Typography
                                    key={`${cell.field}-${cellIndex}`}
                                    variant="body2"
                                    color={cell.value === "-" ? "text.disabled" : "text.primary"}
                                    sx={{
                                      minWidth: 0,
                                      display: { xs: cell.value === "-" ? "none" : "-webkit-box", md: "-webkit-box" },
                                      WebkitLineClamp: { xs: 1, md: 1 },
                                      WebkitBoxOrient: "vertical",
                                      overflow: "hidden",
                                      lineHeight: 1.35,
                                    }}
                                    title={cell.value}
                                  >
                                    {cell.value}
                                  </Typography>
                                ))}
                              </Box>
                            );
                          })}
                        </Box>
                      </Box>
                    ) : (
                      <Box sx={{ p: 2, textAlign: "center", color: "text.secondary" }}>
                        <PreviewIcon sx={{ fontSize: 32, opacity: 0.3, mb: 0.75 }} />
                        <Typography variant="body2">No preview available</Typography>
                      </Box>
                    )}
                  </Collapse>
                </Box>
              </Box>
            ) : (
              <Box sx={{ p: 2, textAlign: "center" }}>
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
            bgcolor: "background.paper",
            backdropFilter: "blur(12px)",
            border: "1px solid",
            borderColor: "divider",
            borderRadius: "8px",
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
    </PageShell>
  );
};

export default DatasetManagementPage;
