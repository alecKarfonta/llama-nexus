import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Grid,
  Slider,
  Switch,
  FormControlLabel,
  Collapse,
  Alert,
  Divider,
  alpha,
  Tabs,
  Tab,
  Dialog,
  DialogContent,
} from "@mui/material";
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Add as AddIcon,
  CloudUpload as UploadIcon,
  Psychology as ModelIcon,
  Storage as DatasetIcon,
  Tune as ConfigIcon,
  CheckCircle as ReviewIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Timeline as TimelineIcon,
  Science as BenchmarkIcon,
  Merge as MergeIcon,
  Download as ExportIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  School as TrainingIcon,
  AutoFixHigh as AdapterIcon,
  Assessment as MetricsIcon,
  Layers as LayersIcon,
  Bolt as BoltIcon,
  TrendingDown as LossIcon,
  RestartAlt as ResumeIcon,
  MonitorHeart as MonitorIcon,
  Close as CloseIcon,
} from "@mui/icons-material";
import { TrainingDashboard } from "@/components/finetuning";

// Types
type FineTuningJob = {
  id: string;
  name: string;
  dataset_id: string;
  base_model: string;
  status: string;
  progress?: number;
  current_loss?: number;
  created_at?: string;
  metrics?: Record<string, any>;
  adapter_path?: string;
};

type Dataset = {
  id: string;
  name: string;
  format: string;
};

type Preset = {
  name: string;
  display_name: string;
  description: string;
  lora_config: any;
  training_config: any;
  qlora_config: any;
};

type MetricsPoint = {
  step: number;
  loss: number;
  timestamp: string;
};

type VRAMEstimate = {
  model_name: string;
  estimated_params_b: number;
  total_gb: number;
  recommended_gpu: string;
  fits_on: string[];
  warning?: string;
};

type BenchmarkInfo = {
  name: string;
  display_name: string;
  description: string;
  metrics: string[];
};

type BenchmarkJob = {
  id: string;
  name: string;
  base_model: string;
  adapter_id?: string;
  status: string;
  progress: number;
  current_benchmark?: string;
  results: Array<{
    benchmark_name: string;
    base_model_score?: number;
    finetuned_score?: number;
    improvement?: number;
    num_samples: number;
  }>;
};

const defaultJob = {
  name: "",
  base_model: "meta-llama/Meta-Llama-3-8B",
  dataset_id: "",
  lora_config: { rank: 32, alpha: 64, dropout: 0.05, target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"] },
  training_config: { learning_rate: 2e-4, batch_size: 4, gradient_accumulation_steps: 4, num_epochs: 3, warmup_steps: 100, max_seq_length: 2048 },
  qlora_config: { enabled: true, bits: 4, quant_type: "nf4", double_quant: true, compute_dtype: "bfloat16" },
};

// Wizard steps
type WizardStep = 0 | 1 | 2 | 3;
const wizardSteps = ["Select Model", "Choose Dataset", "Configure Training", "Review & Start"];

// Accent colors for different sections
const accentColors = {
  primary: "#6366f1",
  success: "#10b981",
  warning: "#f59e0b",
  info: "#06b6d4",
  purple: "#8b5cf6",
  rose: "#f43f5e",
};

// Section Card Component matching DashboardPage style
interface SectionCardProps {
  title: string;
  subtitle?: string;
  icon?: React.ReactNode;
  accentColor?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
  collapsible?: boolean;
  defaultExpanded?: boolean;
}

const SectionCard: React.FC<SectionCardProps> = ({
  title,
  subtitle,
  icon,
  accentColor = accentColors.primary,
  children,
  action,
  collapsible = false,
  defaultExpanded = true,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
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
      {/* Top accent gradient */}
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
        {/* Header */}
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: expanded ? 2.5 : 0 }}>
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
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  fontSize: "1rem",
                  color: "text.primary",
                  lineHeight: 1.3,
                }}
              >
                {title}
              </Typography>
              {subtitle && (
                <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.75rem" }}>
                  {subtitle}
                </Typography>
              )}
            </Box>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            {action}
            {collapsible && (
              <IconButton size="small" onClick={() => setExpanded(!expanded)}>
                {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            )}
          </Box>
        </Box>

        <Collapse in={expanded}>
          <Box sx={{ maxWidth: "100%", overflow: "auto" }}>{children}</Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

// Stat Card for metrics
interface StatBoxProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
  subtitle?: string;
}

const StatBox: React.FC<StatBoxProps> = ({ label, value, icon, color, subtitle }) => (
  <Box
    sx={{
      p: 2,
      borderRadius: 2,
      bgcolor: alpha(color, 0.08),
      border: `1px solid ${alpha(color, 0.15)}`,
      display: "flex",
      alignItems: "center",
      gap: 1.5,
      transition: "all 0.2s",
      "&:hover": {
        bgcolor: alpha(color, 0.12),
        transform: "translateY(-2px)",
      },
    }}
  >
    <Box
      sx={{
        width: 44,
        height: 44,
        borderRadius: 2,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: `linear-gradient(135deg, ${color} 0%, ${alpha(color, 0.7)} 100%)`,
        boxShadow: `0 4px 12px ${alpha(color, 0.4)}`,
        "& .MuiSvgIcon-root": { fontSize: 22, color: "#fff" },
      }}
    >
      {icon}
    </Box>
    <Box>
      <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.7rem", display: "block" }}>
        {label}
      </Typography>
      <Typography sx={{ fontWeight: 700, fontSize: "1.25rem", color: "text.primary", lineHeight: 1.2 }}>
        {value}
      </Typography>
      {subtitle && (
        <Typography variant="caption" sx={{ color: alpha(color, 0.8), fontSize: "0.65rem" }}>
          {subtitle}
        </Typography>
      )}
    </Box>
  </Box>
);

// Loss Chart Component with beautiful styling
const LossChart: React.FC<{ data: MetricsPoint[] }> = ({ data }) => {
  if (data.length < 2) {
    return (
      <Box sx={{ p: 3, textAlign: "center", color: "text.secondary" }}>
        <TimelineIcon sx={{ fontSize: 40, opacity: 0.3, mb: 1 }} />
        <Typography variant="body2">Not enough data for chart</Typography>
      </Box>
    );
  }

  const width = 100;
  const height = 120;
  const padding = { left: 8, right: 8, top: 8, bottom: 8 };

  const losses = data.map((d) => d.loss);
  const steps = data.map((d) => d.step);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const maxStep = Math.max(...steps);

  const scaleX = (step: number) => padding.left + (step / maxStep) * (width - padding.left - padding.right);
  const scaleY = (loss: number) =>
    height - padding.bottom - ((loss - minLoss) / (maxLoss - minLoss + 0.001)) * (height - padding.top - padding.bottom);

  const pathD = data.map((d, i) => `${i === 0 ? "M" : "L"} ${scaleX(d.step)} ${scaleY(d.loss)}`).join(" ");

  // Gradient area fill
  const areaD = `${pathD} L ${scaleX(maxStep)} ${height - padding.bottom} L ${padding.left} ${height - padding.bottom} Z`;

  return (
    <Box sx={{ position: "relative" }}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: 180 }}>
        <defs>
          <linearGradient id="lossGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={accentColors.success} stopOpacity="0.3" />
            <stop offset="100%" stopColor={accentColors.success} stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={accentColors.success} />
            <stop offset="100%" stopColor={accentColors.info} />
          </linearGradient>
        </defs>

        {/* Area fill */}
        <path d={areaD} fill="url(#lossGradient)" />

        {/* Loss curve */}
        <path d={pathD} fill="none" stroke="url(#lineGradient)" strokeWidth="2" strokeLinecap="round" />

        {/* End point dot */}
        <circle
          cx={scaleX(data[data.length - 1].step)}
          cy={scaleY(data[data.length - 1].loss)}
          r="3"
          fill={accentColors.success}
        />
      </svg>

      {/* Min/Max labels */}
      <Box sx={{ display: "flex", justifyContent: "space-between", px: 1, mt: 1 }}>
        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem" }}>
          Loss: {minLoss.toFixed(4)}
        </Typography>
        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem" }}>
          Step: {maxStep}
        </Typography>
      </Box>
    </Box>
  );
};

// Status Chip with proper styling
const StatusChip: React.FC<{ status: string }> = ({ status }) => {
  const getStatusConfig = (s: string) => {
    switch (s.toLowerCase()) {
      case "running":
      case "training":
        return { color: accentColors.info, label: "Training" };
      case "completed":
        return { color: accentColors.success, label: "Completed" };
      case "failed":
        return { color: accentColors.rose, label: "Failed" };
      case "cancelled":
        return { color: accentColors.warning, label: "Cancelled" };
      case "pending":
      case "queued":
        return { color: accentColors.purple, label: "Queued" };
      default:
        return { color: "#64748b", label: status };
    }
  };

  const config = getStatusConfig(status);

  return (
    <Chip
      label={config.label}
      size="small"
      sx={{
        bgcolor: alpha(config.color, 0.1),
        color: config.color,
        border: `1px solid ${alpha(config.color, 0.3)}`,
        fontWeight: 600,
        fontSize: "0.7rem",
        height: 24,
      }}
    />
  );
};

export const FineTuningPage: React.FC = () => {
  const navigate = useNavigate();
  const [jobs, setJobs] = useState<FineTuningJob[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [form, setForm] = useState(defaultJob);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [metricsHistory, setMetricsHistory] = useState<MetricsPoint[]>([]);
  const [adapters, setAdapters] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [showDashboard, setShowDashboard] = useState(false);
  const [vramEstimate, setVramEstimate] = useState<VRAMEstimate | null>(null);
  const [activeStep, setActiveStep] = useState<WizardStep>(0);
  const [showWizard, setShowWizard] = useState(true);
  const [activeTab, setActiveTab] = useState(0);

  // Benchmarks
  const [availableBenchmarks, setAvailableBenchmarks] = useState<BenchmarkInfo[]>([]);
  const [benchmarkJobs, setBenchmarkJobs] = useState<BenchmarkJob[]>([]);
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>(["mmlu", "hellaswag"]);
  const [benchmarkAdapterId, setBenchmarkAdapterId] = useState<string>("");

  // Checkpoints
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>("");

  const fetchJobs = () =>
    fetch("/api/v1/finetune/jobs")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load jobs");
        return res.json();
      })
      .then(setJobs)
      .catch((err) => setError(err.message));

  const fetchBenchmarkJobs = () =>
    fetch("/api/v1/finetune/benchmarks/jobs")
      .then((res) => res.json())
      .then((data) => setBenchmarkJobs(data.jobs || []))
      .catch(() => {});

  useEffect(() => {
    fetchJobs();
    fetch("/api/v1/finetune/datasets")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch datasets");
        return res.json();
      })
      .then((data) => setDatasets(Array.isArray(data) ? data : []))
      .catch(() => {});
    fetch("/api/v1/finetune/adapters")
      .then((res) => res.json())
      .then((data) => setAdapters(data.adapters || []))
      .catch(() => {});
    fetch("/api/v1/finetune/presets")
      .then((res) => res.json())
      .then((data) => setPresets(data.presets || []))
      .catch(() => {});
    fetch("/api/v1/finetune/benchmarks/available")
      .then((res) => res.json())
      .then((data) => setAvailableBenchmarks(data.benchmarks || []))
      .catch(() => {});
    fetchBenchmarkJobs();
    const interval = setInterval(() => {
      fetchJobs();
      fetchBenchmarkJobs();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const applyPreset = (presetName: string) => {
    const preset = presets.find((p) => p.name === presetName);
    if (preset) {
      setForm({
        ...form,
        lora_config: preset.lora_config,
        training_config: preset.training_config,
        qlora_config: preset.qlora_config,
      });
    }
  };

  const handleMerge = async (jobId: string) => {
    setActionStatus("Merging adapter with base model...");
    try {
      const res = await fetch(`/api/v1/finetune/adapters/${jobId}/merge`, { method: "POST" });
      const data = await res.json();
      setActionStatus(data.message || "Merge queued successfully");
    } catch {
      setActionStatus("Merge failed");
    }
  };

  const handleExport = async (jobId: string, quantType: string = "q4_k_m") => {
    setActionStatus("Exporting to GGUF...");
    try {
      const res = await fetch(`/api/v1/finetune/adapters/${jobId}/export?quant_type=${quantType}`, { method: "POST" });
      const data = await res.json();
      setActionStatus(data.message || "Export queued");
    } catch {
      setActionStatus("Export failed");
    }
  };

  const toggleBenchmark = (name: string) => {
    setSelectedBenchmarks((prev) => (prev.includes(name) ? prev.filter((b) => b !== name) : [...prev, name]));
  };

  const handleRunBenchmark = async () => {
    if (selectedBenchmarks.length === 0) {
      setError("Select at least one benchmark");
      return;
    }
    setActionStatus("Creating benchmark job...");
    try {
      const benchmarks = selectedBenchmarks.map((name) => ({
        benchmark_name: name,
        num_samples: 50,
        num_few_shot: 3,
        temperature: 0.0,
      }));
      const res = await fetch("/api/v1/finetune/benchmarks/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Benchmark ${new Date().toISOString().slice(0, 16)}`,
          base_model: form.base_model,
          adapter_id: benchmarkAdapterId || undefined,
          benchmarks,
        }),
      });
      const job = await res.json();
      await fetch(`/api/v1/finetune/benchmarks/jobs/${job.id}/start`, { method: "POST" });
      setActionStatus("Benchmark started successfully");
      fetchBenchmarkJobs();
    } catch {
      setActionStatus("Failed to start benchmark");
    }
  };

  useEffect(() => {
    if (!selectedJob) return;
    fetch(`/api/v1/finetune/jobs/${selectedJob}/logs?limit=100`)
      .then((res) => res.json())
      .then((data) => setLogs(data.logs || []))
      .catch(() => {});
    fetch(`/api/v1/finetune/jobs/${selectedJob}/metrics/history`)
      .then((res) => res.json())
      .then((data) => setMetricsHistory(data.history || []))
      .catch(() => setMetricsHistory([]));
    fetch(`/api/v1/finetune/jobs/${selectedJob}/checkpoints`)
      .then((res) => res.json())
      .then((data) => {
        const ckpts = (data.checkpoints || []).filter((c: string) => c.includes("checkpoint"));
        setCheckpoints(ckpts);
        setSelectedCheckpoint("");
      })
      .catch(() => setCheckpoints([]));
  }, [selectedJob]);

  const handleResumeFromCheckpoint = async () => {
    if (!selectedJob || !selectedCheckpoint) return;
    setActionStatus("Resuming from checkpoint...");
    try {
      const res = await fetch(`/api/v1/finetune/jobs/${selectedJob}/resume`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint_path: selectedCheckpoint }),
      });
      if (res.ok) {
        setActionStatus("Training resumed successfully");
        fetchJobs();
      } else {
        const data = await res.json();
        setActionStatus(`Resume failed: ${data.detail || "unknown error"}`);
      }
    } catch {
      setActionStatus("Resume failed");
    }
  };

  // VRAM estimation
  useEffect(() => {
    const estimate = async () => {
      try {
        const res = await fetch("/api/v1/finetune/estimate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model_name: form.base_model,
            lora_rank: form.lora_config.rank,
            batch_size: form.training_config.batch_size,
            seq_length: form.training_config.max_seq_length,
            qlora_enabled: form.qlora_config.enabled,
            gradient_accumulation: form.training_config.gradient_accumulation_steps,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          setVramEstimate(data);
        }
      } catch {
        setVramEstimate(null);
      }
    };
    const timer = setTimeout(estimate, 500);
    return () => clearTimeout(timer);
  }, [
    form.base_model,
    form.lora_config.rank,
    form.training_config.batch_size,
    form.training_config.max_seq_length,
    form.qlora_config.enabled,
  ]);

  const onSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    setError(null);
    try {
      const res = await fetch("/api/v1/finetune/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error("Failed to create job");
      await fetchJobs();
      setShowWizard(false);
      setActiveStep(0);
      setForm(defaultJob);
      setActionStatus("Training job created successfully!");
    } catch (err: any) {
      setError(err.message);
    }
  };

  const selected = useMemo(() => jobs.find((j) => j.id === selectedJob), [jobs, selectedJob]);

  const canProceed = (step: WizardStep): boolean => {
    switch (step) {
      case 0:
        return !!form.name && !!form.base_model;
      case 1:
        return !!form.dataset_id;
      case 2:
        return true;
      case 3:
        return true;
      default:
        return false;
    }
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
                color: "text.primary",
                fontSize: { xs: "1.5rem", sm: "1.75rem", md: "2rem" },
                lineHeight: 1,
                background: "linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              LoRA Fine-Tuning Studio
            </Typography>
            <Chip
              size="small"
              label="Pro"
              sx={{
                height: 22,
                bgcolor: alpha(accentColors.purple, 0.1),
                border: `1px solid ${alpha(accentColors.purple, 0.2)}`,
                color: accentColors.purple,
                fontWeight: 600,
                fontSize: "0.6875rem",
              }}
            />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.875rem", maxWidth: 500 }}>
            Create, train, and deploy custom LoRA adapters for your language models
          </Typography>
        </Box>

        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<DatasetIcon />}
            onClick={() => navigate("/finetuning/datasets")}
            sx={{
              borderColor: "rgba(255, 255, 255, 0.1)",
              color: "text.secondary",
              "&:hover": { borderColor: accentColors.info, color: accentColors.info },
            }}
          >
            Datasets
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<ModelIcon />}
            onClick={() => navigate("/finetuning/distillation")}
            sx={{
              borderColor: "rgba(255, 255, 255, 0.1)",
              color: "text.secondary",
              "&:hover": { borderColor: accentColors.purple, color: accentColors.purple },
            }}
          >
            Distillation
          </Button>
          <Button
            variant="outlined"
            size="small"
            startIcon={<MetricsIcon />}
            onClick={() => navigate("/finetuning/evaluation")}
            sx={{
              borderColor: "rgba(255, 255, 255, 0.1)",
              color: "text.secondary",
              "&:hover": { borderColor: accentColors.success, color: accentColors.success },
            }}
          >
            Evaluation
          </Button>
          <Tooltip title="Refresh">
            <IconButton
              size="small"
              onClick={fetchJobs}
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

      {/* Main Content Tabs */}
      <Box sx={{ mb: 3 }}>
        <Tabs
          value={activeTab}
          onChange={(_, v) => setActiveTab(v)}
          sx={{
            "& .MuiTab-root": {
              textTransform: "none",
              fontWeight: 600,
              fontSize: "0.875rem",
              minHeight: 44,
            },
          }}
        >
          <Tab icon={<AddIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Create Job" />
          <Tab
            icon={<TrainingIcon sx={{ fontSize: 18 }} />}
            iconPosition="start"
            label={`Training Jobs (${jobs.length})`}
          />
          <Tab
            icon={<AdapterIcon sx={{ fontSize: 18 }} />}
            iconPosition="start"
            label={`Adapters (${adapters.length})`}
          />
          <Tab icon={<BenchmarkIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Benchmarks" />
        </Tabs>
      </Box>

      {/* Tab Panels */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          {/* Create Job Wizard */}
          <Grid item xs={12} lg={8}>
            <SectionCard
              title="Create Training Job"
              subtitle="Configure and start a new LoRA fine-tuning job"
              icon={<AddIcon />}
              accentColor={accentColors.primary}
              action={
                <Button
                  size="small"
                  onClick={() => setShowWizard(!showWizard)}
                  sx={{ textTransform: "none", fontSize: "0.75rem" }}
                >
                  {showWizard ? "Simple Form" : "Use Wizard"}
                </Button>
              }
            >
              {showWizard ? (
                <Box>
                  {/* Stepper */}
                  <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
                    {wizardSteps.map((label, index) => (
                      <Step key={label}>
                        <StepLabel
                          StepIconProps={{
                            sx: {
                              "&.Mui-active": { color: accentColors.primary },
                              "&.Mui-completed": { color: accentColors.success },
                            },
                          }}
                        >
                          {label}
                        </StepLabel>
                      </Step>
                    ))}
                  </Stepper>

                  {/* Step 0: Model Selection */}
                  {activeStep === 0 && (
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
                      <TextField
                        label="Job Name"
                        value={form.name}
                        onChange={(e) => setForm({ ...form, name: e.target.value })}
                        placeholder="e.g., llama3-coding-assistant"
                        fullWidth
                        size="small"
                        helperText="A descriptive name for your training job"
                      />
                      <TextField
                        label="Base Model"
                        value={form.base_model}
                        onChange={(e) => setForm({ ...form, base_model: e.target.value })}
                        placeholder="meta-llama/Meta-Llama-3-8B"
                        fullWidth
                        size="small"
                        helperText="HuggingFace model ID or local path"
                      />
                    </Box>
                  )}

                  {/* Step 1: Dataset Selection */}
                  {activeStep === 1 && (
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Training Dataset</InputLabel>
                        <Select
                          value={form.dataset_id}
                          onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}
                          label="Training Dataset"
                        >
                          <MenuItem value="">
                            <em>Select a dataset...</em>
                          </MenuItem>
                          {datasets.map((d) => (
                            <MenuItem key={d.id} value={d.id}>
                              {d.name} ({d.format})
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      {datasets.length === 0 && (
                        <Alert severity="info" sx={{ mt: 1 }}>
                          No datasets found.{" "}
                          <Button size="small" onClick={() => navigate("/finetuning/datasets")}>
                            Upload a dataset
                          </Button>
                        </Alert>
                      )}
                    </Box>
                  )}

                  {/* Step 2: Configuration */}
                  {activeStep === 2 && (
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Preset</InputLabel>
                        <Select defaultValue="" onChange={(e) => applyPreset(e.target.value as string)} label="Preset">
                          <MenuItem value="">Custom Configuration</MenuItem>
                          {presets.map((p) => (
                            <MenuItem key={p.name} value={p.name}>
                              {p.display_name} - {p.description}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>

                      <Divider sx={{ my: 1 }}>
                        <Chip label="LoRA Configuration" size="small" />
                      </Divider>

                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <TextField
                            label="LoRA Rank"
                            type="number"
                            value={form.lora_config.rank}
                            onChange={(e) =>
                              setForm({ ...form, lora_config: { ...form.lora_config, rank: Number(e.target.value) } })
                            }
                            fullWidth
                            size="small"
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            label="LoRA Alpha"
                            type="number"
                            value={form.lora_config.alpha}
                            onChange={(e) =>
                              setForm({ ...form, lora_config: { ...form.lora_config, alpha: Number(e.target.value) } })
                            }
                            fullWidth
                            size="small"
                          />
                        </Grid>
                      </Grid>

                      <Divider sx={{ my: 1 }}>
                        <Chip label="Training Configuration" size="small" />
                      </Divider>

                      <Grid container spacing={2}>
                        <Grid item xs={6} sm={3}>
                          <TextField
                            label="Batch Size"
                            type="number"
                            value={form.training_config.batch_size}
                            onChange={(e) =>
                              setForm({
                                ...form,
                                training_config: { ...form.training_config, batch_size: Number(e.target.value) },
                              })
                            }
                            fullWidth
                            size="small"
                          />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <TextField
                            label="Epochs"
                            type="number"
                            value={form.training_config.num_epochs}
                            onChange={(e) =>
                              setForm({
                                ...form,
                                training_config: { ...form.training_config, num_epochs: Number(e.target.value) },
                              })
                            }
                            fullWidth
                            size="small"
                          />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <TextField
                            label="Learning Rate"
                            type="number"
                            inputProps={{ step: 0.00001 }}
                            value={form.training_config.learning_rate}
                            onChange={(e) =>
                              setForm({
                                ...form,
                                training_config: { ...form.training_config, learning_rate: Number(e.target.value) },
                              })
                            }
                            fullWidth
                            size="small"
                          />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <TextField
                            label="Max Seq Length"
                            type="number"
                            value={form.training_config.max_seq_length}
                            onChange={(e) =>
                              setForm({
                                ...form,
                                training_config: { ...form.training_config, max_seq_length: Number(e.target.value) },
                              })
                            }
                            fullWidth
                            size="small"
                          />
                        </Grid>
                      </Grid>

                      <FormControlLabel
                        control={
                          <Switch
                            checked={form.qlora_config.enabled}
                            onChange={(e) =>
                              setForm({ ...form, qlora_config: { ...form.qlora_config, enabled: e.target.checked } })
                            }
                            color="primary"
                          />
                        }
                        label={
                          <Box>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              Enable QLoRA (4-bit Quantization)
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Reduces VRAM usage by ~75% with minimal quality loss
                            </Typography>
                          </Box>
                        }
                      />
                    </Box>
                  )}

                  {/* Step 3: Review */}
                  {activeStep === 3 && (
                    <Box>
                      <Grid container spacing={2} sx={{ mb: 3 }}>
                        <Grid item xs={12} sm={6}>
                          <StatBox
                            label="Model"
                            value={form.base_model.split("/").pop() || form.base_model}
                            icon={<ModelIcon />}
                            color={accentColors.primary}
                          />
                        </Grid>
                        <Grid item xs={12} sm={6}>
                          <StatBox
                            label="Dataset"
                            value={datasets.find((d) => d.id === form.dataset_id)?.name || "-"}
                            icon={<DatasetIcon />}
                            color={accentColors.info}
                          />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <StatBox label="LoRA Rank" value={form.lora_config.rank} icon={<LayersIcon />} color={accentColors.purple} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <StatBox label="Batch Size" value={form.training_config.batch_size} icon={<SpeedIcon />} color={accentColors.success} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <StatBox label="Epochs" value={form.training_config.num_epochs} icon={<TimelineIcon />} color={accentColors.warning} />
                        </Grid>
                        <Grid item xs={6} sm={3}>
                          <StatBox
                            label="QLoRA"
                            value={form.qlora_config.enabled ? "ON" : "OFF"}
                            icon={<BoltIcon />}
                            color={form.qlora_config.enabled ? accentColors.success : "#64748b"}
                          />
                        </Grid>
                      </Grid>

                      {vramEstimate && (
                        <Alert
                          severity={vramEstimate.warning ? "warning" : "info"}
                          icon={<MemoryIcon />}
                          sx={{ mb: 2 }}
                        >
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            Estimated VRAM: {vramEstimate.total_gb} GB
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Recommended GPU: {vramEstimate.recommended_gpu} | Model: ~{vramEstimate.estimated_params_b}B
                            params
                          </Typography>
                          {vramEstimate.warning && (
                            <Typography variant="caption" sx={{ display: "block", color: "warning.main", mt: 0.5 }}>
                              {vramEstimate.warning}
                            </Typography>
                          )}
                        </Alert>
                      )}
                    </Box>
                  )}

                  {/* Navigation Buttons */}
                  <Box sx={{ display: "flex", justifyContent: "space-between", mt: 4 }}>
                    <Button disabled={activeStep === 0} onClick={() => setActiveStep((s) => (s - 1) as WizardStep)}>
                      Back
                    </Button>
                    <Box sx={{ display: "flex", gap: 1 }}>
                      {activeStep < 3 ? (
                        <Button
                          variant="contained"
                          onClick={() => setActiveStep((s) => (s + 1) as WizardStep)}
                          disabled={!canProceed(activeStep)}
                          sx={{
                            background: `linear-gradient(135deg, ${accentColors.primary} 0%, ${accentColors.purple} 100%)`,
                          }}
                        >
                          Next
                        </Button>
                      ) : (
                        <Button
                          variant="contained"
                          onClick={() => onSubmit()}
                          startIcon={<StartIcon />}
                          sx={{
                            background: `linear-gradient(135deg, ${accentColors.success} 0%, ${accentColors.info} 100%)`,
                          }}
                        >
                          Start Training
                        </Button>
                      )}
                    </Box>
                  </Box>
                </Box>
              ) : (
                /* Simple Form */
                <Box component="form" onSubmit={onSubmit} sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Job Name"
                        value={form.name}
                        onChange={(e) => setForm({ ...form, name: e.target.value })}
                        fullWidth
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        label="Base Model"
                        value={form.base_model}
                        onChange={(e) => setForm({ ...form, base_model: e.target.value })}
                        fullWidth
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Dataset</InputLabel>
                        <Select
                          value={form.dataset_id}
                          onChange={(e) => setForm({ ...form, dataset_id: e.target.value })}
                          label="Dataset"
                        >
                          {datasets.map((d) => (
                            <MenuItem key={d.id} value={d.id}>
                              {d.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <TextField
                        label="LoRA Rank"
                        type="number"
                        value={form.lora_config.rank}
                        onChange={(e) =>
                          setForm({ ...form, lora_config: { ...form.lora_config, rank: Number(e.target.value) } })
                        }
                        fullWidth
                        size="small"
                      />
                    </Grid>
                    <Grid item xs={6} sm={3}>
                      <TextField
                        label="Epochs"
                        type="number"
                        value={form.training_config.num_epochs}
                        onChange={(e) =>
                          setForm({
                            ...form,
                            training_config: { ...form.training_config, num_epochs: Number(e.target.value) },
                          })
                        }
                        fullWidth
                        size="small"
                      />
                    </Grid>
                  </Grid>
                  <Button
                    type="submit"
                    variant="contained"
                    startIcon={<StartIcon />}
                    sx={{
                      alignSelf: "flex-start",
                      background: `linear-gradient(135deg, ${accentColors.success} 0%, ${accentColors.info} 100%)`,
                    }}
                  >
                    Start Training
                  </Button>
                </Box>
              )}
            </SectionCard>
          </Grid>

          {/* VRAM Estimate Sidebar */}
          <Grid item xs={12} lg={4}>
            <SectionCard title="Resource Estimate" subtitle="Based on current configuration" icon={<MemoryIcon />} accentColor={accentColors.info}>
              {vramEstimate ? (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <StatBox label="VRAM Required" value={`${vramEstimate.total_gb} GB`} icon={<MemoryIcon />} color={accentColors.info} />
                  <StatBox
                    label="Recommended GPU"
                    value={vramEstimate.recommended_gpu}
                    icon={<SpeedIcon />}
                    color={accentColors.success}
                  />
                  <StatBox
                    label="Model Size"
                    value={`~${vramEstimate.estimated_params_b}B`}
                    icon={<ModelIcon />}
                    color={accentColors.purple}
                  />
                  {vramEstimate.warning && (
                    <Alert severity="warning" sx={{ mt: 1 }}>
                      {vramEstimate.warning}
                    </Alert>
                  )}
                </Box>
              ) : (
                <Box sx={{ p: 3, textAlign: "center", color: "text.secondary" }}>
                  <MemoryIcon sx={{ fontSize: 40, opacity: 0.3, mb: 1 }} />
                  <Typography variant="body2">Configure model to see estimates</Typography>
                </Box>
              )}
            </SectionCard>
          </Grid>
        </Grid>
      )}

      {/* Training Jobs Tab */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} lg={8}>
            <SectionCard
              title="Training Jobs"
              subtitle={`${jobs.length} jobs total`}
              icon={<TrainingIcon />}
              accentColor={accentColors.success}
            >
              {jobs.length === 0 ? (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <TrainingIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                  <Typography color="text.secondary">No training jobs yet</Typography>
                  <Button sx={{ mt: 2 }} variant="outlined" onClick={() => setActiveTab(0)}>
                    Create Your First Job
                  </Button>
                </Box>
              ) : (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Name</TableCell>
                        <TableCell>Model</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell align="right">Progress</TableCell>
                        <TableCell align="right">Loss</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {jobs.map((job) => (
                        <TableRow
                          key={job.id}
                          hover
                          selected={selectedJob === job.id}
                          onClick={() => setSelectedJob(job.id)}
                          sx={{ cursor: "pointer" }}
                        >
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {job.name}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="caption" color="text.secondary">
                              {job.base_model.split("/").pop()}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <StatusChip status={job.status} />
                          </TableCell>
                          <TableCell align="right">
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1, justifyContent: "flex-end" }}>
                              <LinearProgress
                                variant="determinate"
                                value={job.progress ?? 0}
                                sx={{
                                  width: 60,
                                  height: 6,
                                  borderRadius: 3,
                                  bgcolor: "rgba(255,255,255,0.1)",
                                  "& .MuiLinearProgress-bar": {
                                    background: `linear-gradient(90deg, ${accentColors.success}, ${accentColors.info})`,
                                  },
                                }}
                              />
                              <Typography variant="caption" sx={{ minWidth: 36 }}>
                                {job.progress ?? 0}%
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2" sx={{ fontFamily: "monospace", color: accentColors.success }}>
                              {job.current_loss?.toFixed(4) ?? "-"}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </SectionCard>
          </Grid>

          {/* Job Details */}
          <Grid item xs={12} lg={4}>
            <SectionCard
              title={selected ? selected.name : "Job Details"}
              subtitle={selected ? `Status: ${selected.status}` : "Select a job to view details"}
              icon={<MetricsIcon />}
              accentColor={accentColors.purple}
            >
              {selected ? (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {/* Live Dashboard Button for Running Jobs */}
                  {(selected.status === "training" || selected.status === "running") && (
                    <Button
                      variant="contained"
                      fullWidth
                      startIcon={<MonitorIcon />}
                      onClick={() => setShowDashboard(true)}
                      sx={{
                        background: `linear-gradient(135deg, ${accentColors.info} 0%, ${accentColors.purple} 100%)`,
                        py: 1.5,
                        fontWeight: 700,
                        boxShadow: `0 4px 14px ${alpha(accentColors.info, 0.4)}`,
                        "&:hover": {
                          boxShadow: `0 6px 20px ${alpha(accentColors.info, 0.5)}`,
                        },
                      }}
                    >
                      Open Live Dashboard
                    </Button>
                  )}

                  <Box>
                    <Typography variant="overline" color="text.secondary" sx={{ fontSize: "0.65rem" }}>
                      Training Progress
                    </Typography>
                    <LossChart data={metricsHistory} />
                  </Box>

                  <Divider />

                  <Box>
                    <Typography variant="overline" color="text.secondary" sx={{ fontSize: "0.65rem", mb: 1, display: "block" }}>
                      Training Logs
                    </Typography>
                    <Box
                      sx={{
                        bgcolor: "rgba(0,0,0,0.3)",
                        p: 1.5,
                        borderRadius: 1,
                        maxHeight: 150,
                        overflow: "auto",
                        fontFamily: "monospace",
                        fontSize: "0.7rem",
                        color: accentColors.success,
                      }}
                    >
                      {logs.length > 0 ? logs.slice(-20).join("") : "No logs available"}
                    </Box>
                  </Box>

                  {/* Checkpoint Resume */}
                  {(selected.status === "failed" || selected.status === "cancelled") && checkpoints.length > 0 && (
                    <Box
                      sx={{
                        p: 2,
                        bgcolor: alpha(accentColors.warning, 0.1),
                        border: `1px solid ${alpha(accentColors.warning, 0.2)}`,
                        borderRadius: 2,
                      }}
                    >
                      <Typography variant="subtitle2" sx={{ mb: 1.5, display: "flex", alignItems: "center", gap: 1 }}>
                        <ResumeIcon sx={{ fontSize: 18 }} /> Resume from Checkpoint
                      </Typography>
                      <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                        <Select
                          value={selectedCheckpoint}
                          onChange={(e) => setSelectedCheckpoint(e.target.value)}
                          displayEmpty
                        >
                          <MenuItem value="">Select checkpoint...</MenuItem>
                          {checkpoints.map((ckpt) => (
                            <MenuItem key={ckpt} value={ckpt}>
                              {ckpt.split("/").pop()}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <Button
                        variant="contained"
                        size="small"
                        startIcon={<ResumeIcon />}
                        onClick={handleResumeFromCheckpoint}
                        disabled={!selectedCheckpoint}
                        fullWidth
                        sx={{ bgcolor: accentColors.warning }}
                      >
                        Resume Training
                      </Button>
                    </Box>
                  )}
                </Box>
              ) : (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <MetricsIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                  <Typography color="text.secondary">Select a job to view metrics</Typography>
                </Box>
              )}
            </SectionCard>
          </Grid>
        </Grid>
      )}

      {/* Adapters Tab */}
      {activeTab === 2 && (
        <SectionCard
          title="Trained Adapters"
          subtitle="Manage your LoRA adapters"
          icon={<AdapterIcon />}
          accentColor={accentColors.purple}
          action={
            adapters.length >= 2 && (
              <Button
                variant="outlined"
                size="small"
                startIcon={<MetricsIcon />}
                onClick={() => navigate("/finetuning/compare")}
                sx={{
                  borderColor: accentColors.purple,
                  color: accentColors.purple,
                  "&:hover": { bgcolor: alpha(accentColors.purple, 0.1) },
                }}
              >
                Compare Adapters
              </Button>
            )
          }
        >
          {adapters.length === 0 ? (
            <Box sx={{ p: 4, textAlign: "center" }}>
              <AdapterIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
              <Typography color="text.secondary">No adapters yet. Complete a training job to create one.</Typography>
            </Box>
          ) : (
            <Grid container spacing={2}>
              {adapters.map((adapter) => (
                <Grid item xs={12} sm={6} md={4} key={adapter.job_id}>
                  <Card
                    sx={{
                      bgcolor: "rgba(255,255,255,0.02)",
                      border: "1px solid rgba(255,255,255,0.06)",
                      borderRadius: 2,
                    }}
                  >
                    <CardContent>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", mb: 2 }}>
                        <Box>
                          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                            {adapter.name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {adapter.base_model}
                          </Typography>
                        </Box>
                        <StatusChip status={adapter.status} />
                      </Box>

                      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                        <Tooltip title="Merge LoRA weights into base model">
                          <span>
                            <Button
                              size="small"
                              startIcon={<MergeIcon />}
                              onClick={() => handleMerge(adapter.job_id)}
                              disabled={adapter.status !== "completed"}
                              sx={{ fontSize: "0.7rem" }}
                            >
                              Merge
                            </Button>
                          </span>
                        </Tooltip>
                        <Tooltip title="Export merged model to GGUF format">
                          <span>
                            <Button
                              size="small"
                              startIcon={<ExportIcon />}
                              onClick={() => handleExport(adapter.job_id)}
                              disabled={adapter.status !== "completed"}
                              sx={{ fontSize: "0.7rem" }}
                            >
                              Export GGUF
                            </Button>
                          </span>
                        </Tooltip>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </SectionCard>
      )}

      {/* Benchmarks Tab */}
      {activeTab === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12} lg={5}>
            <SectionCard
              title="Run Benchmarks"
              subtitle="Evaluate your fine-tuned models"
              icon={<BenchmarkIcon />}
              accentColor={accentColors.info}
            >
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
                <FormControl fullWidth size="small">
                  <InputLabel>Adapter to Evaluate</InputLabel>
                  <Select
                    value={benchmarkAdapterId}
                    onChange={(e) => setBenchmarkAdapterId(e.target.value)}
                    label="Adapter to Evaluate"
                  >
                    <MenuItem value="">Base Model Only</MenuItem>
                    {jobs
                      .filter((j) => j.status === "completed" && j.adapter_path)
                      .map((j) => (
                        <MenuItem key={j.id} value={j.id}>
                          {j.name} ({j.base_model.split("/").pop()})
                        </MenuItem>
                      ))}
                  </Select>
                </FormControl>

                <Box>
                  <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
                    Select Benchmarks
                  </Typography>
                  <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                    {availableBenchmarks.map((b) => (
                      <Tooltip key={b.name} title={b.description}>
                        <Chip
                          label={b.display_name}
                          onClick={() => toggleBenchmark(b.name)}
                          sx={{
                            bgcolor: selectedBenchmarks.includes(b.name) ? alpha(accentColors.success, 0.2) : "rgba(255,255,255,0.05)",
                            color: selectedBenchmarks.includes(b.name) ? accentColors.success : "text.secondary",
                            border: `1px solid ${selectedBenchmarks.includes(b.name) ? accentColors.success : "rgba(255,255,255,0.1)"}`,
                            fontWeight: 600,
                            "&:hover": {
                              bgcolor: alpha(accentColors.success, 0.15),
                            },
                          }}
                        />
                      </Tooltip>
                    ))}
                  </Box>
                </Box>

                <Button
                  variant="contained"
                  startIcon={<BenchmarkIcon />}
                  onClick={handleRunBenchmark}
                  disabled={selectedBenchmarks.length === 0}
                  sx={{
                    background: `linear-gradient(135deg, ${accentColors.info} 0%, ${accentColors.purple} 100%)`,
                  }}
                >
                  Run Benchmarks
                </Button>
              </Box>
            </SectionCard>
          </Grid>

          <Grid item xs={12} lg={7}>
            <SectionCard
              title="Benchmark Results"
              subtitle={`${benchmarkJobs.length} benchmark jobs`}
              icon={<MetricsIcon />}
              accentColor={accentColors.success}
            >
              {benchmarkJobs.length === 0 ? (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <BenchmarkIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                  <Typography color="text.secondary">No benchmark results yet</Typography>
                </Box>
              ) : (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {benchmarkJobs.map((bj) => (
                    <Box
                      key={bj.id}
                      sx={{
                        p: 2,
                        bgcolor: "rgba(255,255,255,0.02)",
                        border: "1px solid rgba(255,255,255,0.06)",
                        borderRadius: 2,
                      }}
                    >
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1.5 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          {bj.name}
                        </Typography>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <StatusChip status={bj.status} />
                          {bj.status === "running" && (
                            <Typography variant="caption" color="text.secondary">
                              ({bj.current_benchmark})
                            </Typography>
                          )}
                        </Box>
                      </Box>

                      {bj.status === "running" && (
                        <LinearProgress
                          variant="determinate"
                          value={bj.progress}
                          sx={{
                            mb: 1.5,
                            height: 4,
                            borderRadius: 2,
                            bgcolor: "rgba(255,255,255,0.1)",
                            "& .MuiLinearProgress-bar": {
                              background: `linear-gradient(90deg, ${accentColors.info}, ${accentColors.success})`,
                            },
                          }}
                        />
                      )}

                      {bj.results.length > 0 && (
                        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                          {bj.results.map((r) => (
                            <Chip
                              key={r.benchmark_name}
                              size="small"
                              label={
                                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                  <span>{r.benchmark_name}:</span>
                                  <span style={{ fontWeight: 700 }}>
                                    {r.finetuned_score?.toFixed(1) ?? r.base_model_score?.toFixed(1) ?? "-"}%
                                  </span>
                                  {r.improvement !== undefined && r.improvement !== null && (
                                    <span
                                      style={{
                                        color: r.improvement > 0 ? accentColors.success : accentColors.rose,
                                        fontWeight: 600,
                                      }}
                                    >
                                      ({r.improvement > 0 ? "+" : ""}
                                      {r.improvement.toFixed(1)}%)
                                    </span>
                                  )}
                                </Box>
                              }
                              sx={{
                                bgcolor: "rgba(255,255,255,0.05)",
                                border: "1px solid rgba(255,255,255,0.1)",
                              }}
                            />
                          ))}
                        </Box>
                      )}
                    </Box>
                  ))}
                </Box>
              )}
            </SectionCard>
          </Grid>
        </Grid>
      )}

      {/* Real-time Training Dashboard Dialog */}
      <Dialog
        open={showDashboard}
        onClose={() => setShowDashboard(false)}
        maxWidth="lg"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: "transparent",
            boxShadow: "none",
            backgroundImage: "none",
          },
        }}
      >
        <DialogContent sx={{ p: 0 }}>
          <Box sx={{ position: "relative" }}>
            <IconButton
              onClick={() => setShowDashboard(false)}
              sx={{
                position: "absolute",
                right: 8,
                top: 8,
                zIndex: 10,
                bgcolor: "rgba(0, 0, 0, 0.4)",
                "&:hover": { bgcolor: "rgba(0, 0, 0, 0.6)" },
              }}
            >
              <CloseIcon />
            </IconButton>
            {selected && (
              <TrainingDashboard
                jobId={selected.id}
                jobName={selected.name}
                onClose={() => setShowDashboard(false)}
              />
            )}
          </Box>
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default FineTuningPage;
