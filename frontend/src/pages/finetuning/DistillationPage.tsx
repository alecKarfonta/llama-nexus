import React, { useEffect, useState, useCallback } from "react";
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
  IconButton,
  Chip,
  Grid,
  Alert,
  Tooltip,
  LinearProgress,
  Divider,
  Switch,
  FormControlLabel,
  Stepper,
  Step,
  StepLabel,
  alpha,
} from "@mui/material";
import {
  Psychology as DistillIcon,
  School as TeacherIcon,
  Storage as DatasetIcon,
  PlayArrow as StartIcon,
  Stop as CancelIcon,
  Download as DownloadIcon,
  ArrowBack as BackIcon,
  Refresh as RefreshIcon,
  RocketLaunch as TrainIcon,
  CheckCircle as SuccessIcon,
  Timer as TimerIcon,
  Speed as SpeedIcon,
  Grade as GradeIcon,
  Lightbulb as TipIcon,
  Api as ApiIcon,
  AutoAwesome as AutoIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

// Types
type PromptTemplate = {
  id: string;
  name: string;
  description: string;
  system_prompt: string;
  user_template: string;
};

type DistillationJob = {
  id: string;
  name: string;
  status: string;
  progress: number;
  generated_count: number;
  total_prompts: number;
  config: {
    teacher_provider: string;
    teacher_model: string;
  };
  created_at: string;
  output_dataset_id?: string;
};

type GeneratedExample = {
  prompt: string;
  response: string;
  metadata?: Record<string, any>;
};

type QualityAssessment = {
  overall_grade: string;
  grade_color: string;
  metrics: {
    success_rate: number;
    avg_quality_score: number;
    total_examples: number;
  };
  estimated_performance: {
    level: string;
    description: string;
    confidence: number;
  };
  recommendations: Array<{
    type: string;
    title: string;
    description: string;
    action: string;
  }>;
};

type Dataset = {
  id: string;
  name: string;
  num_examples: number;
  format: string;
};

const TEACHER_PROVIDERS = [
  { value: "openai", label: "OpenAI (GPT-4, GPT-3.5)", icon: <ApiIcon /> },
  { value: "anthropic", label: "Anthropic (Claude)", icon: <AutoIcon /> },
  { value: "local", label: "Local Llama Nexus Model", icon: <SpeedIcon /> },
];

// Model presets per provider
const MODEL_PRESETS: Record<string, { value: string; label: string }[]> = {
  openai: [
    { value: "gpt-4o", label: "GPT-4o (Recommended)" },
    { value: "gpt-4-turbo", label: "GPT-4 Turbo" },
    { value: "gpt-3.5-turbo", label: "GPT-3.5 Turbo (Faster)" },
  ],
  anthropic: [
    { value: "claude-sonnet-4-20250514", label: "Claude Sonnet 4 (Recommended)" },
    { value: "claude-3-5-haiku-20241022", label: "Claude 3.5 Haiku (Faster)" },
  ],
  local: [
    { value: "auto", label: "Active Model (Auto-detect)" },
  ],
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

// Status Chip
const StatusChip: React.FC<{ status: string }> = ({ status }) => {
  const config: Record<string, { color: string; label: string }> = {
    completed: { color: accentColors.success, label: "Completed" },
    running: { color: accentColors.info, label: "Running" },
    failed: { color: accentColors.rose, label: "Failed" },
    pending: { color: accentColors.warning, label: "Pending" },
    cancelled: { color: "#64748b", label: "Cancelled" },
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


export const DistillationPage: React.FC = () => {
  const navigate = useNavigate();
  const [templates, setTemplates] = useState<PromptTemplate[]>([]);
  const [jobs, setJobs] = useState<DistillationJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<DistillationJob | null>(null);
  const [generatedExamples, setGeneratedExamples] = useState<GeneratedExample[]>([]);
  const [loadingExamples, setLoadingExamples] = useState(false);
  const [examplesSearch, setExamplesSearch] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [qualityAssessment, setQualityAssessment] = useState<QualityAssessment | null>(null);

  // Form state
  const [jobName, setJobName] = useState("");
  const [teacherProvider, setTeacherProvider] = useState("openai");
  const [teacherModel, setTeacherModel] = useState("gpt-4");
  const [apiKey, setApiKey] = useState("");
  const [selectedTemplate, setSelectedTemplate] = useState<string>("");
  const [customSystemPrompt, setCustomSystemPrompt] = useState("");
  const [prompts, setPrompts] = useState<string>("");
  const [createManagedDataset, setCreateManagedDataset] = useState(true);
  const [datasetName, setDatasetName] = useState("");
  const [toolDefinitions, setToolDefinitions] = useState("");

  const [datasetDescription, setDatasetDescription] = useState("");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetMode, setDatasetMode] = useState<"create" | "append">("create");
  const [appendDatasetId, setAppendDatasetId] = useState("");

  // Topic-based generation state
  const [inputMode, setInputMode] = useState<"manual" | "topic">("manual");
  const [topic, setTopic] = useState("");
  const [subtopics, setSubtopics] = useState<string[]>([]);
  const [generationMode, setGenerationMode] = useState("qa_pairs");
  const [targetExamples, setTargetExamples] = useState(100);
  const [refinementEnabled, setRefinementEnabled] = useState(true);
  const [refinementPasses, setRefinementPasses] = useState(2);
  const [previewPrompts, setPreviewPrompts] = useState<string[]>([]);
  const [costEstimate, setCostEstimate] = useState<any>(null);
  const [expandingTopic, setExpandingTopic] = useState(false);
  const [generatingPreview, setGeneratingPreview] = useState(false);

  const fetchTemplates = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/distillation/templates");
      if (res.ok) {
        const data = await res.json();
        setTemplates(data.templates || []);
      }
    } catch {
      console.error("Failed to load templates");
    }
  }, []);

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/distillation/jobs");
      if (res.ok) {
        const data = await res.json();
        setJobs(data.jobs || []);
      }
    } catch {
      console.error("Failed to load jobs");
    }
  }, []);

  const fetchDatasets = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/datasets");
      if (res.ok) {
        const data = await res.json();
        // API returns array directly, not {datasets: []}
        setDatasets(Array.isArray(data) ? data : (data.datasets || []));
      }
    } catch {
      console.error("Failed to load datasets");
    }
  }, []);


  useEffect(() => {
    fetchTemplates();
    fetchJobs();
    fetchDatasets();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, [fetchTemplates, fetchJobs, fetchDatasets]);

  // Auto-set default model when provider changes
  useEffect(() => {
    const presets = MODEL_PRESETS[teacherProvider];
    if (presets && presets.length > 0) {
      setTeacherModel(presets[0].value);
    }
  }, [teacherProvider]);

  useEffect(() => {
    if (!selectedJob) {
      setGeneratedExamples([]);
      setQualityAssessment(null);
      return;
    }

    const fetchExamples = () => {
      setLoadingExamples(true);
      fetch(`/api/v1/finetune/distillation/jobs/${selectedJob.id}/preview?limit=50`)
        .then((res) => res.json())
        .then((data) => {
          console.log("Examples response:", data);
          setGeneratedExamples(data.examples || []);
        })
        .catch((err) => {
          console.error("Failed to fetch examples:", err);
          setGeneratedExamples([]);
        })
        .finally(() => setLoadingExamples(false));
    };

    fetchExamples();

    // Auto-refresh while job is running
    let interval: NodeJS.Timeout | null = null;
    if (selectedJob.status === "running") {
      interval = setInterval(fetchExamples, 3000);
    }

    if (selectedJob.status === "completed") {
      fetch(`/api/v1/finetune/distillation/jobs/${selectedJob.id}/quality-assessment`)
        .then((res) => res.json())
        .then((data) => setQualityAssessment(data))
        .catch(() => setQualityAssessment(null));
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [selectedJob, selectedJob?.status]);

  const handleTemplateSelect = (templateId: string) => {
    setSelectedTemplate(templateId);
    const template = templates.find((t) => t.id === templateId);
    if (template) {
      setCustomSystemPrompt(template.system_prompt);
    }
  };

  const handleCreateJob = async () => {
    // Validate based on input mode
    if (inputMode === "manual" && (!jobName || !prompts.trim())) {
      setError("Please provide a job name and prompts");
      return;
    }
    if (inputMode === "topic" && (!jobName || !topic)) {
      setError("Please provide a job name and topic");
      return;
    }
    if (datasetMode === "append" && !appendDatasetId) {
      setError("Please select a dataset to append to");
      return;
    }
    setError(null);
    setActionStatus("Creating distillation job...");

    const promptList = inputMode === "manual"
      ? prompts.split("\n").filter((p) => p.trim())
      : [];

    // Build request body based on mode
    const requestBody: any = {
      name: jobName,
      config: {
        teacher_provider: teacherProvider,
        teacher_model: teacherModel,
        teacher_api_key: apiKey || undefined,
        system_prompt: customSystemPrompt || undefined,
        create_managed_dataset: datasetMode === "create" ? createManagedDataset : false,
        dataset_name: datasetMode === "create" ? (datasetName || undefined) : undefined,
        dataset_description: datasetMode === "create" ? (datasetDescription || undefined) : undefined,
        append_to_dataset_id: datasetMode === "append" ? appendDatasetId : undefined,
      },
      prompts: promptList,
    };

    // Add topic configuration if in topic mode
    if (inputMode === "topic") {
      requestBody.config.use_topic_generation = true;
      requestBody.config.topic_config = {
        topic,
        mode: generationMode,
        target_examples: targetExamples,
        difficulty_levels: ["beginner", "intermediate", "advanced"],
        tool_definitions: generationMode === "tool_calling" ? toolDefinitions : undefined,
        refinement: {
          enabled: refinementEnabled,
          max_passes: refinementPasses,
        },
      };
      requestBody.target_examples = targetExamples;
    }

    try {
      const res = await fetch("/api/v1/finetune/distillation/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      if (res.ok) {
        const job = await res.json();
        setActionStatus("Job created! Starting distillation...");
        await fetch(`/api/v1/finetune/distillation/jobs/${job.id}/start`, { method: "POST" });
        setActionStatus("Distillation started successfully");
        setJobName("");
        setPrompts("");
        setTopic("");
        setPreviewPrompts([]);
        setCostEstimate(null);
        fetchJobs();
      } else {
        const data = await res.json();
        setError(data.detail || "Failed to create job");
      }
    } catch {
      setError("Failed to create job");
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await fetch(`/api/v1/finetune/distillation/jobs/${jobId}/cancel`, { method: "POST" });
      fetchJobs();
      setActionStatus("Job cancelled");
    } catch {
      setError("Failed to cancel job");
    }
  };

  const handleStartTraining = async (job: DistillationJob) => {
    if (!job.output_dataset_id) {
      setError("No dataset available for training");
      return;
    }

    try {
      setActionStatus("Creating training job...");

      const trainingJobData = {
        name: `${job.name} - Fine-tuned`,
        dataset_id: job.output_dataset_id,
        base_model: "meta-llama/Llama-3.2-8B-Instruct",
        preset_name: "balanced",
        config: {
          num_train_epochs: Math.min(3, Math.max(1, Math.ceil(1000 / job.generated_count))),
          learning_rate: job.generated_count < 100 ? 5e-4 : 2e-4,
          per_device_train_batch_size: job.generated_count < 500 ? 1 : 2,
          gradient_accumulation_steps: 4,
          warmup_steps: Math.max(10, Math.floor(job.generated_count * 0.1)),
          save_steps: Math.max(50, Math.floor(job.generated_count * 0.2)),
          eval_steps: Math.max(25, Math.floor(job.generated_count * 0.1)),
        },
      };

      const res = await fetch("/api/v1/finetune/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trainingJobData),
      });

      if (res.ok) {
        const trainingJob = await res.json();
        setActionStatus("Training job created! Redirecting...");
        setTimeout(() => {
          navigate(`/finetuning?job=${trainingJob.id}`);
        }, 1000);
      } else {
        const data = await res.json();
        setError(data.detail || "Failed to create training job");
      }
    } catch (err) {
      setError("Failed to create training job");
    }
  };

  const promptCount = prompts.split("\n").filter((p) => p.trim()).length;

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
              Knowledge Distillation
            </Typography>
            <Chip
              size="small"
              label="Beta"
              sx={{
                height: 22,
                bgcolor: alpha(accentColors.info, 0.1),
                border: `1px solid ${alpha(accentColors.info, 0.2)}`,
                color: accentColors.info,
                fontWeight: 600,
                fontSize: "0.6875rem",
              }}
            />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.875rem", maxWidth: 500 }}>
            Generate high-quality training data from teacher models like GPT-4 or Claude
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
            Back
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

      <Grid container spacing={3}>
        {/* Create Job Form */}
        <Grid item xs={12} lg={6}>
          <SectionCard
            title="Create Distillation Job"
            subtitle="Configure and generate training examples"
            icon={<DistillIcon />}
            accentColor={accentColors.purple}
          >
            <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
              <TextField
                label="Job Name"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="e.g., gpt4-coding-distill"
                fullWidth
                size="small"
              />

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Teacher Provider</InputLabel>
                    <Select
                      value={teacherProvider}
                      onChange={(e) => setTeacherProvider(e.target.value)}
                      label="Teacher Provider"
                    >
                      {TEACHER_PROVIDERS.map((p) => (
                        <MenuItem key={p.value} value={p.value}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            {p.icon}
                            <span>{p.label}</span>
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Model</InputLabel>
                    <Select
                      value={teacherModel}
                      onChange={(e) => setTeacherModel(e.target.value)}
                      label="Model"
                    >
                      {(MODEL_PRESETS[teacherProvider] || []).map((m) => (
                        <MenuItem key={m.value} value={m.value}>
                          {m.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>

              </Grid>

              {teacherProvider !== "local" && (
                <TextField
                  label="API Key"
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  fullWidth
                  size="small"
                />
              )}

              <FormControl fullWidth size="small">
                <InputLabel>Prompt Template</InputLabel>
                <Select
                  value={selectedTemplate}
                  onChange={(e) => handleTemplateSelect(e.target.value)}
                  label="Prompt Template"
                >
                  <MenuItem value="">Custom / None</MenuItem>
                  {templates.map((t) => (
                    <MenuItem key={t.id} value={t.id}>
                      {t.name} - {t.description}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <TextField
                label="System Prompt"
                value={customSystemPrompt}
                onChange={(e) => setCustomSystemPrompt(e.target.value)}
                placeholder="You are a helpful assistant..."
                multiline
                rows={2}
                fullWidth
                size="small"
              />

              {/* Input Mode Toggle */}
              <Box
                sx={{
                  display: "flex",
                  gap: 1,
                  p: 0.5,
                  borderRadius: 2,
                  bgcolor: "rgba(0, 0, 0, 0.2)",
                  border: "1px solid rgba(255, 255, 255, 0.06)",
                }}
              >
                <Button
                  size="small"
                  variant={inputMode === "manual" ? "contained" : "text"}
                  onClick={() => setInputMode("manual")}
                  sx={{
                    flex: 1,
                    bgcolor: inputMode === "manual" ? alpha(accentColors.primary, 0.2) : "transparent",
                    color: inputMode === "manual" ? accentColors.primary : "text.secondary",
                    "&:hover": { bgcolor: alpha(accentColors.primary, 0.15) },
                  }}
                >
                  Manual Prompts
                </Button>
                <Button
                  size="small"
                  variant={inputMode === "topic" ? "contained" : "text"}
                  onClick={() => setInputMode("topic")}
                  sx={{
                    flex: 1,
                    bgcolor: inputMode === "topic" ? alpha(accentColors.purple, 0.2) : "transparent",
                    color: inputMode === "topic" ? accentColors.purple : "text.secondary",
                    "&:hover": { bgcolor: alpha(accentColors.purple, 0.15) },
                  }}
                >
                  Topic-Based
                </Button>
              </Box>

              {/* Topic-Based Configuration */}
              {inputMode === "topic" && (
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 2,
                    bgcolor: "rgba(0, 0, 0, 0.2)",
                    border: `1px solid ${alpha(accentColors.purple, 0.2)}`,
                  }}
                >
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2, color: accentColors.purple }}>
                    Topic Configuration
                  </Typography>

                  <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                    <TextField
                      label="Topic"
                      value={topic}
                      onChange={(e) => setTopic(e.target.value)}
                      placeholder="e.g., Car diagnostics, Python async programming, Warhammer 40K lore"
                      fullWidth
                      size="small"
                    />

                    <FormControl fullWidth size="small">
                      <InputLabel>Generation Mode</InputLabel>
                      <Select
                        value={generationMode}
                        onChange={(e) => setGenerationMode(e.target.value)}
                        label="Generation Mode"
                      >
                        <MenuItem value="qa_pairs">Question & Answer Pairs</MenuItem>
                        <MenuItem value="instructions">Instruction Following</MenuItem>
                        <MenuItem value="trivia">Trivia & Facts</MenuItem>
                        <MenuItem value="troubleshooting">Troubleshooting & Diagnostics</MenuItem>
                        <MenuItem value="code_examples">Code Examples</MenuItem>
                        <MenuItem value="explanations">Concept Explanations</MenuItem>
                        <MenuItem value="conversations">Multi-turn Conversations</MenuItem>
                        <MenuItem value="tool_calling">Tool Calling / Function Use</MenuItem>
                        <MenuItem value="mixed">Mixed Formats</MenuItem>
                      </Select>
                    </FormControl>

                    {generationMode === "tool_calling" && (
                      <TextField
                        label="Tool Definitions (JSON Schema)"
                        value={toolDefinitions}
                        onChange={(e) => setToolDefinitions(e.target.value)}
                        placeholder='[{"name": "get_weather", "description": "Get weather", "parameters": {...}}]'
                        multiline
                        rows={6}
                        fullWidth
                        size="small"
                        sx={{ "& textarea": { fontFamily: "monospace", fontSize: "0.85rem" } }}
                      />
                    )}

                    <Box>
                      <Typography variant="caption" sx={{ color: "text.secondary" }}>
                        Target Examples: {targetExamples}
                      </Typography>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                        <Typography variant="caption">50</Typography>
                        <input
                          type="range"
                          min={50}
                          max={500}
                          step={10}
                          value={targetExamples}
                          onChange={(e) => setTargetExamples(Number(e.target.value))}
                          style={{ flex: 1 }}
                        />
                        <Typography variant="caption">500</Typography>
                      </Box>
                    </Box>

                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={refinementEnabled}
                            onChange={(e) => setRefinementEnabled(e.target.checked)}
                            size="small"
                          />
                        }
                        label={
                          <Typography variant="body2">Multi-pass refinement</Typography>
                        }
                      />
                      {refinementEnabled && (
                        <FormControl size="small" sx={{ minWidth: 100 }}>
                          <InputLabel>Passes</InputLabel>
                          <Select
                            value={refinementPasses}
                            onChange={(e) => setRefinementPasses(Number(e.target.value))}
                            label="Passes"
                          >
                            <MenuItem value={1}>1 pass</MenuItem>
                            <MenuItem value={2}>2 passes</MenuItem>
                            <MenuItem value={3}>3 passes</MenuItem>
                          </Select>
                        </FormControl>
                      )}
                    </Box>

                    {costEstimate && (
                      <Alert severity="info" sx={{ py: 0.5 }}>
                        Estimated cost: ${costEstimate.estimated_cost_usd} ({costEstimate.estimated_tokens?.toLocaleString()} tokens)
                      </Alert>
                    )}

                    {previewPrompts.length > 0 && (
                      <Box
                        sx={{
                          maxHeight: 120,
                          overflow: "auto",
                          p: 1.5,
                          borderRadius: 1,
                          bgcolor: "rgba(0, 0, 0, 0.3)",
                          fontFamily: "monospace",
                          fontSize: "0.75rem",
                        }}
                      >
                        <Typography variant="caption" sx={{ fontWeight: 600, mb: 1, display: "block" }}>
                          Preview Prompts:
                        </Typography>
                        {previewPrompts.slice(0, 5).map((p, i) => (
                          <Box key={i} sx={{ mb: 0.5, color: "text.secondary" }}>
                            {i + 1}. {p.slice(0, 100)}{p.length > 100 ? "..." : ""}
                          </Box>
                        ))}
                      </Box>
                    )}

                    <Box sx={{ display: "flex", gap: 1 }}>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={async () => {
                          if (!topic || !apiKey) return;
                          setGeneratingPreview(true);
                          try {
                            const res = await fetch("/api/v1/finetune/distillation/topics/preview", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({
                                topic,
                                mode: generationMode,
                                num_prompts: 5,
                                teacher_provider: teacherProvider,
                                teacher_model: teacherModel,
                                teacher_api_key: apiKey,
                              }),
                            });
                            if (res.ok) {
                              const data = await res.json();
                              setPreviewPrompts(data.prompts || []);
                            }
                          } catch (e) {
                            console.error(e);
                          }
                          setGeneratingPreview(false);
                        }}
                        disabled={!topic || !apiKey || generatingPreview}
                        sx={{ flex: 1 }}
                      >
                        {generatingPreview ? "Generating..." : "Preview"}
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={async () => {
                          try {
                            const res = await fetch("/api/v1/finetune/distillation/topics/estimate-cost", {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({
                                topic,
                                mode: generationMode,
                                target_examples: targetExamples,
                                refinement_enabled: refinementEnabled,
                                refinement_passes: refinementPasses,
                                teacher_model: teacherModel,
                              }),
                            });
                            if (res.ok) {
                              const data = await res.json();
                              setCostEstimate(data);
                            }
                          } catch (e) {
                            console.error(e);
                          }
                        }}
                        disabled={!topic}
                        sx={{ flex: 1 }}
                      >
                        Estimate Cost
                      </Button>
                    </Box>
                  </Box>
                </Box>
              )}

              {/* Dataset Options */}
              <Box
                sx={{
                  p: 2,
                  borderRadius: 2,
                  bgcolor: "rgba(0, 0, 0, 0.2)",
                  border: "1px solid rgba(255, 255, 255, 0.06)",
                }}
              >
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 2 }}>
                  Output Options
                </Typography>
                <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
                  <Button
                    size="small"
                    variant={datasetMode === "create" ? "contained" : "outlined"}
                    onClick={() => setDatasetMode("create")}
                    sx={{ flex: 1 }}
                  >
                    Create New Dataset
                  </Button>
                  <Button
                    size="small"
                    variant={datasetMode === "append" ? "contained" : "outlined"}
                    onClick={() => setDatasetMode("append")}
                    sx={{ flex: 1 }}
                  >
                    Append to Existing
                  </Button>
                </Box>

                {datasetMode === "create" ? (
                  <>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={createManagedDataset}
                          onChange={(e) => setCreateManagedDataset(e.target.checked)}
                          color="primary"
                        />
                      }
                      label={
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            Create managed dataset
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Automatically create a dataset for training
                          </Typography>
                        </Box>
                      }
                    />
                    {createManagedDataset && (
                      <Box sx={{ mt: 2, display: "flex", flexDirection: "column", gap: 1.5 }}>
                        <TextField
                          label="Dataset Name (optional)"
                          value={datasetName}
                          onChange={(e) => setDatasetName(e.target.value)}
                          placeholder="Leave empty to use job name"
                          fullWidth
                          size="small"
                        />
                        <TextField
                          label="Description (optional)"
                          value={datasetDescription}
                          onChange={(e) => setDatasetDescription(e.target.value)}
                          placeholder="Describe the dataset purpose"
                          multiline
                          rows={2}
                          fullWidth
                          size="small"
                        />
                      </Box>
                    )}
                  </>
                ) : (
                  <FormControl fullWidth size="small">
                    <InputLabel>Select Dataset to Append To</InputLabel>
                    <Select
                      value={appendDatasetId}
                      onChange={(e) => setAppendDatasetId(e.target.value)}
                      label="Select Dataset to Append To"
                    >
                      {datasets.map((ds) => (
                        <MenuItem key={ds.id} value={ds.id}>
                          {ds.name} ({ds.num_examples} examples)
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              </Box>

              {/* Manual Prompts - only shown in manual mode */}
              {inputMode === "manual" && (
                <Box>
                  <TextField
                    label="Prompts (one per line)"
                    value={prompts}
                    onChange={(e) => setPrompts(e.target.value)}
                    placeholder={"Write a Python function to sort a list\nExplain quantum computing\nHow do I optimize database queries?"}
                    multiline
                    rows={5}
                    fullWidth
                    size="small"
                    sx={{ "& textarea": { fontFamily: "monospace", fontSize: "0.85rem" } }}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                    {promptCount} prompt{promptCount !== 1 ? "s" : ""} to process
                  </Typography>
                </Box>
              )}

              <Button
                variant="contained"
                startIcon={<StartIcon />}
                onClick={handleCreateJob}
                disabled={!jobName || (inputMode === "manual" ? promptCount === 0 : !topic)}
                sx={{
                  background: `linear-gradient(135deg, ${accentColors.purple} 0%, ${accentColors.primary} 100%)`,
                }}
              >
                Start Distillation
              </Button>
            </Box>
          </SectionCard>
        </Grid>

        {/* Jobs List */}
        <Grid item xs={12} lg={6}>
          <SectionCard
            title="Distillation Jobs"
            subtitle={`${jobs.length} jobs`}
            icon={<TeacherIcon />}
            accentColor={accentColors.info}
          >
            {jobs.length === 0 ? (
              <Box sx={{ p: 4, textAlign: "center" }}>
                <DistillIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                <Typography color="text.secondary">No distillation jobs yet</Typography>
              </Box>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                {jobs.map((job) => (
                  <Box
                    key={job.id}
                    onClick={() => setSelectedJob(job)}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      bgcolor: selectedJob?.id === job.id ? alpha(accentColors.success, 0.1) : "rgba(255, 255, 255, 0.02)",
                      border: `1px solid ${selectedJob?.id === job.id ? accentColors.success : "rgba(255, 255, 255, 0.06)"}`,
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
                        {job.name}
                      </Typography>
                      <StatusChip status={job.status} />
                    </Box>
                    <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
                      {job.config.teacher_provider}/{job.config.teacher_model} | {job.generated_count}/{job.total_prompts} generated
                    </Typography>

                    {job.status === "running" && (
                      <Box sx={{ mb: 1.5 }}>
                        <LinearProgress
                          variant="determinate"
                          value={job.progress}
                          sx={{
                            height: 6,
                            borderRadius: 3,
                            bgcolor: "rgba(255, 255, 255, 0.1)",
                            "& .MuiLinearProgress-bar": {
                              background: `linear-gradient(90deg, ${accentColors.info}, ${accentColors.success})`,
                            },
                          }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                          {job.progress.toFixed(1)}% complete
                        </Typography>
                      </Box>
                    )}

                    <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                      {job.status === "running" && (
                        <Button
                          size="small"
                          variant="outlined"
                          color="error"
                          startIcon={<CancelIcon />}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCancelJob(job.id);
                          }}
                          sx={{ fontSize: "0.7rem" }}
                        >
                          Cancel
                        </Button>
                      )}
                      {job.status === "completed" && (
                        <>
                          <Button
                            size="small"
                            variant="outlined"
                            startIcon={<DownloadIcon />}
                            onClick={(e) => {
                              e.stopPropagation();
                              window.open(`/api/v1/finetune/distillation/jobs/${job.id}/download`, "_blank");
                            }}
                            sx={{ fontSize: "0.7rem" }}
                          >
                            Download
                          </Button>
                          {job.output_dataset_id && (
                            <Button
                              size="small"
                              variant="contained"
                              startIcon={<TrainIcon />}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleStartTraining(job);
                              }}
                              sx={{
                                fontSize: "0.7rem",
                                background: `linear-gradient(135deg, ${accentColors.success} 0%, ${accentColors.info} 100%)`,
                              }}
                            >
                              Start Training
                            </Button>
                          )}
                        </>
                      )}
                    </Box>
                  </Box>
                ))}
              </Box>
            )}
          </SectionCard>
        </Grid>


        {/* Quality Assessment */}
        {selectedJob && selectedJob.status === "completed" && qualityAssessment && (
          <Grid item xs={12} lg={5}>
            <SectionCard title="Quality Assessment" icon={<GradeIcon />} accentColor={accentColors.success}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 3, mb: 3 }}>
                <Box
                  sx={{
                    width: 64,
                    height: 64,
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: qualityAssessment.grade_color,
                    color: "#fff",
                    fontSize: "1.75rem",
                    fontWeight: 700,
                    boxShadow: `0 4px 20px ${alpha(qualityAssessment.grade_color, 0.5)}`,
                  }}
                >
                  {qualityAssessment.overall_grade}
                </Box>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 700 }}>
                    {qualityAssessment.estimated_performance.level} Quality
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Confidence: {qualityAssessment.estimated_performance.confidence}%
                  </Typography>
                </Box>
              </Box>

              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: "center", p: 1.5, borderRadius: 2, bgcolor: "rgba(0, 0, 0, 0.2)" }}>
                    <Typography sx={{ fontSize: "1.5rem", fontWeight: 700, color: accentColors.success }}>
                      {qualityAssessment.metrics.success_rate}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Success Rate
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: "center", p: 1.5, borderRadius: 2, bgcolor: "rgba(0, 0, 0, 0.2)" }}>
                    <Typography sx={{ fontSize: "1.5rem", fontWeight: 700, color: accentColors.info }}>
                      {qualityAssessment.metrics.avg_quality_score}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Avg Quality
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
                  <Box sx={{ textAlign: "center", p: 1.5, borderRadius: 2, bgcolor: "rgba(0, 0, 0, 0.2)" }}>
                    <Typography sx={{ fontSize: "1.5rem", fontWeight: 700, color: accentColors.warning }}>
                      {qualityAssessment.metrics.total_examples}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Examples
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              {qualityAssessment.recommendations.length > 0 && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {qualityAssessment.recommendations.map((rec, i) => (
                    <Alert
                      key={i}
                      severity={rec.type === "warning" ? "warning" : rec.type === "success" ? "success" : "info"}
                      sx={{ py: 1 }}
                    >
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {rec.title}
                      </Typography>
                      <Typography variant="caption">{rec.description}</Typography>
                    </Alert>
                  ))}
                </Box>
              )}
            </SectionCard>
          </Grid>
        )}

        {/* Generated Examples Preview */}
        {selectedJob && (
          <Grid item xs={12} lg={selectedJob.status === "completed" && qualityAssessment ? 7 : 12}>
            <SectionCard
              title="Generated Examples"
              subtitle={`${selectedJob.generated_count} of ${selectedJob.total_prompts} examples${selectedJob.status === "running" ? " (auto-refreshing)" : ""}`}
              icon={<DatasetIcon />}
              accentColor={accentColors.primary}
              action={
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => window.open(`/api/v1/finetune/distillation/jobs/${selectedJob.id}/download`, "_blank")}
                  sx={{ fontSize: "0.7rem" }}
                >
                  Export JSONL
                </Button>
              }
            >
              {/* Search Filter */}
              <TextField
                placeholder="Search examples..."
                value={examplesSearch}
                onChange={(e) => setExamplesSearch(e.target.value)}
                size="small"
                fullWidth
                sx={{ mb: 2 }}
              />

              {loadingExamples && generatedExamples.length === 0 ? (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <LinearProgress sx={{ mb: 2 }} />
                  <Typography color="text.secondary">Loading examples...</Typography>
                </Box>
              ) : generatedExamples.length === 0 ? (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <DatasetIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                  <Typography color="text.secondary">No examples generated yet</Typography>
                </Box>
              ) : (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5, maxHeight: 500, overflow: "auto" }}>
                  {generatedExamples
                    .filter((ex) =>
                      !examplesSearch ||
                      ex.prompt.toLowerCase().includes(examplesSearch.toLowerCase()) ||
                      ex.response.toLowerCase().includes(examplesSearch.toLowerCase())
                    )
                    .map((ex, i) => (
                      <Box
                        key={i}
                        sx={{
                          p: 2,
                          borderRadius: 2,
                          bgcolor: "rgba(0, 0, 0, 0.2)",
                          border: "1px solid rgba(255, 255, 255, 0.06)",
                        }}
                      >
                        <Box sx={{ mb: 1.5 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                            Prompt:
                          </Typography>
                          <Typography variant="body2" sx={{ mt: 0.5 }}>
                            {ex.prompt}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                            Response:
                          </Typography>
                          <Typography
                            variant="body2"
                            sx={{
                              mt: 0.5,
                              whiteSpace: "pre-wrap",
                              color: accentColors.success,
                              fontFamily: "monospace",
                              fontSize: "0.8rem",
                              maxHeight: 200,
                              overflow: "auto",
                            }}
                          >
                            {ex.response}
                          </Typography>
                        </Box>
                      </Box>
                    ))}
                </Box>
              )}
            </SectionCard>
          </Grid>
        )}

      </Grid>
    </Box>
  );
};

export default DistillationPage;
