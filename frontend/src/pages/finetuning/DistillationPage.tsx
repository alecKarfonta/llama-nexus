import React, { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Grid,
  Alert,
  Tooltip,
  LinearProgress,
  Switch,
  FormControlLabel,
  alpha,
  useMediaQuery,
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
  Speed as SpeedIcon,
  Grade as GradeIcon,
  Api as ApiIcon,
  AutoAwesome as AutoIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { trainingDatasetHandoffFromDistillationJob } from "@/types/trainingHandoff";
import { getRoutePath } from "@/config/navigation";
import { debugLog } from "@/utils/debug";
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from "@/components/common";
import { accentColors } from "@/utils/theme";

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

const DISTILLATION_STATUS_CONFIG: StatusChipConfigMap = {
  studio: { color: accentColors.purple, label: "Distillation studio" },
  manual: { color: accentColors.info, label: "Manual source" },
  topic: { color: accentColors.info, label: "Topic source" },
  create: { color: accentColors.success, label: "New dataset" },
  append: { color: accentColors.success, label: "Append mode" },
  active: { color: accentColors.warning, label: "Active" },
  completed: { color: accentColors.success, label: "Completed" },
  examples: { color: accentColors.purple, label: "Examples" },
};

const distillationModeButtonSx = {
  height: 30,
  borderRadius: "8px",
  fontWeight: 800,
  px: 1,
  textTransform: "none",
  minWidth: 0,
  boxShadow: "none",
  "&:hover": {
    boxShadow: "none",
  },
};

export const DistillationPage: React.FC = () => {
  const navigate = useNavigate();
  const isNarrow = useMediaQuery("(max-width:599.95px)");
  const trainingPath = getRoutePath("finetuning");
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
  const [showPromptSettings, setShowPromptSettings] = useState(false);
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
  const [generationMode, setGenerationMode] = useState("qa_pairs");
  const [targetExamples, setTargetExamples] = useState(100);
  const [refinementEnabled, setRefinementEnabled] = useState(true);
  const [refinementPasses, setRefinementPasses] = useState(2);
  const [previewPrompts, setPreviewPrompts] = useState<string[]>([]);
  const [costEstimate, setCostEstimate] = useState<any>(null);
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
          debugLog("distillation", "Examples response", data);
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

  const handleStartTraining = (job: DistillationJob) => {
    if (!job.output_dataset_id) {
      setError("No dataset available for training");
      return;
    }

    navigate(trainingPath, {
      state: {
        dataset: trainingDatasetHandoffFromDistillationJob({
          output_dataset_id: job.output_dataset_id,
          name: job.name,
          generated_count: job.generated_count,
        }),
      },
    });
  };

  const promptCount = prompts.split("\n").filter((p) => p.trim()).length;
  const activeDistillationJobs = jobs.filter((job) =>
    ["running", "pending", "queued"].includes(job.status.toLowerCase())
  ).length;
  const completedDistillationJobs = jobs.filter((job) => job.status.toLowerCase() === "completed").length;
  const generatedTotal = jobs.reduce((total, job) => total + (job.generated_count || 0), 0);
  const teacherProviderLabel =
    TEACHER_PROVIDERS.find((provider) => provider.value === teacherProvider)?.label.split(" (")[0] ?? teacherProvider;
  const selectedTeacherLabel = teacherProvider === "local" ? "Local active model" : teacherModel;
  const outputPlanLabel = datasetMode === "create" ? (createManagedDataset ? "Managed dataset" : "JSONL export") : "Append dataset";
  const distillationSummaryParts = [
    `${teacherProviderLabel} / ${selectedTeacherLabel}`,
    inputMode === "topic" ? `${targetExamples} examples` : promptCount > 0 ? `${promptCount} prompt${promptCount === 1 ? "" : "s"}` : null,
    outputPlanLabel,
  ].filter(Boolean);
  const hasDistillationJobs = jobs.length > 0;
  const jobReady = Boolean(jobName.trim());
  const sourceReady = inputMode === "manual" ? promptCount > 0 : Boolean(topic.trim());
  const outputReady = datasetMode === "create" || Boolean(appendDatasetId);
  const canStartDistillation = jobReady && sourceReady && outputReady;
  const runPlanItems = [
    {
      label: "Job",
      ready: jobReady,
      value: jobReady ? jobName : "Name required",
    },
    {
      label: "Teacher",
      ready: true,
      value: `${teacherProviderLabel} / ${selectedTeacherLabel}`,
    },
    {
      label: "Source",
      ready: sourceReady,
      value: inputMode === "manual"
        ? (sourceReady ? `${promptCount} prompt${promptCount === 1 ? "" : "s"}` : "Add prompts")
        : (sourceReady ? topic : "Topic required"),
    },
    {
      label: "Output",
      ready: outputReady,
      value: outputReady ? outputPlanLabel : "Select dataset",
    },
  ];
  const nextRequiredInput = runPlanItems.find((item) => !item.ready)?.value || "Ready";
  const distillationIconButtonSx = {
    width: 32,
    height: 32,
    borderRadius: "8px",
    border: "1px solid",
    borderColor: "divider",
    bgcolor: "action.hover",
    "&:hover": { bgcolor: "action.selected" },
  };
  const distillationHeaderActions = (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap", justifyContent: "flex-end" }}>
      <Button
        size="small"
        variant={inputMode === "manual" ? "contained" : "outlined"}
        onClick={() => setInputMode("manual")}
        sx={distillationModeButtonSx}
      >
        Manual
      </Button>
      <Button
        size="small"
        variant={inputMode === "topic" ? "contained" : "outlined"}
        onClick={() => setInputMode("topic")}
        sx={distillationModeButtonSx}
      >
        Topic
      </Button>
      <Tooltip title="Back to training">
        <IconButton
          size="small"
          aria-label="Back to training"
          onClick={() => navigate(trainingPath)}
          sx={distillationIconButtonSx}
        >
          <BackIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Refresh jobs">
        <IconButton
          size="small"
          aria-label="Refresh jobs"
          onClick={fetchJobs}
          sx={distillationIconButtonSx}
        >
          <RefreshIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
    </Box>
  );
  const shouldShowDistillationSummary =
    !isNarrow && (
      activeDistillationJobs > 0 ||
      completedDistillationJobs > 0 ||
      generatedTotal > 0 ||
      distillationSummaryParts.length > 0
    );
  const distillationSummaryRow = shouldShowDistillationSummary ? (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap", mb: 1 }}>
      {activeDistillationJobs > 0 && (
        <StatusChip status="active" label={`${activeDistillationJobs} active`} statusConfig={DISTILLATION_STATUS_CONFIG} height={22} />
      )}
      {completedDistillationJobs > 0 && (
        <StatusChip status="completed" label={`${completedDistillationJobs} complete`} statusConfig={DISTILLATION_STATUS_CONFIG} height={22} />
      )}
      {generatedTotal > 0 && (
        <StatusChip status="examples" label={`${generatedTotal} examples`} statusConfig={DISTILLATION_STATUS_CONFIG} height={22} />
      )}
      <Typography variant="caption" color="text.secondary" sx={{ minWidth: 0 }}>
        {distillationSummaryParts.join(" / ")}
      </Typography>
    </Box>
  ) : null;

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

      <Grid container spacing={1.25}>
        {/* Create Job Form */}
        <Grid item xs={12} lg={8}>
          <SectionCard
	            title="Create job"
	            icon={<DistillIcon />}
	            accentColor={accentColors.purple}
	            action={distillationHeaderActions}
	          >
              {distillationSummaryRow}
	            <Box
	              sx={{
	                display: "grid",
	                gridTemplateColumns: { xs: "1fr", xl: "minmax(0, 0.95fr) minmax(0, 1.05fr)" },
	                gap: 1.25,
	                alignItems: "start",
	              }}
	            >
	              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.1, minWidth: 0 }}>
              <TextField
                label="Job Name"
                value={jobName}
                onChange={(e) => setJobName(e.target.value)}
                placeholder="e.g., gpt4-coding-distill"
                fullWidth
                size="small"
              />

	              <Grid container spacing={1.25}>
                <Grid item xs={12} sm={6}>
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
                <Grid item xs={12} sm={6}>
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

              <Box>
                <Button
                  size="small"
                  variant={showPromptSettings ? "contained" : "outlined"}
                  onClick={() => setShowPromptSettings((show) => !show)}
                  sx={{ height: 30, borderRadius: "8px", fontWeight: 800, textTransform: "none" }}
                >
                  Prompt settings
                </Button>
              </Box>

              {showPromptSettings && (
                <>
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
                    rows={1}
                    fullWidth
                    size="small"
                  />
                </>
              )}

	              {/* Topic-Based Configuration */}
              {inputMode === "topic" && (
                <Box
                  sx={{
	                    p: 2,
	                    borderRadius: "8px",
	                    bgcolor: "action.hover",
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
	                          borderRadius: "8px",
	                          bgcolor: "action.hover",
	                          border: "1px solid",
	                          borderColor: "divider",
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
                        sx={{ flex: 1, height: 34, borderRadius: "8px" }}
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
                        sx={{ flex: 1, height: 34, borderRadius: "8px" }}
                      >
                        Estimate Cost
                      </Button>
                    </Box>
                  </Box>
                </Box>
              )}

	              </Box>

	              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.1, minWidth: 0 }}>

	              {/* Dataset Options */}
	              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
	                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1, flexWrap: "wrap" }}>
	                  <Typography variant="body2" sx={{ fontWeight: 800 }}>
	                    Output
	                  </Typography>
	                  <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap", justifyContent: "flex-end" }}>
	                    <Button
	                      size="small"
	                      variant={datasetMode === "create" ? "contained" : "outlined"}
	                      onClick={() => setDatasetMode("create")}
	                      sx={distillationModeButtonSx}
	                    >
	                      New dataset
	                    </Button>
	                    <Button
	                      size="small"
	                      variant={datasetMode === "append" ? "contained" : "outlined"}
	                      onClick={() => setDatasetMode("append")}
	                      sx={distillationModeButtonSx}
	                    >
	                      Append
	                    </Button>
	                  </Box>
	                </Box>

	                {datasetMode === "create" ? (
	                  <>
	                    <FormControlLabel
	                      control={
	                        <Switch
	                          checked={createManagedDataset}
	                          onChange={(e) => setCreateManagedDataset(e.target.checked)}
	                          color="primary"
	                          size="small"
	                        />
	                      }
	                      label={<Typography variant="body2" sx={{ fontWeight: 600 }}>Managed dataset</Typography>}
	                      sx={{ my: -0.25 }}
	                    />
	                    {createManagedDataset && (
	                      <Grid container spacing={1}>
	                        <Grid item xs={12} md={5}>
	                          <TextField
	                            label="Dataset Name (optional)"
	                            value={datasetName}
	                            onChange={(e) => setDatasetName(e.target.value)}
	                            placeholder="Leave empty to use job name"
	                            fullWidth
	                            size="small"
	                          />
	                        </Grid>
	                        <Grid item xs={12} md={7}>
	                          <TextField
	                            label="Description (optional)"
	                            value={datasetDescription}
	                            onChange={(e) => setDatasetDescription(e.target.value)}
	                            placeholder="Describe the dataset purpose"
	                            fullWidth
	                            size="small"
	                          />
	                        </Grid>
	                      </Grid>
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
	                    rows={isNarrow ? 3 : 4}
	                    fullWidth
	                    size="small"
                    sx={{ "& textarea": { fontFamily: "monospace", fontSize: "0.85rem" } }}
                  />
                  {promptCount > 0 && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                      {promptCount} prompt{promptCount !== 1 ? "s" : ""} to process
                    </Typography>
                  )}
                </Box>
              )}

              <Box sx={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 1, minHeight: 34 }}>
                {canStartDistillation ? (
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<StartIcon />}
                    onClick={handleCreateJob}
                    sx={{
                      height: 34,
                      borderRadius: "8px",
                      fontWeight: 800,
                      px: 1.5,
                      boxShadow: "none",
                      "&:hover": {
                        boxShadow: "none",
                      },
                    }}
                  >
                    Start
                  </Button>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    {nextRequiredInput}
                  </Typography>
                )}
              </Box>
              </Box>
            </Box>
          </SectionCard>
        </Grid>

        {/* Jobs List */}
        <Grid item xs={12} lg={4}>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
            {hasDistillationJobs ? (
              <SectionCard
                title="Jobs"
                icon={<TeacherIcon />}
                accentColor={accentColors.info}
                action={<StatusChip status="active" label={`${activeDistillationJobs} active`} statusConfig={DISTILLATION_STATUS_CONFIG} height={22} />}
              >
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                  {jobs.map((job) => (
                    <Box
                      key={job.id}
                      onClick={() => setSelectedJob(job)}
                      sx={{
                        p: 1.25,
                        borderRadius: "8px",
                        bgcolor: selectedJob?.id === job.id ? alpha(accentColors.success, 0.1) : "action.hover",
                        border: "1px solid",
                        borderColor: selectedJob?.id === job.id ? accentColors.success : "divider",
                        cursor: "pointer",
                        transition: "all 0.2s",
                        "&:hover": {
                          bgcolor: alpha(accentColors.primary, 0.08),
                          borderColor: alpha(accentColors.primary, 0.3),
                        },
                      }}
                    >
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.75, gap: 1 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700 }} noWrap>
                          {job.name}
                        </Typography>
                        <StatusChip status={job.status} />
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.75 }}>
                        {job.config.teacher_provider}/{job.config.teacher_model} | {job.generated_count}/{job.total_prompts} generated
                      </Typography>

                      {job.status === "running" && (
                        <Box sx={{ mb: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={job.progress}
                            sx={{
                              height: 6,
                              borderRadius: 3,
                              bgcolor: "action.hover",
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
                                sx={{ fontSize: "0.7rem" }}
                              >
                                Open Wizard
                              </Button>
                            )}
                          </>
                        )}
                      </Box>
                    </Box>
                  ))}
                </Box>
              </SectionCard>
            ) : (
              <SectionCard
                title="Run plan"
                icon={<TeacherIcon />}
                accentColor={accentColors.info}
                collapsible
                defaultExpanded={false}
                action={
                  <StatusChip
                    status={canStartDistillation ? "create" : "manual"}
                    label={canStartDistillation ? "Ready" : "Missing input"}
                    statusConfig={DISTILLATION_STATUS_CONFIG}
                    height={22}
                  />
                }
              >
                <Box sx={{ display: "grid", gap: 0.75 }}>
                  {runPlanItems.map((item) => (
                    <Box
                      key={item.label}
                      sx={{
                        p: 1,
                        borderRadius: "8px",
                        border: "1px solid",
                        borderColor: item.ready ? alpha(accentColors.success, 0.28) : "divider",
                        bgcolor: item.ready ? alpha(accentColors.success, 0.07) : "action.hover",
                      }}
                    >
                      <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                        {item.label}
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 800, overflowWrap: "anywhere" }}>
                        {item.value}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </SectionCard>
            )}
          </Box>
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
	                  <Box sx={{ textAlign: "center", p: 1.5, borderRadius: "8px", bgcolor: "action.hover" }}>
                    <Typography sx={{ fontSize: "1.5rem", fontWeight: 700, color: accentColors.success }}>
                      {qualityAssessment.metrics.success_rate}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Success Rate
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
	                  <Box sx={{ textAlign: "center", p: 1.5, borderRadius: "8px", bgcolor: "action.hover" }}>
                    <Typography sx={{ fontSize: "1.5rem", fontWeight: 700, color: accentColors.info }}>
                      {qualityAssessment.metrics.avg_quality_score}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Avg Quality
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={4}>
	                  <Box sx={{ textAlign: "center", p: 1.5, borderRadius: "8px", bgcolor: "action.hover" }}>
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
	                          borderRadius: "8px",
	                          bgcolor: "action.hover",
	                          border: "1px solid",
	                          borderColor: "divider",
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
    </PageShell>
  );
};

export default DistillationPage;
