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
  Tabs,
  Tab,
  Switch,
  FormControlLabel,
  Slider,
  alpha,
} from "@mui/material";
import {
  Assessment as EvalIcon,
  Compare as CompareIcon,
  Science as BenchmarkIcon,
  Balance as ABTestIcon,
  Psychology as JudgeIcon,
  ArrowBack as BackIcon,
  Refresh as RefreshIcon,
  PlayArrow as StartIcon,
  Pause as PauseIcon,
  Add as AddIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  Speed as SpeedIcon,
  Timer as TimerIcon,
  EmojiEvents as TrophyIcon,
  AutoFixHigh as MagicIcon,
  VisibilityOff as BlindIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

// Types
type FineTuningJob = {
  id: string;
  name: string;
  base_model: string;
  status: string;
  adapter_path?: string;
};

type ComparisonSession = {
  id: string;
  name: string;
  base_model: string;
  adapter_id: string;
  comparison_type: string;
  total_comparisons: number;
  base_preferred: number;
  finetuned_preferred: number;
  ties: number;
  created_at: string;
};

type PromptComparison = {
  id: string;
  prompt: string;
  base_response?: { response: string; tokens_per_second: number };
  finetuned_response?: { response: string; tokens_per_second: number };
  preferred_model?: string;
};

type BenchmarkInfo = {
  name: string;
  display_name: string;
  description: string;
};

type BenchmarkJob = {
  id: string;
  name: string;
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

type ABTest = {
  id: string;
  name: string;
  status: string;
  variants: Array<{
    variant_id: string;
    name: string;
    adapter_path?: string;
    weight: number;
  }>;
  created_at: string;
};

type ABTestSummary = {
  test_id: string;
  name: string;
  status: string;
  variants: Array<{
    variant_id: string;
    name: string;
    total_requests: number;
    success_rate: number;
    avg_tokens_per_second: number;
    avg_latency_ms: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    thumbs_up: number;
    thumbs_down: number;
    feedback_score?: number;
  }>;
  winner?: string;
  winner_name?: string;
  has_enough_samples?: boolean;
  samples_needed?: number;
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
      height: "100%",
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
    draft: { color: "#64748b", label: "Draft" },
    paused: { color: accentColors.warning, label: "Paused" },
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

// Metric Bar Component
const MetricBar: React.FC<{
  label: string;
  value: number;
  max: number;
  unit: string;
  color: string;
}> = ({ label, value, max, unit, color }) => {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <Box sx={{ mb: 1.5 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="caption" sx={{ fontWeight: 700, color }}>
          {value.toFixed(1)}
          {unit}
        </Typography>
      </Box>
      <Box sx={{ bgcolor: "rgba(255, 255, 255, 0.1)", borderRadius: 1, height: 6, overflow: "hidden" }}>
        <Box
          sx={{
            width: `${Math.min(pct, 100)}%`,
            height: "100%",
            bgcolor: color,
            borderRadius: 1,
            transition: "width 0.3s",
          }}
        />
      </Box>
    </Box>
  );
};

export const ModelEvaluationPage: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState(0);
  const [jobs, setJobs] = useState<FineTuningJob[]>([]);
  const [sessions, setSessions] = useState<ComparisonSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<ComparisonSession | null>(null);
  const [comparisons, setComparisons] = useState<PromptComparison[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkInfo[]>([]);
  const [benchmarkJobs, setBenchmarkJobs] = useState<BenchmarkJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);

  // Comparison form
  const [sessionName, setSessionName] = useState("");
  const [selectedAdapter, setSelectedAdapter] = useState("");
  const [testPrompt, setTestPrompt] = useState("");
  const [blindMode, setBlindMode] = useState(false);

  // Benchmark form
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>(["mmlu", "hellaswag"]);
  const [benchmarkAdapter, setBenchmarkAdapter] = useState("");

  // A/B Testing
  const [abTests, setAbTests] = useState<ABTest[]>([]);
  const [selectedAbTest, setSelectedAbTest] = useState<ABTest | null>(null);
  const [abTestSummary, setAbTestSummary] = useState<ABTestSummary | null>(null);
  const [abTestName, setAbTestName] = useState("");
  const [abTestVariants, setAbTestVariants] = useState<Array<{ name: string; adapterId: string; weight: number }>>([
    { name: "Base Model", adapterId: "", weight: 0.5 },
    { name: "Fine-tuned", adapterId: "", weight: 0.5 },
  ]);
  const [abTestPrompt, setAbTestPrompt] = useState("");

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/jobs");
      if (res.ok) setJobs(await res.json());
    } catch {}
  }, []);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/eval/sessions");
      if (res.ok) {
        const data = await res.json();
        setSessions(data.sessions || []);
      }
    } catch {}
  }, []);

  const fetchBenchmarks = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/benchmarks/available");
      if (res.ok) {
        const data = await res.json();
        setBenchmarks(data.benchmarks || []);
      }
    } catch {}
  }, []);

  const fetchAbTests = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/ab-tests");
      if (res.ok) {
        const data = await res.json();
        setAbTests(data.tests || []);
      }
    } catch {}
  }, []);

  const fetchBenchmarkJobs = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/benchmarks/jobs");
      if (res.ok) {
        const data = await res.json();
        setBenchmarkJobs(data.jobs || []);
      }
    } catch {}
  }, []);

  useEffect(() => {
    fetchJobs();
    fetchSessions();
    fetchBenchmarks();
    fetchBenchmarkJobs();
    fetchAbTests();
    const interval = setInterval(() => {
      fetchSessions();
      fetchBenchmarkJobs();
      fetchAbTests();
    }, 5000);
    return () => clearInterval(interval);
  }, [fetchJobs, fetchSessions, fetchBenchmarks, fetchBenchmarkJobs, fetchAbTests]);

  useEffect(() => {
    if (!selectedSession) {
      setComparisons([]);
      return;
    }
    fetch(`/api/v1/finetune/eval/sessions/${selectedSession.id}`)
      .then((res) => res.json())
      .then((data) => setComparisons(data.comparisons || []))
      .catch(() => setComparisons([]));
  }, [selectedSession]);

  useEffect(() => {
    if (!selectedAbTest) {
      setAbTestSummary(null);
      return;
    }
    fetch(`/api/v1/finetune/ab-tests/${selectedAbTest.id}/summary`)
      .then((res) => res.json())
      .then(setAbTestSummary)
      .catch(() => setAbTestSummary(null));
  }, [selectedAbTest]);

  const handleCreateSession = async () => {
    if (!sessionName || !selectedAdapter) {
      setError("Please provide session name and select an adapter");
      return;
    }
    const job = jobs.find((j) => j.id === selectedAdapter);
    if (!job) return;

    setActionStatus("Creating session...");
    try {
      const res = await fetch("/api/v1/finetune/eval/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: sessionName,
          base_model: job.base_model,
          adapter_id: selectedAdapter,
          adapter_path: job.adapter_path,
          comparison_type: blindMode ? "blind" : "single",
        }),
      });
      if (res.ok) {
        setActionStatus("Session created successfully");
        setSessionName("");
        fetchSessions();
      }
    } catch {
      setError("Failed to create session");
    }
  };

  const handleRunComparison = async () => {
    if (!selectedSession || !testPrompt) return;
    setActionStatus("Generating responses...");
    try {
      const res = await fetch(`/api/v1/finetune/eval/sessions/${selectedSession.id}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: testPrompt }),
      });
      if (res.ok) {
        setActionStatus("Comparison complete");
        setTestPrompt("");
        const sessionRes = await fetch(`/api/v1/finetune/eval/sessions/${selectedSession.id}`);
        const data = await sessionRes.json();
        setComparisons(data.comparisons || []);
      }
    } catch {
      setError("Comparison failed");
    }
  };

  const handleRate = async (comparisonId: string, preferred: string) => {
    if (!selectedSession) return;
    try {
      await fetch(`/api/v1/finetune/eval/sessions/${selectedSession.id}/rate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comparison_id: comparisonId, preferred }),
      });
      fetchSessions();
      const res = await fetch(`/api/v1/finetune/eval/sessions/${selectedSession.id}`);
      const data = await res.json();
      setComparisons(data.comparisons || []);
    } catch {}
  };

  const handleRunJudge = async (comparisonId: string) => {
    if (!selectedSession) return;
    setActionStatus("Running LLM judge...");
    try {
      const res = await fetch(`/api/v1/finetune/eval/sessions/${selectedSession.id}/judge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comparison_id: comparisonId, judge_model: "gpt-4" }),
      });
      if (res.ok) {
        const data = await res.json();
        setActionStatus(`Judge verdict: ${data.judge_verdict}`);
      }
    } catch {
      setError("Judge evaluation failed");
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
      const job = jobs.find((j) => j.id === benchmarkAdapter);
      const benchmarkConfigs = selectedBenchmarks.map((name) => ({
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
          base_model: job?.base_model || "unknown",
          adapter_id: benchmarkAdapter || undefined,
          benchmarks: benchmarkConfigs,
        }),
      });
      const jobData = await res.json();
      await fetch(`/api/v1/finetune/benchmarks/jobs/${jobData.id}/start`, { method: "POST" });
      setActionStatus("Benchmark started successfully");
      fetchBenchmarkJobs();
    } catch {
      setError("Failed to start benchmark");
    }
  };

  const handleCreateAbTest = async () => {
    if (!abTestName) {
      setError("Please provide a test name");
      return;
    }
    setActionStatus("Creating A/B test...");
    try {
      const variants = abTestVariants.map((v, i) => {
        const job = jobs.find((j) => j.id === v.adapterId);
        return {
          variant_id: `variant_${i}`,
          name: v.name,
          adapter_path: job?.adapter_path || null,
          weight: v.weight,
        };
      });
      const res = await fetch("/api/v1/finetune/ab-tests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: abTestName, base_model: "auto", variants }),
      });
      if (res.ok) {
        setActionStatus("A/B test created successfully");
        setAbTestName("");
        fetchAbTests();
      }
    } catch {
      setError("Failed to create A/B test");
    }
  };

  const handleStartAbTest = async (testId: string) => {
    await fetch(`/api/v1/finetune/ab-tests/${testId}/start`, { method: "POST" });
    fetchAbTests();
  };

  const handlePauseAbTest = async (testId: string) => {
    await fetch(`/api/v1/finetune/ab-tests/${testId}/pause`, { method: "POST" });
    fetchAbTests();
  };

  const handleAbTestRequest = async () => {
    if (!selectedAbTest || !abTestPrompt) return;
    setActionStatus("Running test request...");
    try {
      const res = await fetch(`/api/v1/finetune/ab-tests/${selectedAbTest.id}/request`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: abTestPrompt }),
      });
      if (res.ok) {
        const result = await res.json();
        setActionStatus(`Response from ${result.variant_id}: ${result.tokens_per_second?.toFixed(1)} t/s`);
        setAbTestPrompt("");
        const summaryRes = await fetch(`/api/v1/finetune/ab-tests/${selectedAbTest.id}/summary`);
        if (summaryRes.ok) setAbTestSummary(await summaryRes.json());
      }
    } catch {
      setError("Request failed");
    }
  };

  const completedJobs = jobs.filter((j) => j.status === "completed" && j.adapter_path);

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
              Model Evaluation
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.875rem", maxWidth: 500 }}>
            Compare, benchmark, and evaluate your fine-tuned models
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
              onClick={() => {
                fetchSessions();
                fetchBenchmarkJobs();
                fetchAbTests();
              }}
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

      {/* Tabs */}
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
          <Tab icon={<CompareIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Side-by-Side" />
          <Tab icon={<BenchmarkIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="Benchmarks" />
          <Tab icon={<ABTestIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="A/B Testing" />
          <Tab icon={<JudgeIcon sx={{ fontSize: 18 }} />} iconPosition="start" label="LLM Judge" />
        </Tabs>
      </Box>

      {/* Side-by-Side Comparison Tab */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <SectionCard title="Comparison Sessions" icon={<CompareIcon />} accentColor={accentColors.primary}>
              <Box sx={{ mb: 3, p: 2, bgcolor: "rgba(0, 0, 0, 0.2)", borderRadius: 2 }}>
                <TextField
                  label="Session Name"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder="e.g., coding-quality-test"
                  fullWidth
                  size="small"
                  sx={{ mb: 1.5 }}
                />
                <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                  <InputLabel>Select Adapter</InputLabel>
                  <Select value={selectedAdapter} onChange={(e) => setSelectedAdapter(e.target.value)} label="Select Adapter">
                    <MenuItem value="">
                      <em>Choose an adapter...</em>
                    </MenuItem>
                    {completedJobs.map((j) => (
                      <MenuItem key={j.id} value={j.id}>
                        {j.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControlLabel
                  control={<Switch checked={blindMode} onChange={(e) => setBlindMode(e.target.checked)} size="small" />}
                  label={
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <BlindIcon sx={{ fontSize: 16 }} />
                      <Typography variant="body2">Blind mode</Typography>
                    </Box>
                  }
                  sx={{ mb: 1.5 }}
                />
                <Button
                  variant="contained"
                  fullWidth
                  startIcon={<AddIcon />}
                  onClick={handleCreateSession}
                  sx={{ background: `linear-gradient(135deg, ${accentColors.primary} 0%, ${accentColors.purple} 100%)` }}
                >
                  Create Session
                </Button>
              </Box>

              {sessions.length === 0 ? (
                <Box sx={{ p: 3, textAlign: "center" }}>
                  <CompareIcon sx={{ fontSize: 40, color: "text.secondary", opacity: 0.3, mb: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    No sessions yet
                  </Typography>
                </Box>
              ) : (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {sessions.map((s) => (
                    <Box
                      key={s.id}
                      onClick={() => setSelectedSession(s)}
                      sx={{
                        p: 2,
                        borderRadius: 2,
                        bgcolor: selectedSession?.id === s.id ? alpha(accentColors.success, 0.1) : "rgba(255, 255, 255, 0.02)",
                        border: `1px solid ${selectedSession?.id === s.id ? accentColors.success : "rgba(255, 255, 255, 0.06)"}`,
                        cursor: "pointer",
                        transition: "all 0.2s",
                        "&:hover": { bgcolor: alpha(accentColors.primary, 0.08) },
                      }}
                    >
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {s.name}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {s.total_comparisons} comparisons | Base: {s.base_preferred} | Tuned: {s.finetuned_preferred} | Ties: {s.ties}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </SectionCard>
          </Grid>

          <Grid item xs={12} md={8}>
            <SectionCard
              title={selectedSession ? selectedSession.name : "Comparison View"}
              subtitle={selectedSession ? `${selectedSession.total_comparisons} comparisons` : "Select a session"}
              icon={<EvalIcon />}
              accentColor={accentColors.info}
            >
              {selectedSession ? (
                <Box>
                  <Box sx={{ display: "flex", gap: 1.5, mb: 3 }}>
                    <TextField
                      value={testPrompt}
                      onChange={(e) => setTestPrompt(e.target.value)}
                      placeholder="Enter a test prompt..."
                      fullWidth
                      size="small"
                    />
                    <Button variant="contained" onClick={handleRunComparison} disabled={!testPrompt}>
                      Compare
                    </Button>
                  </Box>

                  {comparisons.length === 0 ? (
                    <Box sx={{ p: 4, textAlign: "center" }}>
                      <Typography color="text.secondary">No comparisons yet. Enter a prompt to start.</Typography>
                    </Box>
                  ) : (
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 2, maxHeight: 500, overflow: "auto" }}>
                      {comparisons.map((comp) => (
                        <Box
                          key={comp.id}
                          sx={{
                            p: 2,
                            borderRadius: 2,
                            bgcolor: "rgba(0, 0, 0, 0.2)",
                            border: "1px solid rgba(255, 255, 255, 0.06)",
                          }}
                        >
                          <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: "block" }}>
                            Prompt: {comp.prompt}
                          </Typography>

                          <Grid container spacing={2} sx={{ mb: 2 }}>
                            <Grid item xs={6}>
                              <Box
                                sx={{
                                  p: 1.5,
                                  borderRadius: 1.5,
                                  bgcolor: "rgba(0, 0, 0, 0.3)",
                                  border: `2px solid ${comp.preferred_model === "base" ? accentColors.success : "rgba(255, 255, 255, 0.1)"}`,
                                }}
                              >
                                <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
                                  <Typography variant="caption" color="text.secondary">
                                    {blindMode ? "Response A" : "Base Model"}
                                  </Typography>
                                  <Typography variant="caption" sx={{ color: accentColors.info }}>
                                    {comp.base_response?.tokens_per_second?.toFixed(1)} t/s
                                  </Typography>
                                </Box>
                                <Typography variant="body2" sx={{ fontSize: "0.8rem", whiteSpace: "pre-wrap" }}>
                                  {comp.base_response?.response || "-"}
                                </Typography>
                              </Box>
                            </Grid>
                            <Grid item xs={6}>
                              <Box
                                sx={{
                                  p: 1.5,
                                  borderRadius: 1.5,
                                  bgcolor: "rgba(0, 0, 0, 0.3)",
                                  border: `2px solid ${comp.preferred_model === "finetuned" ? accentColors.success : "rgba(255, 255, 255, 0.1)"}`,
                                }}
                              >
                                <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
                                  <Typography variant="caption" color="text.secondary">
                                    {blindMode ? "Response B" : "Fine-tuned"}
                                  </Typography>
                                  <Typography variant="caption" sx={{ color: accentColors.success }}>
                                    {comp.finetuned_response?.tokens_per_second?.toFixed(1)} t/s
                                  </Typography>
                                </Box>
                                <Typography variant="body2" sx={{ fontSize: "0.8rem", whiteSpace: "pre-wrap" }}>
                                  {comp.finetuned_response?.response || "-"}
                                </Typography>
                              </Box>
                            </Grid>
                          </Grid>

                          {!comp.preferred_model ? (
                            <Box sx={{ display: "flex", gap: 1 }}>
                              <Button size="small" onClick={() => handleRate(comp.id, "base")}>
                                {blindMode ? "A is Better" : "Base Better"}
                              </Button>
                              <Button size="small" variant="outlined" onClick={() => handleRate(comp.id, "tie")}>
                                Tie
                              </Button>
                              <Button size="small" onClick={() => handleRate(comp.id, "finetuned")}>
                                {blindMode ? "B is Better" : "Tuned Better"}
                              </Button>
                              <Button
                                size="small"
                                variant="outlined"
                                startIcon={<JudgeIcon sx={{ fontSize: 14 }} />}
                                onClick={() => handleRunJudge(comp.id)}
                                sx={{ ml: "auto" }}
                              >
                                LLM Judge
                              </Button>
                            </Box>
                          ) : (
                            <Chip
                              label={`Preferred: ${comp.preferred_model}`}
                              size="small"
                              sx={{ bgcolor: alpha(accentColors.success, 0.1), color: accentColors.success }}
                            />
                          )}
                        </Box>
                      ))}
                    </Box>
                  )}
                </Box>
              ) : (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <EvalIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                  <Typography color="text.secondary">Select or create a session to start comparing</Typography>
                </Box>
              )}
            </SectionCard>
          </Grid>
        </Grid>
      )}

      {/* Benchmarks Tab */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <SectionCard title="Run Benchmarks" icon={<BenchmarkIcon />} accentColor={accentColors.info}>
              <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                <InputLabel>Adapter (optional)</InputLabel>
                <Select value={benchmarkAdapter} onChange={(e) => setBenchmarkAdapter(e.target.value)} label="Adapter (optional)">
                  <MenuItem value="">Base model only</MenuItem>
                  {completedJobs.map((j) => (
                    <MenuItem key={j.id} value={j.id}>
                      {j.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
                Select Benchmarks
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mb: 3 }}>
                {benchmarks.map((b) => (
                  <Tooltip key={b.name} title={b.description}>
                    <Chip
                      label={b.display_name}
                      onClick={() => toggleBenchmark(b.name)}
                      sx={{
                        bgcolor: selectedBenchmarks.includes(b.name) ? alpha(accentColors.success, 0.2) : "rgba(255, 255, 255, 0.05)",
                        color: selectedBenchmarks.includes(b.name) ? accentColors.success : "text.secondary",
                        border: `1px solid ${selectedBenchmarks.includes(b.name) ? accentColors.success : "rgba(255, 255, 255, 0.1)"}`,
                        fontWeight: 600,
                      }}
                    />
                  </Tooltip>
                ))}
              </Box>

              <Button
                variant="contained"
                fullWidth
                startIcon={<StartIcon />}
                onClick={handleRunBenchmark}
                disabled={selectedBenchmarks.length === 0}
                sx={{ background: `linear-gradient(135deg, ${accentColors.info} 0%, ${accentColors.purple} 100%)` }}
              >
                Run Benchmarks
              </Button>
            </SectionCard>
          </Grid>

          <Grid item xs={12} md={8}>
            <SectionCard title="Benchmark Results" subtitle={`${benchmarkJobs.length} jobs`} icon={<EvalIcon />} accentColor={accentColors.success}>
              {benchmarkJobs.length === 0 ? (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <BenchmarkIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                  <Typography color="text.secondary">No benchmark jobs yet</Typography>
                </Box>
              ) : (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  {benchmarkJobs.map((bj) => (
                    <Box
                      key={bj.id}
                      sx={{ p: 2, borderRadius: 2, bgcolor: "rgba(0, 0, 0, 0.2)", border: "1px solid rgba(255, 255, 255, 0.06)" }}
                    >
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1.5 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          {bj.name}
                        </Typography>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                          <StatusChip status={bj.status} />
                          {bj.status === "running" && bj.current_benchmark && (
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
                            mb: 2,
                            height: 4,
                            borderRadius: 2,
                            bgcolor: "rgba(255, 255, 255, 0.1)",
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
                                  {r.improvement !== undefined && (
                                    <span style={{ color: r.improvement > 0 ? accentColors.success : accentColors.rose, fontWeight: 600 }}>
                                      ({r.improvement > 0 ? "+" : ""}{r.improvement.toFixed(1)}%)
                                    </span>
                                  )}
                                </Box>
                              }
                              sx={{ bgcolor: "rgba(255, 255, 255, 0.05)", border: "1px solid rgba(255, 255, 255, 0.1)" }}
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

      {/* A/B Testing Tab */}
      {activeTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={5}>
            <SectionCard title="Create A/B Test" icon={<ABTestIcon />} accentColor={accentColors.purple}>
              <TextField
                label="Test Name"
                value={abTestName}
                onChange={(e) => setAbTestName(e.target.value)}
                placeholder="e.g., v1-vs-v2-test"
                fullWidth
                size="small"
                sx={{ mb: 2 }}
              />

              <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
                Variants
              </Typography>
              {abTestVariants.map((v, i) => (
                <Box key={i} sx={{ mb: 2, p: 2, bgcolor: "rgba(0, 0, 0, 0.2)", borderRadius: 2 }}>
                  <TextField
                    value={v.name}
                    onChange={(e) => {
                      const updated = [...abTestVariants];
                      updated[i].name = e.target.value;
                      setAbTestVariants(updated);
                    }}
                    placeholder="Variant name"
                    fullWidth
                    size="small"
                    sx={{ mb: 1 }}
                  />
                  <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                    <InputLabel>Adapter</InputLabel>
                    <Select
                      value={v.adapterId}
                      onChange={(e) => {
                        const updated = [...abTestVariants];
                        updated[i].adapterId = e.target.value;
                        setAbTestVariants(updated);
                      }}
                      label="Adapter"
                    >
                      <MenuItem value="">Base Model</MenuItem>
                      {completedJobs.map((j) => (
                        <MenuItem key={j.id} value={j.id}>
                          {j.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                    <Typography variant="caption">Weight:</Typography>
                    <Slider
                      value={v.weight * 100}
                      onChange={(_, value) => {
                        const updated = [...abTestVariants];
                        updated[i].weight = (value as number) / 100;
                        setAbTestVariants(updated);
                      }}
                      min={0}
                      max={100}
                      size="small"
                      sx={{ flex: 1 }}
                    />
                    <Typography variant="caption" sx={{ minWidth: 40 }}>
                      {(v.weight * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                </Box>
              ))}

              <Button
                size="small"
                startIcon={<AddIcon />}
                onClick={() => setAbTestVariants([...abTestVariants, { name: `Variant ${abTestVariants.length + 1}`, adapterId: "", weight: 0.5 }])}
                sx={{ mb: 2 }}
              >
                Add Variant
              </Button>

              <Button
                variant="contained"
                fullWidth
                onClick={handleCreateAbTest}
                sx={{ background: `linear-gradient(135deg, ${accentColors.purple} 0%, ${accentColors.primary} 100%)` }}
              >
                Create A/B Test
              </Button>

              <Divider sx={{ my: 3, borderColor: "rgba(255, 255, 255, 0.06)" }} />

              <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
                Active Tests
              </Typography>
              {abTests.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No A/B tests yet
                </Typography>
              ) : (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {abTests.map((t) => (
                    <Box
                      key={t.id}
                      onClick={() => setSelectedAbTest(t)}
                      sx={{
                        p: 1.5,
                        borderRadius: 2,
                        bgcolor: selectedAbTest?.id === t.id ? alpha(accentColors.success, 0.1) : "rgba(255, 255, 255, 0.02)",
                        border: `1px solid ${selectedAbTest?.id === t.id ? accentColors.success : "rgba(255, 255, 255, 0.06)"}`,
                        cursor: "pointer",
                      }}
                    >
                      <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                          {t.name}
                        </Typography>
                        <StatusChip status={t.status} />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        {t.variants.length} variants
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </SectionCard>
          </Grid>

          <Grid item xs={12} md={7}>
            <SectionCard
              title={selectedAbTest ? selectedAbTest.name : "A/B Test Details"}
              icon={<EvalIcon />}
              accentColor={accentColors.info}
              action={
                selectedAbTest && (
                  <Box sx={{ display: "flex", gap: 1 }}>
                    {selectedAbTest.status === "draft" && (
                      <Button size="small" startIcon={<StartIcon />} onClick={() => handleStartAbTest(selectedAbTest.id)}>
                        Start
                      </Button>
                    )}
                    {selectedAbTest.status === "running" && (
                      <Button size="small" color="warning" startIcon={<PauseIcon />} onClick={() => handlePauseAbTest(selectedAbTest.id)}>
                        Pause
                      </Button>
                    )}
                  </Box>
                )
              }
            >
              {selectedAbTest ? (
                <Box>
                  {selectedAbTest.status === "running" && (
                    <Box sx={{ display: "flex", gap: 1.5, mb: 3 }}>
                      <TextField
                        value={abTestPrompt}
                        onChange={(e) => setAbTestPrompt(e.target.value)}
                        placeholder="Enter test prompt..."
                        fullWidth
                        size="small"
                      />
                      <Button variant="contained" onClick={handleAbTestRequest} disabled={!abTestPrompt}>
                        Send
                      </Button>
                    </Box>
                  )}

                  {abTestSummary && (
                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 2 }}>
                        Performance Comparison
                      </Typography>
                      {abTestSummary.variants.map((v) => {
                        const maxTps = Math.max(...abTestSummary.variants.map((x) => x.avg_tokens_per_second));
                        const maxLatency = Math.max(...abTestSummary.variants.map((x) => x.p95_latency_ms));
                        return (
                          <Box key={v.variant_id} sx={{ mb: 2, p: 2, bgcolor: "rgba(0, 0, 0, 0.2)", borderRadius: 2 }}>
                            <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1.5 }}>
                              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                {v.name}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {v.total_requests} requests
                              </Typography>
                            </Box>

                            <MetricBar label="Tokens/sec" value={v.avg_tokens_per_second} max={maxTps} unit=" t/s" color={accentColors.success} />
                            <MetricBar label="Avg Latency" value={v.avg_latency_ms} max={maxLatency} unit=" ms" color={accentColors.info} />
                            <MetricBar label="P95 Latency" value={v.p95_latency_ms} max={maxLatency} unit=" ms" color={accentColors.warning} />

                            <Box sx={{ display: "flex", justifyContent: "space-between", mt: 1.5 }}>
                              <Typography variant="caption" color="text.secondary">
                                Success: {v.success_rate}%
                              </Typography>
                              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                <ThumbUpIcon sx={{ fontSize: 14, color: accentColors.success }} />
                                <Typography variant="caption">{v.thumbs_up}</Typography>
                                <ThumbDownIcon sx={{ fontSize: 14, color: accentColors.rose }} />
                                <Typography variant="caption">{v.thumbs_down}</Typography>
                              </Box>
                            </Box>
                          </Box>
                        );
                      })}

                      {abTestSummary.winner && (
                        <Alert severity="success" icon={<TrophyIcon />} sx={{ mt: 2 }}>
                          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                            Winner: {abTestSummary.winner_name}
                          </Typography>
                        </Alert>
                      )}

                      {!abTestSummary.has_enough_samples && abTestSummary.samples_needed && (
                        <Alert severity="warning" sx={{ mt: 2 }}>
                          Need {abTestSummary.samples_needed} more samples for statistical significance
                        </Alert>
                      )}
                    </Box>
                  )}
                </Box>
              ) : (
                <Box sx={{ p: 4, textAlign: "center" }}>
                  <ABTestIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                  <Typography color="text.secondary">Select or create an A/B test to view metrics</Typography>
                </Box>
              )}
            </SectionCard>
          </Grid>
        </Grid>
      )}

      {/* LLM Judge Tab */}
      {activeTab === 3 && (
        <SectionCard title="LLM-as-Judge Evaluation" subtitle="Use AI to automatically evaluate responses" icon={<JudgeIcon />} accentColor={accentColors.warning}>
          <Alert severity="info" icon={<MagicIcon />} sx={{ mb: 3 }}>
            Use GPT-4 or Claude to automatically evaluate and compare responses. Go to the Side-by-Side tab, run comparisons, and click "LLM Judge" on any comparison.
          </Alert>

          <Box sx={{ p: 3, bgcolor: "rgba(0, 0, 0, 0.2)", borderRadius: 2 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
              Evaluation Criteria
            </Typography>
            <Grid container spacing={2}>
              {[
                { label: "Relevance", desc: "How well does the response address the prompt?" },
                { label: "Accuracy", desc: "Is the information correct and factual?" },
                { label: "Coherence", desc: "Is the response logically structured?" },
                { label: "Completeness", desc: "Does it fully answer the question?" },
                { label: "Helpfulness", desc: "How useful is the response overall?" },
              ].map((criteria) => (
                <Grid item xs={12} sm={6} key={criteria.label}>
                  <Box sx={{ p: 2, borderRadius: 1.5, bgcolor: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.06)" }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, color: accentColors.warning }}>
                      {criteria.label}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {criteria.desc}
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        </SectionCard>
      )}
    </Box>
  );
};

export default ModelEvaluationPage;
