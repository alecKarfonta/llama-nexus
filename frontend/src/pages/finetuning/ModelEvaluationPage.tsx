import React, { useEffect, useState, useCallback, useRef } from "react";
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
  Chip,
  Grid,
  Alert,
  Tooltip,
  LinearProgress,
  Divider,
  Switch,
  FormControlLabel,
  Slider,
  useMediaQuery,
  alpha,
} from "@mui/material";
import {
  Assessment as EvalIcon,
  Compare as CompareIcon,
  Science as BenchmarkIcon,
  Balance as ABTestIcon,
  Psychology as JudgeIcon,
  Refresh as RefreshIcon,
  PlayArrow as StartIcon,
  Pause as PauseIcon,
  Add as AddIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
  EmojiEvents as TrophyIcon,
  AutoFixHigh as MagicIcon,
  VisibilityOff as BlindIcon,
} from "@mui/icons-material";
import { useLocation, useNavigate } from "react-router-dom";
import type { EvaluationRouteState } from "@/types/trainingHandoff";
import { adapterHandoffKey, findAdapterForHandoff } from "@/types/trainingHandoff";
import { getRoutePath } from "@/config/navigation";
import { PageShell, SectionCard, StatusChip } from "@/components/common";
import { accentColors } from "@/utils/theme";

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

// LM Evaluation Harness Types (Standardized Benchmarks)
type LMEvalTask = {
  name: string;
  display_name: string;
  description: string;
  default_fewshot: number;
  category: string;
};

type LMEvalJob = {
  id: string;
  tasks: string[];
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  results: {
    tasks: Record<string, {
      score: number;
      metrics: Record<string, number>;
    }>;
    model_name?: string;
    date?: string;
  };
  error?: string;
  created_at: string;
  completed_at?: string;
  log_output?: string;
  model_path?: string;
  gpu_device?: number;
  endpoint_url?: string;
};

type TerminalBenchDataset = {
  name: string;
  display_name: string;
  description: string;
  supports_modes: string[];
  default_smoke_limit: number;
};

type TerminalBenchJob = {
  id: string;
  name: string;
  dataset: string;
  mode: "smoke" | "full";
  task_limit?: number | null;
  n_concurrent: number;
  timeout_multiplier: number;
  model_name: string;
  endpoint_url: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  results: {
    pass_rate?: number;
    total_trials?: number;
    passed_trials?: number;
    failed_trials?: number;
  };
  error?: string;
  artifacts?: Record<string, string>;
  created_at: string;
  completed_at?: string;
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

const evaluationIconButtonSx = {
  width: 32,
  height: 32,
  borderRadius: "8px",
  border: "1px solid",
  borderColor: "divider",
  bgcolor: "action.hover",
  color: "text.secondary",
  "&:hover": {
    bgcolor: "action.selected",
  },
} as const;

const evaluationModeButtonSx = {
  height: 32,
  minWidth: 0,
  px: 1,
  borderRadius: "8px",
  border: "1px solid",
  textTransform: "none",
  fontSize: "0.74rem",
  fontWeight: 800,
  lineHeight: 1,
  "& .MuiButton-startIcon": {
    mr: 0.5,
    ml: 0,
  },
} as const;

export const ModelEvaluationPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const isNarrow = useMediaQuery("(max-width:599.95px)");
  const benchmarkComparisonPath = getRoutePath("finetuning-benchmarks");
  const routeState = location.state as EvaluationRouteState | null;
  const incomingAdapter = routeState?.adapter;
  const appliedAdapterHandoffRef = useRef("");
  const [activeTab, setActiveTab] = useState(0);
  const [jobs, setJobs] = useState<FineTuningJob[]>([]);
  const [sessions, setSessions] = useState<ComparisonSession[]>([]);
  const [selectedSession, setSelectedSession] = useState<ComparisonSession | null>(null);
  const [comparisons, setComparisons] = useState<PromptComparison[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const [handoffNotice, setHandoffNotice] = useState<string | null>(null);

  // Comparison form
  const [sessionName, setSessionName] = useState("");
  const [selectedAdapter, setSelectedAdapter] = useState("");
  const [testPrompt, setTestPrompt] = useState("");
  const [blindMode, setBlindMode] = useState(false);

  // LM-Eval (Standardized Benchmarks)
  const [lmEvalAvailable, setLmEvalAvailable] = useState(false);
  const [lmEvalTasks, setLmEvalTasks] = useState<LMEvalTask[]>([]);
  const [lmEvalJobs, setLmEvalJobs] = useState<LMEvalJob[]>([]);
  const [selectedLmEvalTasks, setSelectedLmEvalTasks] = useState<string[]>(["mmlu", "hellaswag"]);
  const [lmEvalFewshot, setLmEvalFewshot] = useState(5);
  const [lmEvalLimit, setLmEvalLimit] = useState<number | null>(100);
  const [lmEvalRunning, setLmEvalRunning] = useState(false);
  const [lmEvalProgress, setLmEvalProgress] = useState("");
  const [terminalBenchAvailable, setTerminalBenchAvailable] = useState(false);
  const [terminalBenchDatasets, setTerminalBenchDatasets] = useState<TerminalBenchDataset[]>([]);
  const [terminalBenchJobs, setTerminalBenchJobs] = useState<TerminalBenchJob[]>([]);
  const [terminalBenchMode, setTerminalBenchMode] = useState<"smoke" | "full">("smoke");
  const [terminalBenchTaskLimit, setTerminalBenchTaskLimit] = useState<number>(5);
  const [terminalBenchConcurrency, setTerminalBenchConcurrency] = useState<number>(1);
  const [terminalBenchRunning, setTerminalBenchRunning] = useState(false);
  const [terminalBenchProgress, setTerminalBenchProgress] = useState("");

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
    } catch { }
  }, []);

  const fetchSessions = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/eval/sessions");
      if (res.ok) {
        const data = await res.json();
        setSessions(data.sessions || []);
      }
    } catch { }
  }, []);

  const fetchAbTests = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/ab-tests");
      if (res.ok) {
        const data = await res.json();
        setAbTests(data.tests || []);
      }
    } catch { }
  }, []);

  // LM-Eval API functions
  const fetchLmEvalStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/lm-eval/status");
      if (res.ok) {
        const data = await res.json();
        setLmEvalAvailable(data.available);
      }
    } catch {
      setLmEvalAvailable(false);
    }
  }, []);

  const fetchLmEvalTasks = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/lm-eval/tasks");
      if (res.ok) {
        const data = await res.json();
        setLmEvalTasks(data.tasks || []);
      }
    } catch { }
  }, []);

  const fetchLmEvalJobs = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/lm-eval/jobs");
      if (res.ok) {
        const data = await res.json();
        setLmEvalJobs(data.jobs || []);
      }
    } catch { }
  }, []);

  const fetchTerminalBenchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/terminal-bench/status");
      if (res.ok) {
        const data = await res.json();
        setTerminalBenchAvailable(Boolean(data.available));
      }
    } catch {
      setTerminalBenchAvailable(false);
    }
  }, []);

  const fetchTerminalBenchDatasets = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/terminal-bench/datasets");
      if (res.ok) {
        const data = await res.json();
        setTerminalBenchDatasets(data.datasets || []);
      }
    } catch { }
  }, []);

  const fetchTerminalBenchJobs = useCallback(async () => {
    try {
      const res = await fetch("/api/v1/finetune/terminal-bench/jobs");
      if (res.ok) {
        const data = await res.json();
        setTerminalBenchJobs(data.jobs || []);
      }
    } catch { }
  }, []);

  useEffect(() => {
    fetchJobs();
    fetchSessions();
    fetchAbTests();
    // Initialize lm-eval
    fetchLmEvalStatus();
    fetchLmEvalTasks();
    fetchLmEvalJobs();
    fetchTerminalBenchStatus();
    fetchTerminalBenchDatasets();
    fetchTerminalBenchJobs();

    const interval = setInterval(() => {
      fetchSessions();
      fetchAbTests();
      fetchLmEvalJobs();
      fetchTerminalBenchJobs();
    }, 5000);
    return () => clearInterval(interval);
  }, [
    fetchJobs,
    fetchSessions,
    fetchAbTests,
    fetchLmEvalStatus,
    fetchLmEvalTasks,
    fetchLmEvalJobs,
    fetchTerminalBenchStatus,
    fetchTerminalBenchDatasets,
    fetchTerminalBenchJobs,
  ]);

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

  useEffect(() => {
    if (!incomingAdapter || jobs.length === 0) return;

    const handoffKey = adapterHandoffKey(incomingAdapter);
    if (appliedAdapterHandoffRef.current === handoffKey) return;

    const matchedJob = findAdapterForHandoff(
      incomingAdapter,
      jobs,
      (job) => job.status.toLowerCase() === "completed" && Boolean(job.adapter_path)
    );

    appliedAdapterHandoffRef.current = handoffKey;
    if (!matchedJob) {
      setHandoffNotice(`${incomingAdapter.name || "Adapter"} is not available for side-by-side evaluation yet`);
      return;
    }

    setSelectedAdapter(matchedJob.id);
    setSessionName((current) => current || `${incomingAdapter.name || matchedJob.name} evaluation`);
    setActiveTab(0);
    setHandoffNotice(`${incomingAdapter.name || matchedJob.name} selected for side-by-side evaluation`);
  }, [
    incomingAdapter,
    incomingAdapter?.adapterPath,
    incomingAdapter?.baseModel,
    incomingAdapter?.id,
    incomingAdapter?.name,
    incomingAdapter?.source,
    incomingAdapter?.trainingJobId,
    jobs,
  ]);

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
    } catch { }
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

  // LM-Eval toggle and run handler
  const toggleLmEvalTask = (name: string) => {
    setSelectedLmEvalTasks((prev) =>
      prev.includes(name) ? prev.filter((t) => t !== name) : [...prev, name]
    );
  };

  const handleRunLmEval = async () => {
    if (selectedLmEvalTasks.length === 0) {
      setError("Select at least one benchmark task");
      return;
    }

    setLmEvalRunning(true);
    setLmEvalProgress("Creating benchmark job...");
    setError(null);

    try {
      const createRes = await fetch("/api/v1/finetune/lm-eval/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `LM-Eval ${new Date().toISOString().slice(0, 16)}`,
          tasks: selectedLmEvalTasks,
          num_fewshot: lmEvalFewshot,
          limit: lmEvalLimit,
        }),
      });

      if (!createRes.ok) {
        throw new Error("Failed to create benchmark job");
      }

      const job = await createRes.json();
      setLmEvalProgress(`Starting job ${job.id}...`);

      // Start job with streaming
      const eventSource = new EventSource(
        `/api/v1/finetune/lm-eval/jobs/${job.id}/start`
      );

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "progress") {
          setLmEvalProgress(`Progress: ${data.progress}% - ${data.message}`);
        } else if (data.type === "log") {
          setLmEvalProgress(data.message);
        } else if (data.type === "complete") {
          setLmEvalProgress("Benchmark completed!");
          setLmEvalRunning(false);
          fetchLmEvalJobs();
          eventSource.close();
        } else if (data.type === "error") {
          setError(`Benchmark failed: ${data.error}`);
          setLmEvalRunning(false);
          eventSource.close();
        }
      };

      eventSource.onerror = () => {
        setError("Connection lost to benchmark stream");
        setLmEvalRunning(false);
        eventSource.close();
        fetchLmEvalJobs();
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to run benchmark");
      setLmEvalRunning(false);
    }
  };

  const handleRunTerminalBench = async () => {
    setTerminalBenchRunning(true);
    setTerminalBenchProgress("Creating Terminal-Bench job...");
    setError(null);

    try {
      const createRes = await fetch("/api/v1/finetune/terminal-bench/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `Terminal-Bench ${new Date().toISOString().slice(0, 16)}`,
          dataset: "terminal-bench@2.0",
          mode: terminalBenchMode,
          task_limit: terminalBenchMode === "smoke" ? terminalBenchTaskLimit : null,
          n_concurrent: terminalBenchConcurrency,
          timeout_multiplier: 1.0,
          harbor_agent: "local-openai-agent",
          api_key: "placeholder-api-key",
        }),
      });
      if (!createRes.ok) {
        throw new Error("Failed to create Terminal-Bench job");
      }
      const job = await createRes.json();
      setTerminalBenchProgress(`Starting job ${job.id}...`);

      const eventSource = new EventSource(`/api/v1/finetune/terminal-bench/jobs/${job.id}/start`);
      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "progress") {
          setTerminalBenchProgress(`Progress: ${data.progress?.toFixed?.(1) ?? data.progress}% - ${data.message || ""}`);
        } else if (data.type === "log") {
          setTerminalBenchProgress(data.message || "Running...");
        } else if (data.type === "complete") {
          setTerminalBenchProgress("Terminal-Bench completed");
          setTerminalBenchRunning(false);
          fetchTerminalBenchJobs();
          eventSource.close();
        } else if (data.type === "error") {
          setError(`Terminal-Bench failed: ${data.error}`);
          setTerminalBenchRunning(false);
          fetchTerminalBenchJobs();
          eventSource.close();
        } else if (data.type === "cancelled") {
          setActionStatus("Terminal-Bench job cancelled");
          setTerminalBenchRunning(false);
          fetchTerminalBenchJobs();
          eventSource.close();
        }
      };
      eventSource.onerror = () => {
        setError("Terminal-Bench stream connection lost");
        setTerminalBenchRunning(false);
        fetchTerminalBenchJobs();
        eventSource.close();
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to run Terminal-Bench");
      setTerminalBenchRunning(false);
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
  const comparisonTotal = sessions.reduce((total, session) => total + session.total_comparisons, 0);
  const benchmarkJobTotal = lmEvalJobs.length + terminalBenchJobs.length;
  const recentBenchmarkJobs = [...terminalBenchJobs, ...lmEvalJobs]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
    .slice(0, 3);
  const evaluationQueueStats = [
    { label: "Adapters", value: completedJobs.length, color: accentColors.success, summaryLabel: completedJobs.length === 1 ? "adapter" : "adapters" },
    { label: "Sessions", value: sessions.length, color: accentColors.primary, summaryLabel: sessions.length === 1 ? "session" : "sessions" },
    { label: "Benchmarks", value: benchmarkJobTotal, color: accentColors.info, summaryLabel: benchmarkJobTotal === 1 ? "benchmark job" : "benchmark jobs" },
    { label: "A/B", value: abTests.length, color: accentColors.purple, summaryLabel: abTests.length === 1 ? "A/B test" : "A/B tests" },
  ].filter((item) => item.value > 0);
  const evaluationQueueSubtitle = evaluationQueueStats.map((item) => `${item.value} ${item.summaryLabel}`).join(" / ");
  const evaluationModes = [
    { label: "Side-by-side", shortLabel: "Side-by-side", icon: <CompareIcon sx={{ fontSize: 16 }} /> },
    { label: "Benchmark jobs", shortLabel: "Benchmarks", icon: <BenchmarkIcon sx={{ fontSize: 16 }} /> },
    { label: "A/B testing", shortLabel: "A/B", icon: <ABTestIcon sx={{ fontSize: 16 }} /> },
    { label: "LLM judge", shortLabel: "Judge", icon: <JudgeIcon sx={{ fontSize: 16 }} /> },
  ];
  const refreshEvaluationData = () => {
    fetchSessions();
    fetchAbTests();
    fetchLmEvalJobs();
    fetchTerminalBenchJobs();
  };
  const hasEvaluationSummary =
    sessions.length > 0 ||
    completedJobs.length > 0 ||
    benchmarkJobTotal > 0 ||
    comparisonTotal > 0 ||
    abTests.length > 0;
  const evaluationHeaderActions = (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 0.5,
        flexWrap: "nowrap",
        justifyContent: "flex-end",
      }}
    >
      {evaluationModes.map((mode, index) => (
        <Tooltip key={mode.label} title={mode.label}>
          <Button
            size="small"
            aria-label={mode.label}
            aria-pressed={activeTab === index}
            onClick={() => setActiveTab(index)}
            startIcon={isNarrow ? undefined : mode.icon}
            sx={{
              ...evaluationModeButtonSx,
              width: { xs: 32, sm: "auto" },
              px: { xs: 0, sm: 1 },
              color: activeTab === index ? accentColors.primary : "text.secondary",
              borderColor: activeTab === index ? alpha(accentColors.primary, 0.65) : "divider",
              bgcolor: activeTab === index ? alpha(accentColors.primary, 0.16) : "action.hover",
              "&:hover": {
                bgcolor: activeTab === index ? alpha(accentColors.primary, 0.22) : "action.selected",
              },
            }}
          >
            {isNarrow ? mode.icon : mode.shortLabel}
          </Button>
        </Tooltip>
      ))}
      <Tooltip title="Refresh evaluation data">
        <IconButton
          aria-label="Refresh evaluation data"
          size="small"
          onClick={refreshEvaluationData}
          sx={evaluationIconButtonSx}
        >
          <RefreshIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
    </Box>
  );
  const evaluationSummaryRow = hasEvaluationSummary && !isNarrow ? (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap", mb: 1 }}>
      {sessions.length > 0 && (
        <Chip size="small" label={`${sessions.length} sessions`} sx={{ height: 22, fontSize: "0.68rem", fontWeight: 800 }} />
      )}
      {completedJobs.length > 0 && (
        <Chip size="small" label={`${completedJobs.length} adapters`} sx={{ height: 22, fontSize: "0.68rem", fontWeight: 800 }} />
      )}
      {benchmarkJobTotal > 0 && (
        <Chip size="small" label={`${benchmarkJobTotal} benchmark jobs`} sx={{ height: 22, fontSize: "0.68rem", fontWeight: 800 }} />
      )}
      {(comparisonTotal > 0 || abTests.length > 0) && (
        <Typography variant="caption" color="text.secondary" sx={{ minWidth: 0 }}>
          {comparisonTotal} comparisons / {abTests.length} A/B tests
        </Typography>
      )}
    </Box>
  ) : null;
  const queueItemSx = {
    p: 1,
    borderRadius: "8px",
    border: "1px solid",
    borderColor: "divider",
    bgcolor: "action.hover",
    minWidth: 0,
  } as const;

  return (
    <PageShell sx={{ py: 1.25 }}>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {handoffNotice && (
        <Alert severity="info" sx={{ mb: 1.25 }} onClose={() => setHandoffNotice(null)}>
          {handoffNotice}
        </Alert>
      )}
      {actionStatus && (
        <Alert severity="success" sx={{ mb: 1.25 }} onClose={() => setActionStatus(null)}>
          {actionStatus}
        </Alert>
      )}

      {/* Side-by-Side Comparison Tab */}
      {activeTab === 0 && (
        <Grid container spacing={1.25} alignItems="flex-start">
          <Grid item xs={12} md={selectedSession ? 4 : 8} lg={selectedSession ? 4 : 8}>
            <SectionCard title="Comparison Sessions" icon={<CompareIcon />} accentColor={accentColors.primary} action={evaluationHeaderActions}>
              {evaluationSummaryRow}
              <Box
                sx={{
                  mb: 1,
                  p: 1,
                  bgcolor: "action.hover",
                  borderRadius: "8px",
                  border: "1px solid",
                  borderColor: "divider",
                  display: "grid",
                  gridTemplateColumns: {
                    xs: "1fr",
                    md: selectedSession ? "1fr" : "minmax(180px, 1.15fr) minmax(180px, 1fr) auto",
                  },
                  alignItems: "center",
                  gap: 1,
                }}
              >
                <TextField
                  label="Session Name"
                  value={sessionName}
                  onChange={(e) => setSessionName(e.target.value)}
                  placeholder="e.g., coding-quality-test"
                  fullWidth
                  size="small"
                />
                <FormControl fullWidth size="small">
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
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 1, minWidth: 0 }}>
                  <FormControlLabel
                    control={<Switch checked={blindMode} onChange={(e) => setBlindMode(e.target.checked)} size="small" />}
                    label={
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <BlindIcon sx={{ fontSize: 16 }} />
                        <Typography variant="body2">Blind mode</Typography>
                      </Box>
                    }
                    sx={{ m: 0 }}
                  />
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={handleCreateSession}
                    sx={{
                      height: 32,
                      borderRadius: "8px",
                      fontWeight: 800,
                      px: 1.25,
                      minWidth: 0,
                      textTransform: "none",
                      background: `linear-gradient(135deg, ${accentColors.primary} 0%, ${accentColors.purple} 100%)`,
                    }}
                  >
                    Create
                  </Button>
                </Box>
              </Box>

              {sessions.length > 0 && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                  {sessions.map((s) => (
                    <Box
                      key={s.id}
                      onClick={() => setSelectedSession(s)}
                      sx={{
                        p: 1,
                        borderRadius: "8px",
                        bgcolor: selectedSession?.id === s.id ? alpha(accentColors.success, 0.1) : "background.paper",
                        border: "1px solid",
                        borderColor: selectedSession?.id === s.id ? accentColors.success : "divider",
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

          {!selectedSession && (
            <Grid item xs={12} md={4} lg={4}>
              <SectionCard
                title="Evaluation queue"
                subtitle={evaluationQueueSubtitle || undefined}
                icon={<EvalIcon />}
                accentColor={accentColors.info}
                collapsible={isNarrow}
                defaultExpanded={!isNarrow}
              >
                {evaluationQueueStats.length > 0 && (
                  <Box sx={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 1, mb: recentBenchmarkJobs.length > 0 ? 1 : 0 }}>
                    {evaluationQueueStats.map((item) => (
                      <Box key={item.label} sx={{ ...queueItemSx, borderColor: alpha(item.color, 0.24), bgcolor: alpha(item.color, 0.08) }}>
                        <Typography variant="caption" color="text.secondary" sx={{ display: "block", lineHeight: 1.1 }}>
                          {item.label}
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 900 }}>
                          {item.value}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}

                {recentBenchmarkJobs.length > 0 && (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                    {recentBenchmarkJobs.map((job) => (
                      <Box key={job.id} sx={queueItemSx}>
                        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
                          <Typography variant="body2" sx={{ fontWeight: 700 }} noWrap>
                            {"tasks" in job ? job.tasks.join(", ") : job.name}
                          </Typography>
                          <StatusChip status={job.status} />
                        </Box>
                        <Typography variant="caption" color="text.secondary" noWrap>
                          {new Date(job.created_at).toLocaleString()}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                )}
              </SectionCard>
            </Grid>
          )}

          {selectedSession && (
            <Grid item xs={12} md={8}>
            <SectionCard
              title={selectedSession.name}
              subtitle={`${selectedSession.total_comparisons} comparisons`}
              icon={<EvalIcon />}
              accentColor={accentColors.info}
            >
              <Box>
                <Box sx={{ display: "flex", gap: 1, mb: 1.25 }}>
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
                  <Box
                    sx={{
                      px: 1,
                      py: 0.75,
                      borderRadius: "8px",
                      border: "1px dashed",
                      borderColor: "divider",
                    }}
                  >
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
                      No comparisons
                    </Typography>
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
            </SectionCard>
            </Grid>
          )}
        </Grid>
      )}

      {/* Benchmarks Tab - LM Evaluation Harness */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={5}>
            <SectionCard
              fullHeight
              title="LM Evaluation Harness"
              subtitle={lmEvalAvailable ? "Standardized Benchmarks" : "Checking availability..."}
              icon={<BenchmarkIcon />}
              accentColor={accentColors.info}
              action={evaluationHeaderActions}
            >
              {evaluationSummaryRow}
              {!lmEvalAvailable && lmEvalTasks.length === 0 ? (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  lm-evaluation-harness is initializing or not available. Rebuild the backend container to install.
                </Alert>
              ) : null}

              <Alert severity="info" sx={{ mb: 3 }}>
                Benchmarks run against the currently deployed model.
              </Alert>

              <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
                Select Benchmark Tasks
              </Typography>

              {/* Group tasks by category */}
              {["reasoning", "knowledge", "math", "truthfulness", "reading"].map((category) => {
                const categoryTasks = lmEvalTasks.filter((t) => t.category === category);
                if (categoryTasks.length === 0) return null;
                return (
                  <Box key={category} sx={{ mb: 2 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", fontWeight: 600 }}>
                      {category}
                    </Typography>
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 0.5 }}>
                      {categoryTasks.map((t) => (
                        <Tooltip key={t.name} title={t.description}>
                          <Chip
                            label={t.display_name}
                            onClick={() => toggleLmEvalTask(t.name)}
                            size="small"
                            sx={{
                              bgcolor: selectedLmEvalTasks.includes(t.name) ? alpha(accentColors.success, 0.2) : "rgba(255, 255, 255, 0.05)",
                              color: selectedLmEvalTasks.includes(t.name) ? accentColors.success : "text.secondary",
                              border: `1px solid ${selectedLmEvalTasks.includes(t.name) ? accentColors.success : "rgba(255, 255, 255, 0.1)"}`,
                              fontWeight: 600,
                              cursor: "pointer",
                            }}
                          />
                        </Tooltip>
                      ))}
                    </Box>
                  </Box>
                );
              })}

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Few-shot Examples: {lmEvalFewshot}
              </Typography>
              <Slider
                value={lmEvalFewshot}
                onChange={(_, value) => setLmEvalFewshot(value as number)}
                min={0}
                max={25}
                step={1}
                marks={[
                  { value: 0, label: "0" },
                  { value: 5, label: "5" },
                  { value: 10, label: "10" },
                  { value: 25, label: "25" },
                ]}
                sx={{ mb: 2 }}
              />

              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Sample Limit: {lmEvalLimit ?? "Full dataset"}
              </Typography>
              <Slider
                value={lmEvalLimit ?? 0}
                onChange={(_, value) => setLmEvalLimit((value as number) || null)}
                min={0}
                max={500}
                step={10}
                marks={[
                  { value: 0, label: "Full" },
                  { value: 100, label: "100" },
                  { value: 250, label: "250" },
                  { value: 500, label: "500" },
                ]}
                sx={{ mb: 3 }}
              />

              {lmEvalRunning && (
                <Box sx={{ mb: 2 }}>
                  <LinearProgress sx={{ mb: 1, borderRadius: 1 }} />
                  <Typography variant="caption" color="text.secondary">
                    {lmEvalProgress}
                  </Typography>
                </Box>
              )}

              <Button
                variant="contained"
                fullWidth
                startIcon={<StartIcon />}
                onClick={handleRunLmEval}
                disabled={selectedLmEvalTasks.length === 0 || lmEvalRunning}
                sx={{ background: `linear-gradient(135deg, ${accentColors.info} 0%, ${accentColors.purple} 100%)` }}
              >
                {lmEvalRunning ? "Running..." : "Run Standardized Benchmarks"}
              </Button>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Terminal-Bench 2.0
              </Typography>
              {!terminalBenchAvailable ? (
                <Alert severity="warning" sx={{ mb: 2 }}>
                  Terminal-Bench worker is not available yet. Start the `terminal-bench-worker` service in the `eval` profile.
                </Alert>
              ) : (
                <Box sx={{ mb: 2 }}>
                  <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                    <InputLabel>Run Mode</InputLabel>
                    <Select
                      value={terminalBenchMode}
                      onChange={(e) => setTerminalBenchMode(e.target.value as "smoke" | "full")}
                      label="Run Mode"
                    >
                      <MenuItem value="smoke">Smoke</MenuItem>
                      <MenuItem value="full">Full</MenuItem>
                    </Select>
                  </FormControl>

                  <TextField
                    type="number"
                    label="Task Limit (smoke mode)"
                    size="small"
                    fullWidth
                    value={terminalBenchTaskLimit}
                    onChange={(e) => setTerminalBenchTaskLimit(Number(e.target.value || 0))}
                    disabled={terminalBenchMode !== "smoke"}
                    sx={{ mb: 1.5 }}
                    inputProps={{ min: 1, max: 2000 }}
                  />

                  <TextField
                    type="number"
                    label="Concurrency"
                    size="small"
                    fullWidth
                    value={terminalBenchConcurrency}
                    onChange={(e) => setTerminalBenchConcurrency(Number(e.target.value || 1))}
                    sx={{ mb: 1.5 }}
                    inputProps={{ min: 1, max: 64 }}
                  />

                  {terminalBenchRunning && (
                    <Box sx={{ mb: 1.5 }}>
                      <LinearProgress sx={{ mb: 1, borderRadius: 1 }} />
                      <Typography variant="caption" color="text.secondary">
                        {terminalBenchProgress}
                      </Typography>
                    </Box>
                  )}

                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={<StartIcon />}
                    onClick={handleRunTerminalBench}
                    disabled={!terminalBenchAvailable || terminalBenchRunning}
                    sx={{ background: `linear-gradient(135deg, ${accentColors.warning} 0%, ${accentColors.rose} 100%)` }}
                  >
                    {terminalBenchRunning
                      ? "Running Terminal-Bench..."
                      : `Run Terminal-Bench (${terminalBenchMode})`}
                  </Button>
                </Box>
              )}

              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 2, textAlign: "center" }}>
                Powered by lm-eval and Harbor Terminal-Bench ({terminalBenchDatasets.length} dataset profile)
              </Typography>
            </SectionCard>
          </Grid>

          <Grid item xs={12} md={7}>
            <SectionCard
              fullHeight
              title="Benchmark Jobs"
              subtitle={`${lmEvalJobs.length + terminalBenchJobs.length} jobs`}
              icon={<EvalIcon />}
              accentColor={accentColors.success}
            >
              {lmEvalJobs.length === 0 && terminalBenchJobs.length === 0 ? (
                  <Box sx={{ p: 4, textAlign: "center" }}>
                    <BenchmarkIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                    <Typography color="text.secondary">No benchmark jobs yet</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Select tasks and run to get standardized evaluation scores
                    </Typography>
                  </Box>
                ) : (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                    {terminalBenchJobs.length > 0 && (
                      <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", fontWeight: 700 }}>
                        Terminal-Bench
                      </Typography>
                    )}
                    {terminalBenchJobs.map((job) => (
                      <Box
                        key={job.id}
                        sx={{
                          p: 2,
                          borderRadius: 2,
                          bgcolor: "rgba(0, 0, 0, 0.2)",
                          border: "1px solid rgba(255, 255, 255, 0.06)",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <Box sx={{ flex: 1 }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                              {job.name}
                            </Typography>
                            <StatusChip status={job.status} />
                          </Box>
                          <Typography variant="caption" color="text.secondary">
                            {job.dataset} • mode: {job.mode} • model: {job.model_name} • {new Date(job.created_at).toLocaleString()}
                          </Typography>

                          {job.status === "running" && (
                            <LinearProgress
                              variant="determinate"
                              value={job.progress}
                              sx={{
                                mt: 1,
                                height: 4,
                                borderRadius: 2,
                                bgcolor: "rgba(255, 255, 255, 0.1)",
                                "& .MuiLinearProgress-bar": {
                                  background: `linear-gradient(90deg, ${accentColors.warning}, ${accentColors.rose})`,
                                },
                              }}
                            />
                          )}

                          {job.status === "completed" && (
                            <Box sx={{ display: "flex", gap: 1, mt: 1, flexWrap: "wrap" }}>
                              <Chip
                                size="small"
                                label={`Pass: ${job.results?.pass_rate?.toFixed?.(2) ?? 0}%`}
                                sx={{ bgcolor: "rgba(16, 185, 129, 0.1)", color: accentColors.success, fontSize: "0.7rem" }}
                              />
                              <Chip
                                size="small"
                                label={`Passed ${job.results?.passed_trials ?? 0}/${job.results?.total_trials ?? 0}`}
                                sx={{ fontSize: "0.7rem" }}
                              />
                            </Box>
                          )}

                          {job.error && (
                            <Alert severity="error" sx={{ mt: 1, py: 0 }}>
                              {job.error.slice(0, 180)}
                            </Alert>
                          )}
                        </Box>
                      </Box>
                    ))}

                    {terminalBenchJobs.length > 0 && lmEvalJobs.length > 0 && <Divider sx={{ my: 1 }} />}
                    {lmEvalJobs.length > 0 && (
                      <Typography variant="caption" color="text.secondary" sx={{ textTransform: "uppercase", fontWeight: 700 }}>
                        LM-Eval
                      </Typography>
                    )}
                    {lmEvalJobs.map((job) => (
                      <Box
                        key={job.id}
                        sx={{
                          p: 2,
                          borderRadius: 2,
                          bgcolor: "rgba(0, 0, 0, 0.2)",
                          border: "1px solid rgba(255, 255, 255, 0.06)",
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <Box sx={{ flex: 1 }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                              {job.model_path || job.results?.model_name || job.tasks.join(", ")}
                            </Typography>
                            <StatusChip status={job.status} />
                          </Box>
                          <Typography variant="caption" color="text.secondary">
                            Tasks: {job.tasks.join(", ")} • {new Date(job.created_at).toLocaleString()}
                          </Typography>

                          {job.status === "running" && (
                            <LinearProgress
                              variant="determinate"
                              value={job.progress}
                              sx={{
                                mt: 1,
                                height: 4,
                                borderRadius: 2,
                                bgcolor: "rgba(255, 255, 255, 0.1)",
                                "& .MuiLinearProgress-bar": {
                                  background: `linear-gradient(90deg, ${accentColors.info}, ${accentColors.success})`,
                                },
                              }}
                            />
                          )}

                          {job.status === "completed" && job.results?.tasks && (
                            <Box sx={{ display: "flex", gap: 1, mt: 1, flexWrap: "wrap" }}>
                              {Object.entries(job.results.tasks).slice(0, 4).map(([taskName, taskResult]: [string, any]) => (
                                <Chip
                                  key={taskName}
                                  size="small"
                                  label={`${taskName}: ${taskResult.score?.toFixed(1) || 0}%`}
                                  sx={{
                                    bgcolor: "rgba(16, 185, 129, 0.1)",
                                    color: accentColors.success,
                                    fontSize: "0.7rem",
                                  }}
                                />
                              ))}
                              {Object.keys(job.results.tasks).length > 4 && (
                                <Chip
                                  size="small"
                                  label={`+${Object.keys(job.results.tasks).length - 4} more`}
                                  sx={{ fontSize: "0.7rem" }}
                                />
                              )}
                            </Box>
                          )}

                          {job.error && (
                            <Alert severity="error" sx={{ mt: 1, py: 0 }}>
                              {job.error.slice(0, 100)}
                            </Alert>
                          )}
                        </Box>

                        {job.status === "completed" && (
                          <Button
                            size="small"
                            onClick={() => navigate(benchmarkComparisonPath, { state: { jobId: job.id } })}
                            sx={{ ml: 2, color: accentColors.info }}
                          >
                            View Results
                          </Button>
                        )}
                      </Box>
                    ))}
                  </Box>
                )
              }
            </SectionCard>
          </Grid>
        </Grid>
      )
      }

      {/* A/B Testing Tab */}
      {
        activeTab === 2 && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={5}>
              <SectionCard fullHeight title="Create A/B Test" icon={<ABTestIcon />} accentColor={accentColors.purple} action={evaluationHeaderActions}>
                {evaluationSummaryRow}
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
                fullHeight
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
        )
      }

      {/* LLM Judge Tab */}
      {
        activeTab === 3 && (
          <SectionCard fullHeight title="LLM-as-Judge Evaluation" subtitle="Use AI to automatically evaluate responses" icon={<JudgeIcon />} accentColor={accentColors.warning} action={evaluationHeaderActions}>
            {evaluationSummaryRow}
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
        )
      }
    </PageShell>
  );
};

export default ModelEvaluationPage;
