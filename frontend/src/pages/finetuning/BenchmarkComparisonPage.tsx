/**
 * Benchmark Comparison Dashboard
 * 
 * A professional, feature-rich dashboard for comparing LLM benchmark results
 * across local models and public reference models.
 */

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
    Box,
    Typography,
    Button,
    IconButton,
    Tooltip,
    Chip,
    Switch,
    Paper,
    CircularProgress,
    Alert,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    TextField,
    alpha,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Autocomplete,
    useMediaQuery,
} from "@mui/material";
import {
    ArrowBack as BackIcon,
    Refresh as RefreshIcon,
    Download as DownloadIcon,
    CheckCircle as CheckIcon,
    BarChart as BarChartIcon,
    DonutLarge as RadarIcon,
    TableChart as TableIcon,
    Public as PublicIcon,
    Storage as ModelIcon,
    TrendingUp as TrendIcon,
    FilterList as FilterIcon,
    Delete as DeleteIcon,
} from "@mui/icons-material";
import { useLocation, useNavigate } from "react-router-dom";
import type { BenchmarkComparisonRouteState } from "@/types/trainingHandoff";
import { useToast } from "@/contexts/ToastContext";
import { getRoutePath } from "@/config/navigation";
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from "@/components/common";
import {
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar,
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    Legend,
} from "recharts";
import { accentColors as colors } from "@/utils/theme";

const BENCHMARK_COMPARISON_STATUS_CONFIG: StatusChipConfigMap = {
    bar: { color: colors.warning, label: "Bar" },
    comparison: { color: colors.rose, label: "Benchmark comparison" },
    exportable: { color: colors.success, label: "Export ready" },
    local: { color: colors.success, label: "Local runs" },
    public: { color: colors.purple, label: "Public refs" },
    radar: { color: colors.primary, label: "Radar" },
    summary: { color: colors.success, label: "Summary" },
    table: { color: colors.info, label: "Table" },
};

const benchmarkCompareIconButtonSx = {
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

// Chart colors for models
const modelColors = [
    colors.success,
    colors.primary,
    colors.warning,
    colors.info,
    colors.rose,
    colors.purple,
    "#ec4899", // Pink
    "#14b8a6", // Teal
];

// Public model colors (using vibrant palette but distinct order)
const publicModelColors = [
    colors.purple,
    colors.rose,
    colors.info,
    colors.warning,
    colors.success,
    colors.primary,
    "#ec4899", // Pink
    "#14b8a6", // Teal
];

// Types
interface BenchmarkResult {
    id: string;
    model_name: string;
    model_path?: string;
    created_at: string;
    tasks: Record<string, {
        score: number;
        metrics: Record<string, number>;
    }>;
    isPublic?: boolean;
}

interface PublicModel {
    model_name: string;
    model_size?: string;
    scores: Record<string, number>;
    source: string;
    organization?: string;
}

type ViewMode = "radar" | "bar" | "table";

interface PresetConfig {
    id: string;
    label: string;
    description: string;
}

const PRESETS: PresetConfig[] = [
    { id: "core", label: "Core Summary", description: "High-level view with aggregated scores" },
    { id: "mmlu", label: "MMLU Detailed", description: "Detailed breakdown of all MMLU subjects" },
    { id: "arc", label: "ARC Detailed", description: "ARC Easy vs Challenge details" },
    { id: "all", label: "All Benchmarks", description: "Everything available" },
    { id: "custom", label: "Custom Selection", description: "Manually selected benchmarks" },
];

const MAX_INLINE_TABLE_ROWS = 8;
const MAX_INLINE_LOCAL_RESULTS = 5;
const MAX_INLINE_METRIC_CHIPS = 10;
const MAX_INLINE_PUBLIC_CHIPS = 4;

const AGGREGATION_GROUPS = [
    { prefix: "mmlu_", name: "MMLU (Avg)", color: colors.primary, publicMatch: "mmlu" },
    { prefix: "arc_", name: "ARC (Avg)", color: colors.warning, publicMatch: "arc_challenge" },
    { prefix: "truthfulqa_", name: "TruthfulQA (Avg)", color: colors.success, publicMatch: "truthfulqa_mc2" },
    // New metrics mapping
    { prefix: "gpqa_", name: "GPQA", color: colors.purple, publicMatch: "gpqa" },
    { prefix: "math_", name: "MATH", color: colors.rose, publicMatch: "math" },
    { prefix: "humaneval_", name: "HumanEval", color: colors.info, publicMatch: "humaneval" },
];


// Custom Tooltip for Recharts
const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <Paper sx={{ p: 1.5, border: "1px solid rgba(255,255,255,0.1)", bgcolor: "rgba(0,0,0,0.9)" }}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>{label}</Typography>
                {payload.map((p: any) => (
                    <Box key={p.name} sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
                        <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: p.color }} />
                        <Typography variant="caption" sx={{ color: "text.secondary", width: 140 }}>
                            {p.name}:
                        </Typography>
                        <Typography variant="caption" fontWeight={600}>
                            {Number(p.value).toFixed(1)}
                        </Typography>
                    </Box>
                ))}
            </Paper>
        );
    }
    return null;
};

const BenchmarkComparisonPage = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const toast = useToast();
    const isNarrow = useMediaQuery("(max-width:599.95px)");
    const evaluationPath = getRoutePath("finetuning-evaluation");
    const routeState = location.state as BenchmarkComparisonRouteState | null;
    const requestedJobIds = useMemo(() => {
        const fromState = routeState?.jobIds ?? (routeState?.jobId ? [routeState.jobId] : []);
        const fromQuery = new URLSearchParams(location.search).get("job");
        return Array.from(new Set([...fromState, ...(fromQuery ? [fromQuery] : [])].filter(Boolean)));
    }, [location.search, routeState?.jobId, routeState?.jobIds]);

    // State
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [localResults, setLocalResults] = useState<BenchmarkResult[]>([]);
    const [publicModels, setPublicModels] = useState<PublicModel[]>([]);
    const [deleteTarget, setDeleteTarget] = useState<BenchmarkResult | null>(null);
    const [deletingId, setDeletingId] = useState<string | null>(null);

    // View Config
    const [activePreset, setActivePreset] = useState("core");
    const [useAggregates, setUseAggregates] = useState(true);
    const [viewMode, setViewMode] = useState<ViewMode>("radar");
    const [datasetSearchQuery, setDatasetSearchQuery] = useState("");

    // Selection
    const [selectedLocalIds, setSelectedLocalIds] = useState<Set<string>>(new Set());
    const [selectedPublicNames, setSelectedPublicNames] = useState<Set<string>>(new Set());
    const [showPublicModels, setShowPublicModels] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");

    const loadBenchmarkData = useCallback(async (
        options: { preserveSelection?: boolean; showInitialLoader?: boolean } = {}
    ) => {
        const { preserveSelection = false, showInitialLoader = true } = options;

        try {
            setError(null);
            if (showInitialLoader) {
                setLoading(true);
            } else {
                setRefreshing(true);
            }

            // 1. Fetch Local Jobs
            const jobsRes = await fetch("/api/v1/finetune/lm-eval/jobs");
            if (!jobsRes.ok) {
                throw new Error(`Failed to load local benchmark jobs (${jobsRes.status})`);
            }

            const jobsData = await jobsRes.json();
            // API returns {jobs: [...]} - extract the array
            const rawJobs = jobsData?.jobs || jobsData;
            const jobsArray = Array.isArray(rawJobs) ? rawJobs : [];
            const completedJobs = jobsArray.filter((j: any) => j.status === "completed" && j.results);

            const parsedResults: BenchmarkResult[] = completedJobs.map((j: any) => {
                const modelPath = String(j.model_path || j.model_name || j.id || "Unknown Model");

                return {
                    id: j.id,
                    model_name: modelPath.split("/").pop() || "Unknown Model",
                    model_path: modelPath,
                    created_at: j.created_at,
                    // API returns {tasks: {...}, model_name, date} - extract just the tasks map
                    tasks: j.results?.tasks || j.results || {},
                    isPublic: false
                };
            });
            const sortedResults = [...parsedResults].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
            setLocalResults(sortedResults);

            const availableLocalIds = new Set(sortedResults.map((result) => result.id));
            setSelectedLocalIds((prev) => {
                if (preserveSelection) {
                    const retainedIds = Array.from(prev).filter((id) => availableLocalIds.has(id));
                    if (retainedIds.length > 0) {
                        return new Set(retainedIds);
                    }
                }

                const matchedRequestedIds = requestedJobIds.filter((id) => availableLocalIds.has(id));
                if (matchedRequestedIds.length > 0) {
                    return new Set(matchedRequestedIds);
                }

                return sortedResults.length > 0 ? new Set([sortedResults[0].id]) : new Set();
            });

            // 2. Fetch Public Benchmarks
            const publicRes = await fetch("/api/v1/finetune/lm-eval/public-benchmarks?limit=50");
            if (!publicRes.ok) {
                throw new Error(`Failed to load public benchmarks (${publicRes.status})`);
            }

            const publicData = await publicRes.json();
            // API returns {benchmarks: [...]} - extract the array
            const rawModels = publicData?.benchmarks || publicData;
            const modelsArray = Array.isArray(rawModels) ? rawModels : [];
            setPublicModels(modelsArray);

            const availablePublicNames = new Set(modelsArray.map((model: any) => model.model_name).filter(Boolean));
            setSelectedPublicNames((prev) => {
                if (preserveSelection) {
                    const retainedNames = Array.from(prev).filter((name) => availablePublicNames.has(name));
                    if (retainedNames.length > 0) {
                        return new Set(retainedNames);
                    }
                }

                return new Set(modelsArray.slice(0, 3).map((model: any) => model.model_name).filter(Boolean));
            });
        } catch (err) {
            console.error("Failed to fetch benchmark data:", err);
            setError(err instanceof Error ? err.message : "Failed to load benchmark data");
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    }, [requestedJobIds]);

    // Fetch Data
    useEffect(() => {
        void loadBenchmarkData({ preserveSelection: false, showInitialLoader: true });
    }, [loadBenchmarkData]);

    // Derived Lists
    const aggregatedTasks = useMemo(() => AGGREGATION_GROUPS.map(g => g.name), []);

    const allRawTasks = useMemo(() => {
        const tasks = new Set<string>();
        localResults.forEach(r => {
            if (r.tasks) Object.keys(r.tasks).forEach(t => tasks.add(t));
        });
        publicModels.forEach(m => {
            if (m.scores) Object.keys(m.scores).forEach(t => tasks.add(t));
        });
        return Array.from(tasks).sort();
    }, [localResults, publicModels]);

    // Determine tasks to show based on preset
    const displayedTasks = useMemo(() => {
        if (activePreset === "all") return useAggregates ? aggregatedTasks : allRawTasks;

        if (activePreset === "core") {
            // Core shows generic benchmarks + aggregated averages
            const core = ["hellaswag", "gsm8k", "winogrande", "piqa"];
            if (useAggregates) {
                // Add the averages
                AGGREGATION_GROUPS.forEach(g => core.push(g.name));
            }
            // Filter to include only what's available
            return core.filter(t => aggregatedTasks.includes(t) || allRawTasks.includes(t));
        }

        if (activePreset === "mmlu") {
            // Show all MMLU tasks
            return allRawTasks.filter(t => t.startsWith("mmlu_"));
        }

        if (activePreset === "arc") {
            // Show all ARC tasks
            return allRawTasks.filter(t => t.startsWith("arc_"));
        }

        // Custom or fallback
        return useAggregates ? aggregatedTasks : allRawTasks;
    }, [activePreset, useAggregates, aggregatedTasks, allRawTasks]);

    // Calculate chart data points
    const chartData = useMemo(() => {
        return displayedTasks.map(task => {
            const point: Record<string, string | number> = { task };

            // Helper to get score for a model (handling aggregation)
            const getScore = (tasksMap: Record<string, any>) => {
                // Check direct match first
                if (tasksMap[task]?.score !== undefined) return tasksMap[task].score;
                if (tasksMap[task] !== undefined && typeof tasksMap[task] === 'number') return tasksMap[task];

                // Check aggregation
                if (useAggregates) {
                    const group = AGGREGATION_GROUPS.find(g => g.name === task);
                    if (group) {
                        // 1. Try public match key (e.g. "mmlu" for "MMLU (Avg)")
                        if (group.publicMatch && tasksMap[group.publicMatch] !== undefined) {
                            return typeof tasksMap[group.publicMatch] === 'object'
                                ? tasksMap[group.publicMatch].score
                                : tasksMap[group.publicMatch];
                        }

                        // 2. Calculate average of all tasks starting with prefix (local results)
                        const subtasks = Object.keys(tasksMap).filter(t => t.startsWith(group.prefix));
                        if (subtasks.length === 0) return 0;

                        let sum = 0;
                        let count = 0;
                        subtasks.forEach(t => {
                            const val = typeof tasksMap[t] === 'object' ? tasksMap[t].score : tasksMap[t];
                            if (val !== undefined) {
                                sum += val;
                                count++;
                            }
                        });
                        return count > 0 ? sum / count : 0;
                    }
                }
                return 0;
            };

            // Add local models
            localResults.forEach((result) => {
                if (selectedLocalIds.has(result.id)) {
                    point[result.model_name] = getScore(result.tasks);
                }
            });

            // Add public models
            if (showPublicModels) {
                publicModels.forEach(model => {
                    if (selectedPublicNames.has(model.model_name)) {
                        point[model.model_name] = getScore(model.scores);
                    }
                });
            }

            return point;
        });
    }, [displayedTasks, localResults, selectedLocalIds, publicModels, selectedPublicNames, showPublicModels, useAggregates]);

    // Data keys for charts
    const dataKeys = useMemo(() => {
        const keys: { name: string; isPublic: boolean }[] = [];
        localResults.forEach(r => {
            if (selectedLocalIds.has(r.id)) {
                keys.push({ name: r.model_name, isPublic: false });
            }
        });
        if (showPublicModels) {
            publicModels.forEach(m => {
                if (selectedPublicNames.has(m.model_name)) {
                    keys.push({ name: m.model_name, isPublic: true });
                }
            });
        }
        return keys;
    }, [localResults, selectedLocalIds, publicModels, selectedPublicNames, showPublicModels]);

    const modelSummaries = useMemo(() => {
        return dataKeys.map((dk, idx) => {
            const scores = displayedTasks
                .map((task) => {
                    const dataPoint = chartData.find((point) => point.task === task);
                    return Number(dataPoint?.[dk.name] || 0);
                })
                .filter((score) => Number.isFinite(score) && score > 0);
            const average = scores.length > 0 ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;
            return {
                ...dk,
                average,
                metricCount: scores.length,
                color: dk.isPublic ? publicModelColors[idx % publicModelColors.length] : modelColors[idx % modelColors.length],
            };
        }).sort((a, b) => b.average - a.average);
    }, [chartData, dataKeys, displayedTasks]);

    // Export to CSV
    const exportCSV = () => {
        const headers = ["Model", ...displayedTasks, "Average"];
        const rows = dataKeys.map(dk => {
            const scores = displayedTasks.map(task => {
                const dataPoint = chartData.find(d => d.task === task);
                return Number(dataPoint?.[dk.name] || 0);
            });
            const avg = scores.reduce((a, b) => Number(a) + Number(b), 0) / scores.length;
            return [dk.name, ...scores.map(s => Number(s).toFixed(1)), avg.toFixed(1)];
        });

        const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "benchmark_comparison.csv";
        a.click();
    };

    // Toggles
    const toggleLocal = (id: string) => {
        setSelectedLocalIds(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const togglePublic = (name: string) => {
        setSelectedPublicNames(prev => {
            const next = new Set(prev);
            if (next.has(name)) next.delete(name);
            else next.add(name);
            return next;
        });
    };

    const handleRequestDeleteLocal = (result: BenchmarkResult, e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent row selection
        setDeleteTarget(result);
    };

    const handleConfirmDeleteLocal = async () => {
        if (!deleteTarget) return;

        try {
            setDeletingId(deleteTarget.id);
            const res = await fetch(`/api/v1/finetune/lm-eval/jobs/${deleteTarget.id}`, { method: 'DELETE' });
            if (res.ok) {
                // Remove from local state
                setLocalResults(prev => prev.filter(r => r.id !== deleteTarget.id));
                setSelectedLocalIds(prev => {
                    const next = new Set(prev);
                    next.delete(deleteTarget.id);
                    return next;
                });
                toast.success(`${deleteTarget.model_name} benchmark result was removed.`, "Benchmark deleted");
                setDeleteTarget(null);
            } else {
                const message = await res.text();
                console.error('Failed to delete benchmark:', message);
                setError(message || "Failed to delete benchmark result");
                toast.error(message || "Failed to delete benchmark result", "Delete failed");
            }
        } catch (err) {
            console.error('Delete error:', err);
            const message = err instanceof Error ? err.message : "Failed to delete benchmark result";
            setError(message);
            toast.error(message, "Delete failed");
        } finally {
            setDeletingId(null);
        }
    };

    // Filter models
    const filteredLocal = localResults.filter(r =>
        r.model_name.toLowerCase().includes(searchQuery.toLowerCase())
    );
    const selectedPublicModels = useMemo(
        () => publicModels.filter((model) => selectedPublicNames.has(model.model_name)),
        [publicModels, selectedPublicNames]
    );

    const publicModelOptions = useMemo(() => {
        const orgPriority: Record<string, number> = {
            'OpenAI': 1,
            'Anthropic': 2,
            'Google': 3,
            'Meta': 4,
            'Mistral AI': 5,
            'Alibaba Cloud / Qwen Team': 6,
            'Microsoft': 7,
        };

        return [...publicModels].sort((a, b) => {
            const aOrg = a.organization || 'Other';
            const bOrg = b.organization || 'Other';
            const aPriority = orgPriority[aOrg] || 100;
            const bPriority = orgPriority[bOrg] || 100;
            if (aPriority !== bPriority) return aPriority - bPriority;
            if (aOrg !== bOrg) return aOrg.localeCompare(bOrg);
            return Object.keys(b.scores || {}).length - Object.keys(a.scores || {}).length;
        });
    }, [publicModels]);

    const selectedPublicCount = showPublicModels ? selectedPublicNames.size : 0;
    const selectedModelCount = selectedLocalIds.size + selectedPublicCount;
    const filteredDisplayedTasks = displayedTasks.filter(t => t.toLowerCase().includes(datasetSearchQuery.toLowerCase()));
    const visibleLocalResults = filteredLocal.slice(0, MAX_INLINE_LOCAL_RESULTS);
    const remainingLocalResults = Math.max(0, filteredLocal.length - visibleLocalResults.length);
    const showCompactScoreSummary = isNarrow && viewMode !== "table";
    const scoreModeStatus = showCompactScoreSummary ? "summary" : viewMode;

    if (loading) {
        return (
            <PageShell>
              <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "60vh" }}>
                <CircularProgress />
              </Box>
            </PageShell>
        );
    }

    return (
        <PageShell sx={{ maxWidth: 1800, mx: "auto", py: 1.25 }}>
            {error && <Alert severity="error" sx={{ mb: 1.25 }}>{error}</Alert>}

            <Box
                sx={{
                    display: "grid",
                    gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1fr) 320px", xl: "minmax(0, 1fr) 340px" },
                    gap: 1.25,
                    mb: 1.25,
                    alignItems: "start",
                }}
            >
                <SectionCard
                    title="Scores"
                    subtitle={`${activePreset}${useAggregates ? " / aggregated" : ""}`}
                    icon={showCompactScoreSummary ? <TrendIcon /> : viewMode === "radar" ? <RadarIcon /> : viewMode === "bar" ? <BarChartIcon /> : <TableIcon />}
                    accentColor={colors.primary}
                    action={
                        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap", justifyContent: "flex-end" }}>
                            <StatusChip status={scoreModeStatus} statusConfig={BENCHMARK_COMPARISON_STATUS_CONFIG} height={22} />
                            <Box sx={{ display: { xs: "none", sm: "flex" }, bgcolor: "action.hover", borderRadius: "8px", border: "1px solid", borderColor: "divider", p: 0.25 }}>
                                {[
                                    { mode: "radar" as ViewMode, icon: <RadarIcon />, label: "Radar" },
                                    { mode: "bar" as ViewMode, icon: <BarChartIcon />, label: "Bar" },
                                    { mode: "table" as ViewMode, icon: <TableIcon />, label: "Table" },
                                ].map(({ mode, icon, label }) => (
                                    <Tooltip key={mode} title={label}>
                                        <IconButton
                                            aria-label={label}
                                            size="small"
                                            onClick={() => setViewMode(mode)}
                                            sx={{
                                                width: 30,
                                                height: 28,
                                                borderRadius: "6px",
                                                color: viewMode === mode ? colors.primary : "text.secondary",
                                                bgcolor: viewMode === mode ? alpha(colors.primary, 0.1) : "transparent",
                                            }}
                                        >
                                            {React.isValidElement(icon)
                                                ? React.cloneElement(icon as React.ReactElement<{ sx?: any }>, { sx: { fontSize: 17 } })
                                                : icon}
                                        </IconButton>
                                    </Tooltip>
                                ))}
                            </Box>
                            <Tooltip title={refreshing ? "Refreshing benchmarks" : "Refresh benchmarks"}>
                                <span>
                                    <IconButton
                                        aria-label={refreshing ? "Refreshing benchmarks" : "Refresh benchmarks"}
                                        size="small"
                                        onClick={() => void loadBenchmarkData({ preserveSelection: true, showInitialLoader: false })}
                                        disabled={refreshing}
                                        sx={benchmarkCompareIconButtonSx}
                                    >
                                        {refreshing ? <CircularProgress size={15} /> : <RefreshIcon sx={{ fontSize: 18 }} />}
                                    </IconButton>
                                </span>
                            </Tooltip>
                            <Tooltip title="Export CSV">
                                <span>
                                    <IconButton
                                        aria-label="Export CSV"
                                        size="small"
                                        onClick={exportCSV}
                                        disabled={dataKeys.length === 0}
                                        sx={{
                                            ...benchmarkCompareIconButtonSx,
                                            bgcolor: colors.primary,
                                            color: "#fff",
                                            borderColor: colors.primary,
                                            "&:hover": {
                                                bgcolor: colors.primary,
                                            },
                                        }}
                                    >
                                        <DownloadIcon sx={{ fontSize: 18 }} />
                                    </IconButton>
                                </span>
                            </Tooltip>
                            <Tooltip title="Back to evaluation">
                                <IconButton
                                    aria-label="Back to evaluation"
                                    onClick={() => navigate(evaluationPath)}
                                    sx={benchmarkCompareIconButtonSx}
                                >
                                    <BackIcon sx={{ fontSize: 18 }} />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    }
                    sx={{
                        "& .MuiCardContent-root > .MuiBox-root": {
                            overflow: "hidden",
                        },
                    }}
                >
                    {dataKeys.length === 0 ? (
                        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: 180, opacity: 0.5 }}>
                            <TrendIcon sx={{ fontSize: 40, mb: 1 }} />
                            <Typography>Select models below to begin comparison</Typography>
                        </Box>
                    ) : (
                        <Box
                            sx={{
                                display: showCompactScoreSummary || viewMode === "table" ? "block" : "grid",
                                gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1fr) 220px" },
                                gap: 1.25,
                                height: showCompactScoreSummary ? "auto" : { xs: 220, md: 232 },
                                minHeight: 0,
                                overflow: "hidden",
                            }}
                        >
                            {showCompactScoreSummary ? (
                                <Box data-benchmark-mobile-summary sx={{ display: "grid", gap: 0.75 }}>
                                    {modelSummaries.slice(0, 5).map((model) => (
                                        <Box
                                            key={`${model.name}-${model.isPublic ? "public" : "local"}-mobile`}
                                            sx={{
                                                p: 0.85,
                                                borderRadius: "8px",
                                                border: "1px solid",
                                                borderColor: "divider",
                                                bgcolor: alpha(model.color, 0.045),
                                                minWidth: 0,
                                            }}
                                        >
                                            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, minWidth: 0 }}>
                                                <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: model.color, flexShrink: 0 }} />
                                                <Typography variant="caption" sx={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: 700 }}>
                                                    {model.name}
                                                </Typography>
                                                <Typography variant="caption" sx={{ fontWeight: 800, color: "text.primary" }}>
                                                    {model.average.toFixed(1)}
                                                </Typography>
                                            </Box>
                                            <Box sx={{ height: 5, mt: 0.55, borderRadius: 999, bgcolor: alpha(model.color, 0.16), overflow: "hidden" }}>
                                                <Box sx={{ width: `${Math.min(100, model.average)}%`, height: "100%", bgcolor: model.color }} />
                                            </Box>
                                            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.35, fontSize: "0.68rem" }}>
                                                {model.metricCount} metric{model.metricCount === 1 ? "" : "s"}
                                            </Typography>
                                        </Box>
                                    ))}
                                </Box>
                            ) : (
                            <Box sx={{ height: "100%", minWidth: 0, overflow: "hidden" }}>
                                {viewMode === "radar" && (
                                    <ResponsiveContainer width="100%" height="100%">
                                    <RadarChart data={chartData} margin={{ top: 4, right: 18, bottom: 2, left: 18 }}>
                                        <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                        <PolarAngleAxis
                                            dataKey="task"
                                            tick={{ fill: "#94a3b8", fontSize: 11 }}
                                        />
                                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 10 }} />
                                        {dataKeys.map((dk, idx) => (
                                            <Radar
                                                key={dk.name}
                                                name={dk.name}
                                                dataKey={dk.name}
                                                stroke={dk.isPublic ? publicModelColors[idx % publicModelColors.length] : modelColors[idx % modelColors.length]}
                                                fill={dk.isPublic ? publicModelColors[idx % publicModelColors.length] : modelColors[idx % modelColors.length]}
                                                fillOpacity={dk.isPublic ? 0.1 : 0.2}
                                                strokeWidth={dk.isPublic ? 1 : 2}
                                                strokeDasharray={dk.isPublic ? "5 5" : undefined}
                                            />
                                        ))}
                                        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 2 }} />
                                        <RechartsTooltip content={<CustomTooltip />} />
                                    </RadarChart>
                                    </ResponsiveContainer>
                                )}
                                {viewMode === "bar" && (
                                    <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData} margin={{ top: 4, right: 18, bottom: 2, left: 18 }}>
                                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                                        <XAxis dataKey="task" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                                        <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                                        <RechartsTooltip content={<CustomTooltip />} />
                                        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 2 }} />
                                        {dataKeys.map((dk, idx) => (
                                            <Bar
                                                key={dk.name}
                                                dataKey={dk.name}
                                                fill={dk.isPublic ? publicModelColors[idx % publicModelColors.length] : modelColors[idx % modelColors.length]}
                                                fillOpacity={dk.isPublic ? 0.6 : 0.9}
                                                radius={[4, 4, 0, 0]}
                                            />
                                        ))}
                                    </BarChart>
                                    </ResponsiveContainer>
                                )}
                                {viewMode === "table" && (
                                    <Box sx={{ height: "100%", overflow: "hidden" }}>
                                        <table style={{ width: "100%", borderCollapse: "collapse" }}>
                                            <thead>
                                                <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                                                    <th style={{ padding: 8, textAlign: "left" }}>Task</th>
                                                    {dataKeys.map(dk => <th key={dk.name} style={{ padding: 8, textAlign: "right" }}>{dk.name}</th>)}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {chartData.slice(0, MAX_INLINE_TABLE_ROWS).map((d: any) => (
                                                    <tr key={d.task} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                                                        <td style={{ padding: 8 }}>{d.task}</td>
                                                        {dataKeys.map(dk => (
                                                            <td key={dk.name} style={{ padding: 8, textAlign: "right", color: alpha("#fff", 0.7) }}>
                                                                {Number(d[dk.name]).toFixed(1)}%
                                                            </td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                        {chartData.length > MAX_INLINE_TABLE_ROWS && (
                                            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                                                Showing {MAX_INLINE_TABLE_ROWS} of {chartData.length} metrics. Use filters or export for the full table.
                                            </Typography>
                                        )}
                                    </Box>
                                )}
                            </Box>
                            )}
                            {!showCompactScoreSummary && viewMode !== "table" && (
                                <Box
                                    sx={{
                                        display: { xs: "none", lg: "flex" },
                                        flexDirection: "column",
                                        minWidth: 0,
                                        borderLeft: "1px solid",
                                        borderColor: "divider",
                                        pl: 1.25,
                                        overflow: "hidden",
                                    }}
                                >
                                    <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 700, mb: 0.5 }}>
                                        Model averages
                                    </Typography>
                                    <Box sx={{ display: "grid", gap: 0.55, minHeight: 0, overflow: "hidden" }}>
                                        {modelSummaries.slice(0, 5).map((model) => (
                                            <Box key={`${model.name}-${model.isPublic ? "public" : "local"}`} sx={{ minWidth: 0 }}>
                                                <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, minWidth: 0 }}>
                                                    <Box sx={{ width: 7, height: 7, borderRadius: "50%", bgcolor: model.color, flexShrink: 0 }} />
                                                    <Typography variant="caption" sx={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                                        {model.name}
                                                    </Typography>
                                                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                                                        {model.average.toFixed(1)}
                                                    </Typography>
                                                </Box>
                                                <Box sx={{ height: 4, mt: 0.35, borderRadius: 999, bgcolor: alpha(model.color, 0.14), overflow: "hidden" }}>
                                                    <Box sx={{ width: `${Math.min(100, model.average)}%`, height: "100%", bgcolor: model.color }} />
                                                </Box>
                                            </Box>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                        </Box>
                    )}
                </SectionCard>

                <SectionCard
                    title="View"
                    subtitle={activePreset}
                    icon={<FilterIcon />}
                    accentColor={colors.info}
                    action={<StatusChip status="table" label={`${displayedTasks.length} metrics`} statusConfig={BENCHMARK_COMPARISON_STATUS_CONFIG} />}
                    collapsible={isNarrow}
                    defaultExpanded={!isNarrow}
                >
                    <Box sx={{ display: "grid", gap: 1, minWidth: 0 }}>
                        <FormControl size="small" fullWidth>
                            <InputLabel>Preset View</InputLabel>
                            <Select
                                value={activePreset}
                                label="Preset View"
                                onChange={(e) => setActivePreset(e.target.value)}
                            >
                                {PRESETS.map(p => (
                                    <MenuItem key={p.id} value={p.id}>
                                        <Box>
                                            <Typography variant="body2">{p.label}</Typography>
                                            <Typography variant="caption" color="text.secondary">{p.description}</Typography>
                                        </Box>
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", p: 0.7, bgcolor: "action.hover", borderRadius: "8px", border: "1px solid", borderColor: "divider" }}>
                            <Typography variant="body2" fontWeight={600}>Smart Aggregation</Typography>
                            <Switch
                                size="small"
                                checked={useAggregates}
                                onChange={(e) => setUseAggregates(e.target.checked)}
                            />
                        </Box>

                        <TextField
                            placeholder="Filter metrics..."
                            size="small"
                            fullWidth
                            value={datasetSearchQuery}
                            onChange={(e) => setDatasetSearchQuery(e.target.value)}
                        />
                        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.45 }}>
                            {filteredDisplayedTasks.slice(0, MAX_INLINE_METRIC_CHIPS).map(task => (
                                <Chip
                                    key={task}
                                    label={task}
                                    size="small"
                                    sx={{
                                        height: 20,
                                        maxWidth: "100%",
                                        fontSize: "0.7rem",
                                        bgcolor: AGGREGATION_GROUPS.some(g => g.name === task) ? alpha(colors.success, 0.1) : alpha(colors.slate, 0.1),
                                        color: AGGREGATION_GROUPS.some(g => g.name === task) ? colors.success : "text.secondary"
                                    }}
                                />
                            ))}
                            {filteredDisplayedTasks.length > MAX_INLINE_METRIC_CHIPS && (
                                <Chip
                                    label={`+${filteredDisplayedTasks.length - MAX_INLINE_METRIC_CHIPS}`}
                                    size="small"
                                    sx={{ height: 20, fontSize: "0.7rem", bgcolor: "action.hover", color: "text.secondary" }}
                                />
                            )}
                        </Box>
                    </Box>
                </SectionCard>
            </Box>

            <SectionCard
                title="Compare Sources"
                subtitle={isNarrow ? undefined : `${filteredLocal.length} local runs / ${publicModels.length} public references`}
                icon={<ModelIcon />}
                accentColor={colors.success}
                action={<StatusChip status="local" label={`${selectedModelCount} selected`} statusConfig={BENCHMARK_COMPARISON_STATUS_CONFIG} />}
                collapsible={isNarrow}
                defaultExpanded={!isNarrow}
            >
                <Box
                    sx={{
                        display: "grid",
                        gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 0.95fr) minmax(320px, 1.05fr)" },
                        gap: 1.25,
                        alignItems: "start",
                    }}
                >
                    <Box sx={{ minWidth: 0 }}>
                        <Typography variant="caption" sx={{ display: "block", color: "text.secondary", fontWeight: 700, mb: 0.65 }}>
                            Local runs
                        </Typography>
                    <TextField
                        placeholder="Filter local runs..."
                        size="small"
                        fullWidth
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                            sx={{ mb: 0.75, "& .MuiInputBase-root": { fontSize: "0.85rem" } }}
                    />
                    <Box>
                        {visibleLocalResults.map((result, idx) => (
                            <Box
                                key={result.id}
                                onClick={() => toggleLocal(result.id)}
                                sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 1,
                                    px: 0.8,
                                    py: 0.65,
                                    borderRadius: "8px",
                                    cursor: "pointer",
                                    bgcolor: selectedLocalIds.has(result.id) ? alpha(colors.success, 0.1) : "transparent",
                                    border: `1px solid ${selectedLocalIds.has(result.id) ? alpha(colors.success, 0.3) : "transparent"}`,
                                    mb: 0.4,
                                    "&:hover": { bgcolor: alpha(colors.success, 0.05) },
                                }}
                            >
                                <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: modelColors[idx % modelColors.length] }} />
                                <Typography variant="caption" sx={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                    {result.model_name}
                                </Typography>
                                {selectedLocalIds.has(result.id) && <CheckIcon sx={{ fontSize: 14, color: colors.success }} />}
                                <IconButton
                                    size="small"
                                    onClick={(e) => handleRequestDeleteLocal(result, e)}
                                    disabled={deletingId === result.id}
                                    sx={{
                                        p: 0.25,
                                        ml: 0.5,
                                        opacity: 0.5,
                                        '&:hover': { opacity: 1, color: 'error.main' }
                                    }}
                                >
                                    {deletingId === result.id ? <CircularProgress size={14} /> : <DeleteIcon sx={{ fontSize: 14 }} />}
                                </IconButton>
                            </Box>
                        ))}
                        {remainingLocalResults > 0 && (
                            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                                {remainingLocalResults} more local run{remainingLocalResults === 1 ? "" : "s"} hidden by the compact view.
                            </Typography>
                        )}
                    </Box>
                    </Box>

                    <Box sx={{ minWidth: 0 }}>
                        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.65 }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, minWidth: 0 }}>
                                <PublicIcon sx={{ fontSize: 16, color: colors.purple }} />
                                <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 700 }}>
                                    Public references
                                </Typography>
                            </Box>
                            <Switch
                                size="small"
                                checked={showPublicModels}
                                onChange={(e) => setShowPublicModels(e.target.checked)}
                            />
                        </Box>
                        <Box sx={{ opacity: showPublicModels ? 1 : 0.5, pointerEvents: showPublicModels ? "auto" : "none" }}>
                        <Autocomplete
                            multiple
                            size="small"
                            disableCloseOnSelect
                            limitTags={2}
                            options={publicModelOptions}
                            value={selectedPublicModels}
                            groupBy={(option) => option.organization || "Other"}
                            getOptionLabel={(option) => option.model_name}
                            isOptionEqualToValue={(option, value) => option.model_name === value.model_name}
                            onChange={(_, value) => {
                                setSelectedPublicNames(new Set(value.map((model) => model.model_name)));
                            }}
                            renderInput={(params) => (
                                <TextField
                                    {...params}
                                        label="Reference models"
                                    placeholder={selectedPublicModels.length === 0 ? "Search models..." : ""}
                                />
                            )}
                            renderOption={(props, model, state) => {
                                const color = publicModelColors[state.index % publicModelColors.length];
                                return (
                                    <Box component="li" {...props} sx={{ display: "flex", alignItems: "center", gap: 1, py: 0.5 }}>
                                        <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: color, flexShrink: 0 }} />
                                        <Typography variant="body2" sx={{ flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis" }}>
                                            {model.model_name}
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                            {Object.keys(model.scores || {}).length}
                                        </Typography>
                                    </Box>
                                );
                            }}
                            sx={{
                                "& .MuiInputBase-root": {
                                    minHeight: 38,
                                    alignItems: "center",
                                },
                                "& .MuiChip-root": {
                                    height: 22,
                                    borderRadius: "8px",
                                    fontSize: "0.7rem",
                                },
                            }}
                        />
                            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.75 }}>
                                {selectedPublicModels.slice(0, MAX_INLINE_PUBLIC_CHIPS).map((model, idx) => (
                                <Chip
                                    key={model.model_name}
                                    size="small"
                                    label={model.model_name}
                                    onDelete={() => togglePublic(model.model_name)}
                                    sx={{
                                        height: 22,
                                        fontSize: "0.7rem",
                                        bgcolor: alpha(publicModelColors[idx % publicModelColors.length], 0.12),
                                        color: publicModelColors[idx % publicModelColors.length],
                                    }}
                                />
                            ))}
                                {selectedPublicModels.length > MAX_INLINE_PUBLIC_CHIPS && (
                                    <Chip size="small" label={`+${selectedPublicModels.length - MAX_INLINE_PUBLIC_CHIPS}`} sx={{ height: 22, fontSize: "0.7rem" }} />
                            )}
                            </Box>
                        </Box>
                    </Box>
                </Box>
            </SectionCard>
            <Dialog
                open={Boolean(deleteTarget)}
                onClose={() => deletingId ? undefined : setDeleteTarget(null)}
                maxWidth="xs"
                fullWidth
            >
                <DialogTitle>Delete benchmark result?</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="text.secondary">
                        {deleteTarget ? (
                            <>
                                Delete <strong>{deleteTarget.model_name}</strong> from local benchmark comparisons?
                                This cannot be undone.
                            </>
                        ) : null}
                    </Typography>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDeleteTarget(null)} disabled={Boolean(deletingId)}>
                        Cancel
                    </Button>
                    <Button
                        color="error"
                        variant="contained"
                        onClick={handleConfirmDeleteLocal}
                        disabled={Boolean(deletingId)}
                        startIcon={deletingId ? <CircularProgress size={16} color="inherit" /> : <DeleteIcon />}
                    >
                        {deletingId ? "Deleting" : "Delete"}
                    </Button>
                </DialogActions>
            </Dialog>
        </PageShell>
    );
};

export default BenchmarkComparisonPage;
