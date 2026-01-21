/**
 * Benchmark Comparison Dashboard
 * 
 * A professional, feature-rich dashboard for comparing LLM benchmark results
 * across local models and public reference models.
 */

import React, { useState, useEffect, useMemo } from "react";
import {
    Box,
    Typography,
    Button,
    IconButton,
    Tooltip,
    Chip,
    FormControlLabel,
    Switch,
    Paper,
    Divider,
    CircularProgress,
    Alert,
    TextField,
    InputAdornment,
    alpha,
    Grid,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Collapse,
} from "@mui/material";
import {
    ArrowBack as BackIcon,
    Refresh as RefreshIcon,
    Download as DownloadIcon,
    Search as SearchIcon,
    CheckCircle as CheckIcon,
    RadioButtonUnchecked as UncheckedIcon,
    BarChart as BarChartIcon,
    DonutLarge as RadarIcon,
    TableChart as TableIcon,
    Public as PublicIcon,
    Storage as ModelIcon,
    TrendingUp as TrendIcon,
    FilterList as FilterIcon,
    ExpandMore as ExpandIcon,
    ExpandLess as CollapseIcon,
    Summarize as SummaryIcon,
    Delete as DeleteIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
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
    Cell,
} from "recharts";

// Color palette
const colors = {
    primary: "#6366f1",
    success: "#10b981",
    warning: "#f59e0b",
    info: "#06b6d4",
    purple: "#8b5cf6",
    rose: "#f43f5e",
    slate: "#64748b",
};

// Chart colors for models
const modelColors = [
    "#10b981", // Green
    "#6366f1", // Indigo
    "#f59e0b", // Amber
    "#06b6d4", // Cyan
    "#f43f5e", // Rose
    "#8b5cf6", // Purple
    "#ec4899", // Pink
    "#14b8a6", // Teal
];

// Public model colors (using vibrant palette but distinct order)
const publicModelColors = [
    "#8b5cf6", // Purple
    "#f43f5e", // Rose
    "#06b6d4", // Cyan
    "#f59e0b", // Amber
    "#10b981", // Green
    "#6366f1", // Indigo
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

    // State
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [localResults, setLocalResults] = useState<BenchmarkResult[]>([]);
    const [publicModels, setPublicModels] = useState<PublicModel[]>([]);

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

    // Fetch Data
    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);

                // 1. Fetch Local Jobs
                const jobsRes = await fetch("/api/v1/finetune/lm-eval/jobs");
                if (jobsRes.ok) {
                    const jobsData = await jobsRes.json();
                    // API returns {jobs: [...]} - extract the array
                    const rawJobs = jobsData?.jobs || jobsData;
                    const jobsArray = Array.isArray(rawJobs) ? rawJobs : [];
                    const completedJobs = jobsArray.filter((j: any) => j.status === "completed" && j.results);

                    const parsedResults: BenchmarkResult[] = completedJobs.map((j: any) => ({
                        id: j.id,
                        model_name: j.model_path.split("/").pop() || "Unknown Model",
                        model_path: j.model_path,
                        created_at: j.created_at,
                        // API returns {tasks: {...}, model_name, date} - extract just the tasks map
                        tasks: j.results?.tasks || j.results || {},
                        isPublic: false
                    }));
                    setLocalResults(parsedResults);

                    // Default select most recent local job
                    if (parsedResults.length > 0) {
                        // Sort by date desc
                        parsedResults.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
                        setSelectedLocalIds(new Set([parsedResults[0].id]));
                    }
                }

                // 2. Fetch Public Benchmarks
                const publicRes = await fetch("/api/v1/finetune/lm-eval/public-benchmarks?limit=50");
                if (publicRes.ok) {
                    const publicData = await publicRes.json();
                    // API returns {benchmarks: [...]} - extract the array
                    const rawModels = publicData?.benchmarks || publicData;
                    const modelsArray = Array.isArray(rawModels) ? rawModels : [];
                    setPublicModels(modelsArray);

                    // Default select top 3 public models
                    if (modelsArray.length > 0) {
                        const defaults = modelsArray.slice(0, 3).map((m: any) => m.model_name);
                        setSelectedPublicNames(new Set(defaults));
                    }
                }

            } catch (err) {
                console.error("Failed to fetch benchmark data:", err);
                setError(err instanceof Error ? err.message : "Failed to load benchmark data");
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

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
            localResults.forEach((result, idx) => {
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

    // Export to CSV
    const exportCSV = () => {
        const headers = ["Model", ...displayedTasks, "Average"];
        const rows = dataKeys.map(dk => {
            const scores = displayedTasks.map(task => {
                const dataPoint = chartData.find(d => d.task === task);
                return dataPoint?.[dk.name] || 0;
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

    const handleDeleteLocal = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation(); // Prevent row selection
        if (!window.confirm('Are you sure you want to delete this benchmark result? This cannot be undone.')) {
            return;
        }
        try {
            const res = await fetch(`/api/v1/finetune/lm-eval/jobs/${id}`, { method: 'DELETE' });
            if (res.ok) {
                // Remove from local state
                setLocalResults(prev => prev.filter(r => r.id !== id));
                setSelectedLocalIds(prev => {
                    const next = new Set(prev);
                    next.delete(id);
                    return next;
                });
            } else {
                console.error('Failed to delete benchmark:', await res.text());
            }
        } catch (err) {
            console.error('Delete error:', err);
        }
    };

    // Filter models
    const filteredLocal = localResults.filter(r =>
        r.model_name.toLowerCase().includes(searchQuery.toLowerCase())
    );
    const filteredPublic = publicModels.filter(m =>
        m.model_name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    // Group public models by organization, sorted by benchmark coverage
    const groupedPublicModels = useMemo(() => {
        // Priority order for organizations (top orgs first)
        const orgPriority: Record<string, number> = {
            'OpenAI': 1,
            'Anthropic': 2,
            'Google': 3,
            'Meta': 4,
            'Mistral AI': 5,
            'Alibaba Cloud / Qwen Team': 6,
            'Microsoft': 7,
        };

        // Group models by organization
        const groups: Record<string, typeof filteredPublic> = {};
        filteredPublic.forEach(m => {
            const org = m.organization || 'Other';
            if (!groups[org]) groups[org] = [];
            groups[org].push(m);
        });

        // Sort models within each group by benchmark coverage (descending)
        Object.values(groups).forEach(models => {
            models.sort((a, b) => Object.keys(b.scores || {}).length - Object.keys(a.scores || {}).length);
        });

        // Sort groups by priority (known orgs first, then alphabetically)
        const sortedOrgs = Object.keys(groups).sort((a, b) => {
            const aPri = orgPriority[a] || 100;
            const bPri = orgPriority[b] || 100;
            if (aPri !== bPri) return aPri - bPri;
            return a.localeCompare(b);
        });

        return sortedOrgs.map(org => ({ org, models: groups[org] }));
    }, [filteredPublic]);

    if (loading) {
        return (
            <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "60vh" }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box sx={{ p: 3, maxWidth: 1800, mx: "auto", display: "flex", flexDirection: "column", height: "calc(100vh - 100px)", overflow: "hidden" }}>

            {/* Header */}
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                    <IconButton onClick={() => navigate("/evaluation")} sx={{ color: "text.secondary" }}>
                        <BackIcon />
                    </IconButton>
                    <Box>
                        <Typography variant="h5" sx={{ fontWeight: 700 }}>
                            Benchmark Comparison
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            {dataKeys.length} models • {displayedTasks.length} benchmarks
                        </Typography>
                    </Box>
                </Box>

                <Box sx={{ display: "flex", gap: 2, alignItems: "center" }}>

                    {/* View Toggles */}
                    <Box sx={{ display: "flex", bgcolor: "rgba(255,255,255,0.05)", borderRadius: 1, p: 0.5 }}>
                        {[
                            { mode: "radar" as ViewMode, icon: <RadarIcon />, label: "Radar" },
                            { mode: "bar" as ViewMode, icon: <BarChartIcon />, label: "Bar" },
                            { mode: "table" as ViewMode, icon: <TableIcon />, label: "Table" },
                        ].map(({ mode, icon, label }) => (
                            <Tooltip key={mode} title={label}>
                                <IconButton
                                    size="small"
                                    onClick={() => setViewMode(mode)}
                                    sx={{
                                        color: viewMode === mode ? colors.primary : "text.secondary",
                                        bgcolor: viewMode === mode ? alpha(colors.primary, 0.1) : "transparent",
                                    }}
                                >
                                    {icon}
                                </IconButton>
                            </Tooltip>
                        ))}
                    </Box>

                    <Button
                        variant="outlined"
                        startIcon={<RefreshIcon />}
                        onClick={() => window.location.reload()}
                        size="small"
                        sx={{ borderColor: alpha(colors.slate, 0.3) }}
                    >
                        Refresh
                    </Button>
                    <Button
                        variant="contained"
                        startIcon={<DownloadIcon />}
                        onClick={exportCSV}
                        disabled={dataKeys.length === 0}
                        size="small"
                        sx={{ bgcolor: colors.primary }}
                    >
                        Export
                    </Button>
                </Box>
            </Box>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

            {/* Top Section - Chart */}
            <Paper
                sx={{
                    p: 3,
                    mb: 3,
                    bgcolor: "rgba(0,0,0,0.2)",
                    border: "1px solid rgba(255,255,255,0.06)",
                    flex: 1,
                    minHeight: 400,
                    display: "flex",
                    flexDirection: "column",
                    overflow: "hidden"
                }}
            >
                {dataKeys.length === 0 ? (
                    <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", opacity: 0.5 }}>
                        <TrendIcon sx={{ fontSize: 64, mb: 2 }} />
                        <Typography>Select models below to begin comparison</Typography>
                    </Box>
                ) : (
                    <Box sx={{ flex: 1, minHeight: 0 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            {viewMode === "radar" ? (
                                <RadarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                    <PolarAngleAxis
                                        dataKey="task"
                                        tick={{ fill: "#94a3b8", fontSize: 12 }}
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
                                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />
                                    <RechartsTooltip content={<CustomTooltip />} />
                                </RadarChart>
                            ) : viewMode === "bar" ? (
                                <BarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                                    <XAxis dataKey="task" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                                    <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                                    <RechartsTooltip content={<CustomTooltip />} />
                                    <Legend wrapperStyle={{ fontSize: 12, paddingTop: 20 }} />
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
                            ) : (
                                // Table view is handled separately or could be embedded here
                                <Box sx={{ height: "100%", overflow: "auto" }}>
                                    {/* Simplified inline table for this view */}
                                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                                        <thead>
                                            <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                                                <th style={{ padding: 8, textAlign: "left" }}>Task</th>
                                                {dataKeys.map(dk => <th key={dk.name} style={{ padding: 8, textAlign: "right" }}>{dk.name}</th>)}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {chartData.map((d: any) => (
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
                                </Box>
                            )}
                        </ResponsiveContainer>
                    </Box>
                )}
            </Paper>

            {/* Bottom Controls Section */}
            <Box
                sx={{
                    height: 380,
                    display: "flex",
                    gap: 3,
                    overflow: "hidden"
                }}
            >
                {/* 1. Model Selection */}
                <Paper sx={{ width: "25%", display: "flex", flexDirection: "column", bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <Box sx={{ p: 2, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 1 }}>
                            <ModelIcon fontSize="small" color="primary" /> Result Selection
                        </Typography>
                        <TextField
                            placeholder="Filter..."
                            size="small"
                            fullWidth
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            sx={{ mt: 1, "& .MuiInputBase-root": { fontSize: "0.85rem" } }}
                        />
                    </Box>
                    <Box sx={{ flex: 1, overflowY: "auto", p: 1 }}>
                        {filteredLocal.map((result, idx) => (
                            <Box
                                key={result.id}
                                onClick={() => toggleLocal(result.id)}
                                sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 1,
                                    p: 1,
                                    borderRadius: 1,
                                    cursor: "pointer",
                                    bgcolor: selectedLocalIds.has(result.id) ? alpha(colors.success, 0.1) : "transparent",
                                    border: `1px solid ${selectedLocalIds.has(result.id) ? alpha(colors.success, 0.3) : "transparent"}`,
                                    mb: 0.5,
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
                                    onClick={(e) => handleDeleteLocal(result.id, e)}
                                    sx={{
                                        p: 0.25,
                                        ml: 0.5,
                                        opacity: 0.5,
                                        '&:hover': { opacity: 1, color: 'error.main' }
                                    }}
                                >
                                    <DeleteIcon sx={{ fontSize: 14 }} />
                                </IconButton>
                            </Box>
                        ))}
                    </Box>
                </Paper>

                {/* 2. Public References */}
                <Paper sx={{ width: "25%", display: "flex", flexDirection: "column", bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <Box sx={{ p: 2, borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 1 }}>
                            <PublicIcon fontSize="small" color="secondary" /> Reference Items
                        </Typography>
                        <Switch
                            size="small"
                            checked={showPublicModels}
                            onChange={(e) => setShowPublicModels(e.target.checked)}
                        />
                    </Box>
                    <Box sx={{ flex: 1, overflowY: "auto", p: 1, opacity: showPublicModels ? 1 : 0.5, pointerEvents: showPublicModels ? "auto" : "none" }}>
                        {groupedPublicModels.map((group, gIdx) => (
                            <Box key={group.org} sx={{ mb: 1 }}>
                                {/* Organization Header */}
                                <Typography
                                    variant="caption"
                                    sx={{
                                        display: 'block',
                                        px: 1,
                                        py: 0.5,
                                        mb: 0.5,
                                        fontWeight: 600,
                                        color: 'text.secondary',
                                        fontSize: '0.7rem',
                                        textTransform: 'uppercase',
                                        letterSpacing: 0.5,
                                        borderBottom: '1px solid rgba(255,255,255,0.05)',
                                    }}
                                >
                                    {group.org} ({group.models.length})
                                </Typography>
                                {/* Models in this organization */}
                                {group.models.map((model, idx) => (
                                    <Box
                                        key={model.model_name}
                                        onClick={() => togglePublic(model.model_name)}
                                        sx={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 1,
                                            p: 0.75,
                                            pl: 2,
                                            borderRadius: 1,
                                            cursor: "pointer",
                                            bgcolor: selectedPublicNames.has(model.model_name) ? alpha(colors.purple, 0.1) : "transparent",
                                            border: `1px solid ${selectedPublicNames.has(model.model_name) ? alpha(colors.purple, 0.3) : "transparent"}`,
                                            mb: 0.25,
                                            "&:hover": { bgcolor: alpha(colors.purple, 0.05) },
                                        }}
                                    >
                                        <Box sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: publicModelColors[(gIdx * 5 + idx) % publicModelColors.length] }} />
                                        <Typography variant="caption" sx={{ flex: 1, fontSize: '0.75rem' }}>
                                            {model.model_name}
                                        </Typography>
                                        <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: '0.65rem' }}>
                                            {Object.keys(model.scores || {}).length}
                                        </Typography>
                                        {selectedPublicNames.has(model.model_name) && <CheckIcon sx={{ fontSize: 12, color: colors.purple }} />}
                                    </Box>
                                ))}
                            </Box>
                        ))}
                    </Box>
                </Paper>

                {/* 3. Benchmark Config (Presets & Aggregation) */}
                <Paper sx={{ width: "25%", display: "flex", flexDirection: "column", bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <Box sx={{ p: 2, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 1 }}>
                            <FilterIcon fontSize="small" sx={{ color: colors.info }} /> Benchmark View
                        </Typography>
                    </Box>

                    <Box sx={{ p: 2, display: "flex", flexDirection: "column", gap: 3 }}>
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

                        <Paper variant="outlined" sx={{ p: 2, bgcolor: alpha(colors.slate, 0.1) }}>
                            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 1 }}>
                                <Typography variant="body2" fontWeight={600}>Smart Aggregation</Typography>
                                <Switch
                                    size="small"
                                    checked={useAggregates}
                                    onChange={(e) => setUseAggregates(e.target.checked)}
                                />
                            </Box>
                            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                                Example: Combines "mmlu_humanities", "mmlu_stem", etc. into a single "MMLU (Avg)" score.
                            </Typography>

                            {useAggregates && (
                                <Box sx={{ mt: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
                                    {AGGREGATION_GROUPS.map(g => (
                                        <Chip
                                            key={g.name}
                                            label={g.name}
                                            size="small"
                                            icon={<SummaryIcon sx={{ fontSize: "14px !important" }} />}
                                            sx={{ borderColor: alpha(g.color, 0.3), height: 24 }}
                                            variant="outlined"
                                        />
                                    ))}
                                </Box>
                            )}
                        </Paper>

                        <Box>
                            <Typography variant="caption" color="text.secondary">
                                Currently showing <strong>{displayedTasks.length}</strong> benchmarks.
                            </Typography>
                        </Box>
                    </Box>
                </Paper>

                {/* 4. Active Benchmarks List */}
                <Paper sx={{ width: "25%", display: "flex", flexDirection: "column", bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <Box sx={{ p: 2, borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, display: "flex", alignItems: "center", gap: 1 }}>
                            <TableIcon fontSize="small" sx={{ color: colors.warning }} /> Active Metrics
                        </Typography>
                        <TextField
                            placeholder="Filter metrics..."
                            size="small"
                            fullWidth
                            value={datasetSearchQuery}
                            onChange={(e) => setDatasetSearchQuery(e.target.value)}
                            sx={{ mt: 1 }}
                        />
                    </Box>
                    <Box sx={{ flex: 1, overflowY: "auto", p: 1 }}>
                        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                            {displayedTasks
                                .filter(t => t.toLowerCase().includes(datasetSearchQuery.toLowerCase()))
                                .map(task => (
                                    <Chip
                                        key={task}
                                        label={task}
                                        size="small"
                                        sx={{
                                            height: 20,
                                            fontSize: "0.7rem",
                                            bgcolor: AGGREGATION_GROUPS.some(g => g.name === task) ? alpha(colors.success, 0.1) : alpha(colors.slate, 0.1),
                                            color: AGGREGATION_GROUPS.some(g => g.name === task) ? colors.success : "text.secondary"
                                        }}
                                    />
                                ))
                            }
                        </Box>
                    </Box>
                </Paper>

            </Box>
        </Box>
    );
};

export default BenchmarkComparisonPage;
