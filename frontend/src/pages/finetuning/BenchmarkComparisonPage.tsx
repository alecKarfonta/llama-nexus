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
} from "@mui/material";
import {
    ArrowBack as BackIcon,
    Refresh as RefreshIcon,
    Download as DownloadIcon,
    Add as AddIcon,
    Search as SearchIcon,
    CheckCircle as CheckIcon,
    RadioButtonUnchecked as UncheckedIcon,
    BarChart as BarChartIcon,
    DonutLarge as RadarIcon,
    TableChart as TableIcon,
    Public as PublicIcon,
    Storage as ModelIcon,
    TrendingUp as TrendIcon,
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

// Public model colors (muted)
const publicModelColors = [
    "#94a3b8",
    "#a1a1aa",
    "#9ca3af",
    "#a3a3a3",
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

// Custom tooltip
const CustomTooltip: React.FC<{ active?: boolean; payload?: any[]; label?: string }> = ({
    active, payload, label
}) => {
    if (active && payload?.length) {
        return (
            <Paper sx={{ p: 1.5, bgcolor: "rgba(15, 23, 42, 0.95)", border: "1px solid rgba(255,255,255,0.1)" }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>{label}</Typography>
                {payload.map((entry: any, i: number) => (
                    <Box key={i} sx={{ display: "flex", alignItems: "center", gap: 1, mt: 0.5 }}>
                        <Box sx={{ width: 8, height: 8, borderRadius: "50%", bgcolor: entry.color }} />
                        <Typography variant="caption" sx={{ color: "text.secondary" }}>
                            {entry.name}: <strong style={{ color: "#fff" }}>{entry.value?.toFixed(1)}%</strong>
                        </Typography>
                    </Box>
                ))}
            </Paper>
        );
    }
    return null;
};

const BenchmarkComparisonPage: React.FC = () => {
    const navigate = useNavigate();

    // Data state
    const [localResults, setLocalResults] = useState<BenchmarkResult[]>([]);
    const [publicModels, setPublicModels] = useState<PublicModel[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Selection state
    const [selectedLocalIds, setSelectedLocalIds] = useState<Set<string>>(new Set());
    const [selectedPublicNames, setSelectedPublicNames] = useState<Set<string>>(new Set());
    const [showPublicModels, setShowPublicModels] = useState(false);

    // View state
    const [viewMode, setViewMode] = useState<ViewMode>("radar");
    const [searchQuery, setSearchQuery] = useState("");

    // Available benchmark tasks
    const allTasks = useMemo(() => {
        const tasks = new Set<string>();
        localResults.forEach(r => Object.keys(r.tasks).forEach(t => tasks.add(t)));
        publicModels.forEach(m => Object.keys(m.scores).forEach(t => tasks.add(t)));
        return Array.from(tasks).sort();
    }, [localResults, publicModels]);

    const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set(allTasks));

    // Update selected tasks when all tasks change
    useEffect(() => {
        if (allTasks.length > 0 && selectedTasks.size === 0) {
            setSelectedTasks(new Set(allTasks));
        }
    }, [allTasks]);

    // Fetch data
    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                // Fetch local benchmark jobs
                const jobsRes = await fetch("/api/v1/finetune/lm-eval/jobs");
                if (jobsRes.ok) {
                    const data = await jobsRes.json();
                    const jobs = data.jobs || [];
                    const completed = jobs.filter((j: any) => j.status === "completed" && j.results?.tasks);
                    setLocalResults(completed.map((j: any) => ({
                        id: j.id,
                        model_name: j.model_path || j.results?.model_name || `Job ${j.id.slice(0, 8)}`,
                        model_path: j.model_path,
                        created_at: j.created_at,
                        tasks: j.results?.tasks || {},
                    })));

                    // Auto-select first 3
                    const autoSelect = new Set<string>();
                    completed.slice(0, 3).forEach((j: any) => autoSelect.add(j.id));
                    setSelectedLocalIds(autoSelect);
                }

                // Fetch public benchmarks
                const publicRes = await fetch("/api/v1/finetune/lm-eval/public-benchmarks?limit=10");
                if (publicRes.ok) {
                    const data = await publicRes.json();
                    setPublicModels(data.benchmarks || []);
                }
            } catch (e) {
                setError("Failed to load benchmark data");
            }
            setLoading(false);
        };
        fetchData();
    }, []);

    // Prepare chart data
    const chartData = useMemo(() => {
        const tasks = Array.from(selectedTasks);
        return tasks.map(task => {
            const point: Record<string, string | number> = { task };

            // Add local models
            localResults.forEach((result, idx) => {
                if (selectedLocalIds.has(result.id)) {
                    point[result.model_name] = result.tasks[task]?.score || 0;
                }
            });

            // Add public models
            if (showPublicModels) {
                publicModels.forEach(model => {
                    if (selectedPublicNames.has(model.model_name)) {
                        point[model.model_name] = model.scores[task] || 0;
                    }
                });
            }

            return point;
        });
    }, [selectedTasks, localResults, selectedLocalIds, publicModels, selectedPublicNames, showPublicModels]);

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
        const headers = ["Model", ...Array.from(selectedTasks), "Average"];
        const rows = dataKeys.map(dk => {
            const scores = Array.from(selectedTasks).map(task => {
                const dataPoint = chartData.find(d => d.task === task);
                return dataPoint?.[dk.name] || 0;
            });
            const avg = scores.reduce((a, b) => Number(a) + Number(b), 0) / scores.length;
            return [dk.name, ...scores.map(s => s.toFixed(1)), avg.toFixed(1)];
        });

        const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "benchmark_comparison.csv";
        a.click();
    };

    // Toggle model selection
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

    const toggleTask = (task: string) => {
        setSelectedTasks(prev => {
            const next = new Set(prev);
            if (next.has(task)) next.delete(task);
            else next.add(task);
            return next;
        });
    };

    // Filter models by search
    const filteredLocal = localResults.filter(r =>
        r.model_name.toLowerCase().includes(searchQuery.toLowerCase())
    );
    const filteredPublic = publicModels.filter(m =>
        m.model_name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    if (loading) {
        return (
            <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: "60vh" }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box sx={{ p: 3, maxWidth: 1600, mx: "auto" }}>
            {/* Header */}
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                    <IconButton onClick={() => navigate("/evaluation")} sx={{ color: "text.secondary" }}>
                        <BackIcon />
                    </IconButton>
                    <Box>
                        <Typography variant="h4" sx={{ fontWeight: 700, background: "linear-gradient(135deg, #fff 0%, #94a3b8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                            Benchmark Comparison
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Compare LLM performance across standardized benchmarks
                        </Typography>
                    </Box>
                </Box>

                <Box sx={{ display: "flex", gap: 1 }}>
                    <Button
                        variant="outlined"
                        startIcon={<RefreshIcon />}
                        onClick={() => window.location.reload()}
                        sx={{ borderColor: alpha(colors.slate, 0.3) }}
                    >
                        Refresh
                    </Button>
                    <Button
                        variant="contained"
                        startIcon={<DownloadIcon />}
                        onClick={exportCSV}
                        disabled={dataKeys.length === 0}
                        sx={{ bgcolor: colors.primary }}
                    >
                        Export CSV
                    </Button>
                </Box>
            </Box>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

            {/* Main Layout */}
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "280px 1fr" }, gap: 3 }}>

                {/* Sidebar - Model Selection */}
                <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>

                    {/* Search */}
                    <TextField
                        size="small"
                        placeholder="Search models..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        InputProps={{
                            startAdornment: <InputAdornment position="start"><SearchIcon sx={{ color: "text.secondary" }} /></InputAdornment>,
                        }}
                        sx={{
                            "& .MuiOutlinedInput-root": {
                                bgcolor: "rgba(0,0,0,0.2)",
                                "&:hover": { bgcolor: "rgba(0,0,0,0.3)" },
                            }
                        }}
                    />

                    {/* Local Models */}
                    <Paper sx={{ p: 2, bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
                            <ModelIcon sx={{ color: colors.success, fontSize: 18 }} />
                            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                Your Models ({localResults.length})
                            </Typography>
                        </Box>

                        {localResults.length === 0 ? (
                            <Typography variant="caption" color="text.secondary">
                                No benchmark results yet. Run benchmarks from the Evaluation page.
                            </Typography>
                        ) : (
                            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, maxHeight: 200, overflowY: "auto" }}>
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
                                            border: selectedLocalIds.has(result.id) ? `1px solid ${alpha(colors.success, 0.3)}` : "1px solid transparent",
                                            "&:hover": { bgcolor: alpha(colors.success, 0.05) },
                                        }}
                                    >
                                        <Box sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: modelColors[idx % modelColors.length] }} />
                                        {selectedLocalIds.has(result.id) ?
                                            <CheckIcon sx={{ fontSize: 16, color: colors.success }} /> :
                                            <UncheckedIcon sx={{ fontSize: 16, color: "text.secondary" }} />
                                        }
                                        <Typography variant="caption" sx={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                            {result.model_name}
                                        </Typography>
                                    </Box>
                                ))}
                            </Box>
                        )}
                    </Paper>

                    {/* Public Models Toggle */}
                    <Paper sx={{ p: 2, bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5 }}>
                            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                <PublicIcon sx={{ color: colors.purple, fontSize: 18 }} />
                                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                                    Reference Models
                                </Typography>
                            </Box>
                            <Switch
                                size="small"
                                checked={showPublicModels}
                                onChange={(e) => setShowPublicModels(e.target.checked)}
                                sx={{ "& .Mui-checked": { color: colors.purple } }}
                            />
                        </Box>

                        {showPublicModels && (
                            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5, maxHeight: 180, overflowY: "auto" }}>
                                {filteredPublic.map((model, idx) => (
                                    <Box
                                        key={model.model_name}
                                        onClick={() => togglePublic(model.model_name)}
                                        sx={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 1,
                                            p: 1,
                                            borderRadius: 1,
                                            cursor: "pointer",
                                            bgcolor: selectedPublicNames.has(model.model_name) ? alpha(colors.purple, 0.1) : "transparent",
                                            border: selectedPublicNames.has(model.model_name) ? `1px solid ${alpha(colors.purple, 0.3)}` : "1px solid transparent",
                                            "&:hover": { bgcolor: alpha(colors.purple, 0.05) },
                                        }}
                                    >
                                        <Box sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: publicModelColors[idx % publicModelColors.length] }} />
                                        {selectedPublicNames.has(model.model_name) ?
                                            <CheckIcon sx={{ fontSize: 16, color: colors.purple }} /> :
                                            <UncheckedIcon sx={{ fontSize: 16, color: "text.secondary" }} />
                                        }
                                        <Typography variant="caption" sx={{ flex: 1 }}>
                                            {model.model_name}
                                        </Typography>
                                        {model.model_size && (
                                            <Chip label={model.model_size} size="small" sx={{ height: 18, fontSize: "0.65rem" }} />
                                        )}
                                    </Box>
                                ))}
                            </Box>
                        )}
                    </Paper>

                    {/* Task Filter */}
                    <Paper sx={{ p: 2, bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5 }}>
                            Benchmarks ({allTasks.length})
                        </Typography>
                        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
                            {allTasks.map(task => (
                                <Chip
                                    key={task}
                                    label={task}
                                    size="small"
                                    onClick={() => toggleTask(task)}
                                    sx={{
                                        bgcolor: selectedTasks.has(task) ? alpha(colors.info, 0.2) : "transparent",
                                        border: `1px solid ${selectedTasks.has(task) ? alpha(colors.info, 0.4) : "rgba(255,255,255,0.1)"}`,
                                        color: selectedTasks.has(task) ? colors.info : "text.secondary",
                                        cursor: "pointer",
                                        "&:hover": { bgcolor: alpha(colors.info, 0.1) },
                                    }}
                                />
                            ))}
                        </Box>
                    </Paper>
                </Box>

                {/* Main Content - Visualizations */}
                <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>

                    {/* View Mode Toggle */}
                    <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            <TrendIcon sx={{ color: colors.success }} />
                            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                {dataKeys.length} model{dataKeys.length !== 1 ? "s" : ""} selected
                            </Typography>
                        </Box>

                        <Box sx={{ display: "flex", gap: 0.5 }}>
                            {[
                                { mode: "radar" as ViewMode, icon: <RadarIcon />, label: "Radar" },
                                { mode: "bar" as ViewMode, icon: <BarChartIcon />, label: "Bar" },
                                { mode: "table" as ViewMode, icon: <TableIcon />, label: "Table" },
                            ].map(({ mode, icon, label }) => (
                                <Tooltip key={mode} title={label}>
                                    <IconButton
                                        onClick={() => setViewMode(mode)}
                                        sx={{
                                            bgcolor: viewMode === mode ? alpha(colors.primary, 0.2) : "transparent",
                                            border: `1px solid ${viewMode === mode ? alpha(colors.primary, 0.4) : "rgba(255,255,255,0.1)"}`,
                                            color: viewMode === mode ? colors.primary : "text.secondary",
                                        }}
                                    >
                                        {icon}
                                    </IconButton>
                                </Tooltip>
                            ))}
                        </Box>
                    </Box>

                    {/* Chart Area */}
                    {dataKeys.length === 0 ? (
                        <Paper sx={{ p: 6, textAlign: "center", bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                            <TrendIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                            <Typography color="text.secondary">Select models to compare</Typography>
                            <Typography variant="caption" color="text.secondary">
                                Choose models from the sidebar to see comparison charts
                            </Typography>
                        </Paper>
                    ) : viewMode === "radar" ? (
                        <Paper sx={{ p: 3, bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                            <ResponsiveContainer width="100%" height={450}>
                                <RadarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                                    <PolarGrid stroke="rgba(255,255,255,0.1)" />
                                    <PolarAngleAxis dataKey="task" tick={{ fill: "#94a3b8", fontSize: 12 }} />
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
                                    <Legend wrapperStyle={{ fontSize: 12 }} />
                                    <RechartsTooltip content={<CustomTooltip />} />
                                </RadarChart>
                            </ResponsiveContainer>
                        </Paper>
                    ) : viewMode === "bar" ? (
                        <Paper sx={{ p: 3, bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)" }}>
                            <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 50)}>
                                <BarChart data={chartData} layout="vertical" margin={{ left: 100, right: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                                    <XAxis type="number" domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                                    <YAxis type="category" dataKey="task" tick={{ fill: "#94a3b8", fontSize: 11 }} width={100} />
                                    {dataKeys.map((dk, idx) => (
                                        <Bar
                                            key={dk.name}
                                            dataKey={dk.name}
                                            fill={dk.isPublic ? publicModelColors[idx % publicModelColors.length] : modelColors[idx % modelColors.length]}
                                            fillOpacity={dk.isPublic ? 0.6 : 0.9}
                                            radius={[0, 4, 4, 0]}
                                        />
                                    ))}
                                    <Legend wrapperStyle={{ fontSize: 12 }} />
                                    <RechartsTooltip content={<CustomTooltip />} />
                                </BarChart>
                            </ResponsiveContainer>
                        </Paper>
                    ) : (
                        /* Table View */
                        <Paper sx={{ bgcolor: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.06)", overflow: "hidden" }}>
                            <Box sx={{ overflowX: "auto" }}>
                                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                                    <thead>
                                        <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                                            <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 600 }}>Model</th>
                                            {Array.from(selectedTasks).map(task => (
                                                <th key={task} style={{ padding: "12px 16px", textAlign: "right", fontWeight: 500, color: "#94a3b8" }}>{task}</th>
                                            ))}
                                            <th style={{ padding: "12px 16px", textAlign: "right", fontWeight: 600, color: colors.success }}>Average</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {dataKeys.map((dk, idx) => {
                                            const scores = Array.from(selectedTasks).map(task => {
                                                const point = chartData.find(d => d.task === task);
                                                return Number(point?.[dk.name]) || 0;
                                            });
                                            const avg = scores.reduce((a, b) => a + b, 0) / scores.length;

                                            return (
                                                <tr key={dk.name} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                                                    <td style={{ padding: "12px 16px" }}>
                                                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                                            <Box sx={{
                                                                width: 10, height: 10, borderRadius: "50%",
                                                                bgcolor: dk.isPublic ? publicModelColors[idx % publicModelColors.length] : modelColors[idx % modelColors.length]
                                                            }} />
                                                            <Typography variant="body2" sx={{ fontWeight: 500 }}>{dk.name}</Typography>
                                                            {dk.isPublic && <Chip label="Reference" size="small" sx={{ height: 18, fontSize: "0.6rem" }} />}
                                                        </Box>
                                                    </td>
                                                    {scores.map((score, i) => (
                                                        <td key={i} style={{ padding: "12px 16px", textAlign: "right" }}>
                                                            <Typography variant="body2" sx={{
                                                                fontWeight: 500,
                                                                color: score >= 70 ? colors.success : score >= 50 ? colors.warning : colors.rose,
                                                            }}>
                                                                {score.toFixed(1)}%
                                                            </Typography>
                                                        </td>
                                                    ))}
                                                    <td style={{ padding: "12px 16px", textAlign: "right" }}>
                                                        <Typography variant="body2" sx={{ fontWeight: 700, color: colors.success }}>
                                                            {avg.toFixed(1)}%
                                                        </Typography>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </Box>
                        </Paper>
                    )}
                </Box>
            </Box>
        </Box>
    );
};

export default BenchmarkComparisonPage;
