/**
 * Benchmark Results Panel - Comprehensive visualization for LM-Eval results
 * 
 * Features:
 * - Model info display (filename, date, GPU)
 * - Radar chart for multi-task overview
 * - Bar chart for model comparison
 * - Public benchmark data overlay
 * - Job selection for comparison
 */

import React, { useEffect, useState, useCallback, useMemo } from "react";
import {
    Box,
    Typography,
    Chip,
    IconButton,
    Tooltip,
    ToggleButton,
    ToggleButtonGroup,
    Checkbox,
    Button,
    Alert,
    LinearProgress,
    Divider,
    Collapse,
    alpha,
} from "@mui/material";
import {
    Assessment as EvalIcon,
    BarChart as BarChartIcon,
    DonutLarge as RadarIcon,
    ViewList as TableIcon,
    CompareArrows as CompareIcon,
    Public as PublicIcon,
    ExpandMore as ExpandIcon,
    ExpandLess as CollapseIcon,
    Memory as GpuIcon,
    AccessTime as DateIcon,
    Storage as ModelIcon,
    Delete as DeleteIcon,
} from "@mui/icons-material";
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

// Accent colors matching ModelEvaluationPage
const accentColors = {
    primary: "#6366f1",
    success: "#10b981",
    warning: "#f59e0b",
    info: "#06b6d4",
    purple: "#8b5cf6",
    rose: "#f43f5e",
};

// Chart colors for multiple models
const chartColors = [
    "#10b981", // Green - local model
    "#6366f1", // Purple
    "#f59e0b", // Orange
    "#06b6d4", // Cyan
    "#f43f5e", // Rose
    "#8b5cf6", // Violet
];

// Public reference colors (more muted)
const publicColors = [
    "#94a3b8", // Slate
    "#a1a1aa", // Zinc
    "#a3a3a3", // Neutral
];

// Types
export type LMEvalJob = {
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
};

export type PublicBenchmark = {
    model_name: string;
    model_size?: string;
    scores: Record<string, number>;
    source: string;
};

interface BenchmarkResultsPanelProps {
    jobs: LMEvalJob[];
    publicBenchmarks?: PublicBenchmark[];
    onDeleteJob?: (jobId: string) => void;
    onLoadPublicData?: () => void;
    isLoadingPublic?: boolean;
}

type ViewMode = "table" | "radar" | "bar";

// Status chip component
const StatusChip: React.FC<{ status: string }> = ({ status }) => {
    const config: Record<string, { color: string; label: string }> = {
        completed: { color: accentColors.success, label: "Completed" },
        running: { color: accentColors.info, label: "Running" },
        failed: { color: accentColors.rose, label: "Failed" },
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

// Custom tooltip for charts
interface TooltipPayload {
    name: string;
    value: number;
    color: string;
}

interface ChartTooltipProps {
    active?: boolean;
    payload?: TooltipPayload[];
    label?: string;
}

const ChartTooltip: React.FC<ChartTooltipProps> = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <Box
                sx={{
                    bgcolor: "rgba(15, 15, 35, 0.95)",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: 1.5,
                    p: 1.5,
                    boxShadow: "0 4px 20px rgba(0, 0, 0, 0.4)",
                }}
            >
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    {label}
                </Typography>
                {payload.map((entry, index) => (
                    <Typography
                        key={index}
                        variant="body2"
                        sx={{ color: entry.color, fontSize: "0.8rem" }}
                    >
                        {entry.name}: <strong>{entry.value?.toFixed(1)}%</strong>
                    </Typography>
                ))}
            </Box>
        );
    }
    return null;
};

export const BenchmarkResultsPanel: React.FC<BenchmarkResultsPanelProps> = ({
    jobs,
    publicBenchmarks = [],
    onDeleteJob,
    onLoadPublicData,
    isLoadingPublic = false,
}) => {
    const [viewMode, setViewMode] = useState<ViewMode>("bar");
    const [selectedJobIds, setSelectedJobIds] = useState<Set<string>>(new Set());
    const [showPublicData, setShowPublicData] = useState(false);
    const [expandedJobId, setExpandedJobId] = useState<string | null>(null);

    // Filter completed jobs with results
    const completedJobs = useMemo(
        () => jobs.filter((j) => j.status === "completed" && j.results?.tasks),
        [jobs]
    );

    // Auto-select first 3 completed jobs if none selected
    const effectiveSelection = useMemo(() => {
        if (selectedJobIds.size > 0) return selectedJobIds;
        const autoSelect = new Set<string>();
        completedJobs.slice(0, 3).forEach((j) => autoSelect.add(j.id));
        return autoSelect;
    }, [selectedJobIds, completedJobs]);

    // Get all unique task names across selected jobs
    const allTasks = useMemo(() => {
        const tasks = new Set<string>();
        completedJobs.forEach((job) => {
            if (effectiveSelection.has(job.id)) {
                Object.keys(job.results?.tasks || {}).forEach((t) => tasks.add(t));
            }
        });
        return Array.from(tasks);
    }, [completedJobs, effectiveSelection]);

    // Prepare radar chart data
    const radarData = useMemo(() => {
        return allTasks.map((task) => {
            const point: Record<string, string | number> = { task };
            completedJobs.forEach((job) => {
                if (effectiveSelection.has(job.id)) {
                    const modelName = job.model_path || job.results?.model_name || `Job ${job.id.slice(0, 8)}`;
                    point[modelName] = job.results?.tasks?.[task]?.score || 0;
                }
            });
            // Add public benchmarks
            if (showPublicData) {
                publicBenchmarks.forEach((pb) => {
                    point[pb.model_name] = pb.scores[task] || 0;
                });
            }
            return point;
        });
    }, [allTasks, completedJobs, effectiveSelection, showPublicData, publicBenchmarks]);

    // Prepare bar chart data
    const barData = useMemo(() => {
        return allTasks.map((task) => {
            const bar: Record<string, string | number> = { task };
            completedJobs.forEach((job) => {
                if (effectiveSelection.has(job.id)) {
                    const modelName = job.model_path || job.results?.model_name || `Job ${job.id.slice(0, 8)}`;
                    bar[modelName] = job.results?.tasks?.[task]?.score || 0;
                }
            });
            if (showPublicData) {
                publicBenchmarks.forEach((pb) => {
                    bar[pb.model_name] = pb.scores[task] || 0;
                });
            }
            return bar;
        });
    }, [allTasks, completedJobs, effectiveSelection, showPublicData, publicBenchmarks]);

    // Get data keys for charts (model names)
    const dataKeys = useMemo(() => {
        const keys: string[] = [];
        completedJobs.forEach((job) => {
            if (effectiveSelection.has(job.id)) {
                keys.push(job.model_path || job.results?.model_name || `Job ${job.id.slice(0, 8)}`);
            }
        });
        if (showPublicData) {
            publicBenchmarks.forEach((pb) => keys.push(pb.model_name));
        }
        return keys;
    }, [completedJobs, effectiveSelection, showPublicData, publicBenchmarks]);

    const toggleJobSelection = (jobId: string) => {
        setSelectedJobIds((prev) => {
            const next = new Set(prev);
            if (next.has(jobId)) {
                next.delete(jobId);
            } else {
                next.add(jobId);
            }
            return next;
        });
    };

    const handleLoadPublicData = () => {
        setShowPublicData(true);
        onLoadPublicData?.();
    };

    if (jobs.length === 0) {
        return (
            <Box sx={{ p: 4, textAlign: "center" }}>
                <EvalIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                <Typography color="text.secondary">No benchmark jobs yet</Typography>
                <Typography variant="caption" color="text.secondary">
                    Select tasks and run to get standardized evaluation scores
                </Typography>
            </Box>
        );
    }

    return (
        <Box>
            {/* Header with view controls */}
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2, flexWrap: "wrap", gap: 1 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {completedJobs.length} completed / {jobs.length} total
                    </Typography>
                    {effectiveSelection.size > 1 && (
                        <Chip
                            icon={<CompareIcon sx={{ fontSize: 14 }} />}
                            label={`Comparing ${effectiveSelection.size}`}
                            size="small"
                            sx={{ bgcolor: alpha(accentColors.info, 0.1), color: accentColors.info }}
                        />
                    )}
                </Box>

                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    {/* Public data toggle */}
                    <Tooltip title="Compare with public benchmarks (Llama-3, Mistral, etc.)">
                        <Button
                            size="small"
                            variant={showPublicData ? "contained" : "outlined"}
                            startIcon={isLoadingPublic ? null : <PublicIcon />}
                            onClick={handleLoadPublicData}
                            disabled={isLoadingPublic}
                            sx={{
                                borderColor: alpha(accentColors.purple, 0.3),
                                color: showPublicData ? "white" : accentColors.purple,
                                bgcolor: showPublicData ? accentColors.purple : "transparent",
                                "&:hover": {
                                    bgcolor: showPublicData ? accentColors.purple : alpha(accentColors.purple, 0.1),
                                },
                            }}
                        >
                            {isLoadingPublic ? "Loading..." : "Public Data"}
                        </Button>
                    </Tooltip>

                    {/* View mode toggle */}
                    <ToggleButtonGroup
                        size="small"
                        value={viewMode}
                        exclusive
                        onChange={(_, v) => v && setViewMode(v)}
                    >
                        <ToggleButton value="bar">
                            <Tooltip title="Bar Chart">
                                <BarChartIcon sx={{ fontSize: 18 }} />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="radar">
                            <Tooltip title="Radar Chart">
                                <RadarIcon sx={{ fontSize: 18 }} />
                            </Tooltip>
                        </ToggleButton>
                        <ToggleButton value="table">
                            <Tooltip title="Table View">
                                <TableIcon sx={{ fontSize: 18 }} />
                            </Tooltip>
                        </ToggleButton>
                    </ToggleButtonGroup>
                </Box>
            </Box>

            {/* Chart View */}
            {viewMode !== "table" && completedJobs.length > 0 && allTasks.length > 0 && (
                <Box
                    sx={{
                        mb: 3,
                        p: 2,
                        bgcolor: "rgba(0, 0, 0, 0.2)",
                        borderRadius: 2,
                        border: "1px solid rgba(255, 255, 255, 0.06)",
                    }}
                >
                    <ResponsiveContainer width="100%" height={320}>
                        {viewMode === "radar" ? (
                            <RadarChart data={radarData}>
                                <PolarGrid stroke="rgba(255, 255, 255, 0.1)" />
                                <PolarAngleAxis
                                    dataKey="task"
                                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                                />
                                <PolarRadiusAxis
                                    angle={30}
                                    domain={[0, 100]}
                                    tick={{ fill: "#64748b", fontSize: 10 }}
                                />
                                {dataKeys.map((key, index) => {
                                    const isPublic = publicBenchmarks.some((pb) => pb.model_name === key);
                                    const color = isPublic
                                        ? publicColors[index % publicColors.length]
                                        : chartColors[index % chartColors.length];
                                    return (
                                        <Radar
                                            key={key}
                                            name={key}
                                            dataKey={key}
                                            stroke={color}
                                            fill={color}
                                            fillOpacity={isPublic ? 0.1 : 0.2}
                                            strokeWidth={isPublic ? 1 : 2}
                                            strokeDasharray={isPublic ? "5 5" : undefined}
                                        />
                                    );
                                })}
                                <Legend
                                    wrapperStyle={{ fontSize: "11px" }}
                                    formatter={(value: string) => (
                                        <span style={{ color: "#94a3b8" }}>{value}</span>
                                    )}
                                />
                                <RechartsTooltip content={<ChartTooltip />} />
                            </RadarChart>
                        ) : (
                            <BarChart data={barData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.06)" />
                                <XAxis
                                    type="number"
                                    domain={[0, 100]}
                                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                                />
                                <YAxis
                                    type="category"
                                    dataKey="task"
                                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                                    width={100}
                                />
                                {dataKeys.map((key, index) => {
                                    const isPublic = publicBenchmarks.some((pb) => pb.model_name === key);
                                    const color = isPublic
                                        ? publicColors[index % publicColors.length]
                                        : chartColors[index % chartColors.length];
                                    return (
                                        <Bar
                                            key={key}
                                            dataKey={key}
                                            fill={color}
                                            fillOpacity={isPublic ? 0.5 : 0.8}
                                            radius={[0, 4, 4, 0]}
                                        />
                                    );
                                })}
                                <Legend
                                    wrapperStyle={{ fontSize: "11px" }}
                                    formatter={(value: string) => (
                                        <span style={{ color: "#94a3b8" }}>{value}</span>
                                    )}
                                />
                                <RechartsTooltip content={<ChartTooltip />} />
                            </BarChart>
                        )}
                    </ResponsiveContainer>
                </Box>
            )}

            {/* Job List */}
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                {jobs.map((job) => {
                    const isSelected = effectiveSelection.has(job.id);
                    const isExpanded = expandedJobId === job.id;
                    const modelName = job.model_path || job.results?.model_name || "Unknown model";

                    return (
                        <Box
                            key={job.id}
                            sx={{
                                p: 2,
                                borderRadius: 2,
                                bgcolor: isSelected ? alpha(accentColors.success, 0.05) : "rgba(0, 0, 0, 0.2)",
                                border: `1px solid ${isSelected ? alpha(accentColors.success, 0.3) : "rgba(255, 255, 255, 0.06)"}`,
                                transition: "all 0.2s",
                            }}
                        >
                            {/* Job Header */}
                            <Box sx={{ display: "flex", alignItems: "flex-start", gap: 1.5 }}>
                                {/* Selection checkbox (only for completed jobs) */}
                                {job.status === "completed" && (
                                    <Checkbox
                                        checked={isSelected}
                                        onChange={() => toggleJobSelection(job.id)}
                                        size="small"
                                        sx={{
                                            color: alpha(accentColors.success, 0.5),
                                            "&.Mui-checked": { color: accentColors.success },
                                            p: 0.5,
                                            mt: 0.5,
                                        }}
                                    />
                                )}

                                {/* Model info */}
                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                    <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5, flexWrap: "wrap" }}>
                                        <ModelIcon sx={{ fontSize: 16, color: accentColors.info }} />
                                        <Typography
                                            variant="subtitle2"
                                            sx={{
                                                fontWeight: 700,
                                                fontSize: "0.9rem",
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                                whiteSpace: "nowrap",
                                                maxWidth: { xs: 150, sm: 250, md: 400 },
                                            }}
                                            title={modelName}
                                        >
                                            {modelName}
                                        </Typography>
                                        <StatusChip status={job.status} />
                                    </Box>

                                    {/* Metadata row */}
                                    <Box sx={{ display: "flex", alignItems: "center", gap: 2, flexWrap: "wrap" }}>
                                        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                            <DateIcon sx={{ fontSize: 12, color: "text.secondary" }} />
                                            <Typography variant="caption" color="text.secondary">
                                                {new Date(job.created_at).toLocaleString()}
                                            </Typography>
                                        </Box>
                                        {job.gpu_device !== undefined && (
                                            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                                <GpuIcon sx={{ fontSize: 12, color: "text.secondary" }} />
                                                <Typography variant="caption" color="text.secondary">
                                                    GPU {job.gpu_device}
                                                </Typography>
                                            </Box>
                                        )}
                                        <Typography variant="caption" color="text.secondary">
                                            Tasks: {job.tasks.join(", ")}
                                        </Typography>
                                    </Box>
                                </Box>

                                {/* Actions */}
                                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                    {job.status === "completed" && (
                                        <IconButton
                                            size="small"
                                            onClick={() => setExpandedJobId(isExpanded ? null : job.id)}
                                            sx={{ color: "text.secondary" }}
                                        >
                                            {isExpanded ? <CollapseIcon /> : <ExpandIcon />}
                                        </IconButton>
                                    )}
                                    {onDeleteJob && (
                                        <IconButton
                                            size="small"
                                            onClick={() => onDeleteJob(job.id)}
                                            sx={{ color: "text.secondary", "&:hover": { color: accentColors.rose } }}
                                        >
                                            <DeleteIcon sx={{ fontSize: 18 }} />
                                        </IconButton>
                                    )}
                                </Box>
                            </Box>

                            {/* Progress bar for running jobs */}
                            {job.status === "running" && (
                                <LinearProgress
                                    variant="determinate"
                                    value={job.progress}
                                    sx={{
                                        mt: 1.5,
                                        height: 4,
                                        borderRadius: 2,
                                        bgcolor: "rgba(255, 255, 255, 0.1)",
                                        "& .MuiLinearProgress-bar": {
                                            background: `linear-gradient(90deg, ${accentColors.info}, ${accentColors.success})`,
                                        },
                                    }}
                                />
                            )}

                            {/* Error display */}
                            {job.error && (
                                <Alert severity="error" sx={{ mt: 1.5 }}>
                                    {job.error}
                                </Alert>
                            )}

                            {/* Results scores (condensed) */}
                            {job.status === "completed" && job.results?.tasks && !isExpanded && (
                                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mt: 1.5 }}>
                                    {Object.entries(job.results.tasks).map(([taskName, taskResult]) => (
                                        <Chip
                                            key={taskName}
                                            size="small"
                                            label={
                                                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                                    <span style={{ fontWeight: 500 }}>{taskName}:</span>
                                                    <span style={{ fontWeight: 700, color: accentColors.success }}>
                                                        {taskResult.score.toFixed(1)}%
                                                    </span>
                                                </Box>
                                            }
                                            sx={{
                                                bgcolor: "rgba(255, 255, 255, 0.05)",
                                                border: "1px solid rgba(255, 255, 255, 0.1)",
                                                height: 28,
                                            }}
                                        />
                                    ))}
                                </Box>
                            )}

                            {/* Expanded details */}
                            <Collapse in={isExpanded}>
                                <Divider sx={{ my: 2, borderColor: "rgba(255, 255, 255, 0.06)" }} />
                                <Box sx={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 2 }}>
                                    {Object.entries(job.results?.tasks || {}).map(([taskName, taskResult]) => (
                                        <Box
                                            key={taskName}
                                            sx={{
                                                p: 1.5,
                                                bgcolor: "rgba(0, 0, 0, 0.2)",
                                                borderRadius: 1.5,
                                                border: "1px solid rgba(255, 255, 255, 0.06)",
                                            }}
                                        >
                                            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                                {taskName}
                                            </Typography>
                                            <Typography
                                                variant="h5"
                                                sx={{ fontWeight: 700, color: accentColors.success, mb: 1 }}
                                            >
                                                {taskResult.score.toFixed(1)}%
                                            </Typography>
                                            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.25 }}>
                                                {Object.entries(taskResult.metrics || {}).map(([metricKey, metricValue]) => (
                                                    <Box key={metricKey} sx={{ display: "flex", justifyContent: "space-between" }}>
                                                        <Typography variant="caption" color="text.secondary">
                                                            {metricKey}:
                                                        </Typography>
                                                        <Typography variant="caption" sx={{ fontWeight: 600 }}>
                                                            {typeof metricValue === "number" ? metricValue.toFixed(2) : metricValue}%
                                                        </Typography>
                                                    </Box>
                                                ))}
                                            </Box>
                                        </Box>
                                    ))}
                                </Box>
                            </Collapse>
                        </Box>
                    );
                })}
            </Box>
        </Box>
    );
};

export default BenchmarkResultsPanel;
