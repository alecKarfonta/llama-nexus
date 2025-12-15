import React, { useEffect, useState, useMemo } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  Grid,
  Alert,
  Tooltip,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Divider,
  Checkbox,
  alpha,
} from "@mui/material";
import {
  CompareArrows as CompareIcon,
  AutoFixHigh as AdapterIcon,
  Settings as ConfigIcon,
  Assessment as MetricsIcon,
  Science as BenchmarkIcon,
  ArrowBack as BackIcon,
  Refresh as RefreshIcon,
  TrendingUp as UpIcon,
  TrendingDown as DownIcon,
  Remove as NeutralIcon,
  Star as BestIcon,
  CheckCircle as SelectedIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

// Types
type Adapter = {
  id: string;
  name: string;
  base_model: string;
  status: string;
  training_job_id?: string;
  lora_config: Record<string, any>;
  training_config: Record<string, any>;
  metrics: Record<string, any>;
  benchmark_results: Record<string, any>;
  created_at: string;
  tags: string[];
};

type ComparisonResult = {
  adapter_ids: string[];
  comparison_date: string;
  config_diffs: {
    lora_config: Record<string, Record<string, any>>;
    training_config: Record<string, Record<string, any>>;
    base_model: Record<string, string>;
  };
  metric_comparisons: Record<string, Record<string, any>>;
  benchmark_comparisons: Record<string, Record<string, any>>;
  summary: Record<string, any>;
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
    ready: { color: accentColors.success, label: "Ready" },
    training: { color: accentColors.info, label: "Training" },
    merged: { color: accentColors.purple, label: "Merged" },
    exported: { color: accentColors.warning, label: "Exported" },
    archived: { color: "#64748b", label: "Archived" },
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
        height: 22,
      }}
    />
  );
};

// Comparison Cell - shows value differences
interface ComparisonCellProps {
  values: Record<string, any>;
  adapters: Adapter[];
  format?: (val: any) => string;
  highlight?: "higher" | "lower" | "none";
}

const ComparisonCell: React.FC<ComparisonCellProps> = ({
  values,
  adapters,
  format = (v) => String(v ?? "-"),
  highlight = "none",
}) => {
  const numericValues = Object.values(values).filter((v) => typeof v === "number") as number[];
  const max = Math.max(...numericValues);
  const min = Math.min(...numericValues);

  return (
    <Box sx={{ display: "flex", gap: 2 }}>
      {adapters.map((adapter) => {
        const value = values[adapter.id];
        const isMax = value === max && numericValues.length > 1;
        const isMin = value === min && numericValues.length > 1;
        const isBest = (highlight === "higher" && isMax) || (highlight === "lower" && isMin);

        return (
          <Box
            key={adapter.id}
            sx={{
              flex: 1,
              p: 1,
              borderRadius: 1,
              bgcolor: isBest ? alpha(accentColors.success, 0.1) : "rgba(255, 255, 255, 0.02)",
              border: `1px solid ${isBest ? alpha(accentColors.success, 0.3) : "rgba(255, 255, 255, 0.06)"}`,
              textAlign: "center",
            }}
          >
            <Typography
              sx={{
                fontWeight: 600,
                fontFamily: "monospace",
                fontSize: "0.85rem",
                color: isBest ? accentColors.success : "text.primary",
              }}
            >
              {format(value)}
            </Typography>
            {isBest && (
              <BestIcon sx={{ fontSize: 12, color: accentColors.success, ml: 0.5 }} />
            )}
          </Box>
        );
      })}
    </Box>
  );
};

// Benchmark Chart Component
interface BenchmarkChartProps {
  benchmarks: Record<string, Record<string, any>>;
  adapters: Adapter[];
}

const BenchmarkChart: React.FC<BenchmarkChartProps> = ({ benchmarks, adapters }) => {
  if (!benchmarks || Object.keys(benchmarks).length === 0) {
    return (
      <Box sx={{ py: 4, textAlign: "center" }}>
        <BenchmarkIcon sx={{ fontSize: 40, color: "text.secondary", opacity: 0.3, mb: 1 }} />
        <Typography color="text.secondary">No benchmark results available</Typography>
      </Box>
    );
  }

  const colors = [accentColors.info, accentColors.purple, accentColors.success, accentColors.warning];

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {Object.entries(benchmarks).map(([benchmarkName, scores]) => {
        const numericScores = Object.entries(scores)
          .filter(([, v]) => typeof v === "number")
          .map(([id, v]) => ({ id, score: v as number }));
        const maxScore = Math.max(...numericScores.map((s) => s.score), 1);

        return (
          <Box key={benchmarkName}>
            <Typography variant="caption" sx={{ fontWeight: 600, textTransform: "uppercase", mb: 1, display: "block" }}>
              {benchmarkName.replace(/_/g, " ")}
            </Typography>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
              {adapters.map((adapter, idx) => {
                const score = scores[adapter.id];
                const percentage = typeof score === "number" ? (score / maxScore) * 100 : 0;
                const isMax = score === maxScore && numericScores.length > 1;

                return (
                  <Box key={adapter.id} sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <Typography
                      variant="caption"
                      sx={{ width: 80, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    >
                      {adapter.name}
                    </Typography>
                    <Box sx={{ flex: 1, display: "flex", alignItems: "center", gap: 1 }}>
                      <Box
                        sx={{
                          flex: 1,
                          height: 8,
                          bgcolor: "rgba(255, 255, 255, 0.1)",
                          borderRadius: 4,
                          overflow: "hidden",
                        }}
                      >
                        <Box
                          sx={{
                            width: `${percentage}%`,
                            height: "100%",
                            bgcolor: colors[idx % colors.length],
                            borderRadius: 4,
                            transition: "width 0.5s ease-out",
                          }}
                        />
                      </Box>
                      <Typography
                        variant="caption"
                        sx={{
                          width: 50,
                          textAlign: "right",
                          fontWeight: isMax ? 700 : 400,
                          color: isMax ? accentColors.success : "text.secondary",
                          fontFamily: "monospace",
                        }}
                      >
                        {typeof score === "number" ? score.toFixed(1) : "-"}%
                      </Typography>
                    </Box>
                  </Box>
                );
              })}
            </Box>
          </Box>
        );
      })}
    </Box>
  );
};

// Training Metrics Overlay Chart
interface MetricsOverlayProps {
  adapters: Adapter[];
}

const MetricsOverlay: React.FC<MetricsOverlayProps> = ({ adapters }) => {
  const colors = [accentColors.info, accentColors.purple, accentColors.success, accentColors.warning];
  
  // Extract all unique metric keys
  const allMetrics = useMemo(() => {
    const keys = new Set<string>();
    adapters.forEach((a) => {
      Object.keys(a.metrics || {}).forEach((k) => keys.add(k));
    });
    return Array.from(keys);
  }, [adapters]);

  if (allMetrics.length === 0) {
    return (
      <Box sx={{ py: 4, textAlign: "center" }}>
        <MetricsIcon sx={{ fontSize: 40, color: "text.secondary", opacity: 0.3, mb: 1 }} />
        <Typography color="text.secondary">No training metrics available</Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Legend */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1.5, mb: 2 }}>
        {adapters.map((adapter, idx) => (
          <Box key={adapter.id} sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <Box
              sx={{
                width: 12,
                height: 12,
                borderRadius: 1,
                bgcolor: colors[idx % colors.length],
              }}
            />
            <Typography variant="caption" sx={{ fontWeight: 500 }}>
              {adapter.name}
            </Typography>
          </Box>
        ))}
      </Box>

      {/* Metrics Grid */}
      <Grid container spacing={2}>
        {allMetrics.slice(0, 8).map((metric) => {
          const values = adapters.map((a) => a.metrics?.[metric]);
          const numericValues = values.filter((v) => typeof v === "number") as number[];
          const max = Math.max(...numericValues, 1);
          const isLowerBetter = metric.toLowerCase().includes("loss");

          return (
            <Grid item xs={6} md={3} key={metric}>
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: "rgba(0, 0, 0, 0.2)",
                  border: "1px solid rgba(255, 255, 255, 0.06)",
                }}
              >
                <Typography
                  variant="caption"
                  sx={{ color: "text.secondary", textTransform: "capitalize", display: "block", mb: 1 }}
                >
                  {metric.replace(/_/g, " ")}
                </Typography>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                  {adapters.map((adapter, idx) => {
                    const value = adapter.metrics?.[metric];
                    const percentage = typeof value === "number" ? (value / max) * 100 : 0;
                    const isBest =
                      typeof value === "number" &&
                      ((isLowerBetter && value === Math.min(...numericValues)) ||
                        (!isLowerBetter && value === Math.max(...numericValues)));

                    return (
                      <Box key={adapter.id} sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <LinearProgress
                          variant="determinate"
                          value={percentage}
                          sx={{
                            flex: 1,
                            height: 6,
                            borderRadius: 3,
                            bgcolor: "rgba(255, 255, 255, 0.1)",
                            "& .MuiLinearProgress-bar": {
                              bgcolor: colors[idx % colors.length],
                              borderRadius: 3,
                            },
                          }}
                        />
                        <Typography
                          variant="caption"
                          sx={{
                            width: 45,
                            textAlign: "right",
                            fontFamily: "monospace",
                            fontSize: "0.65rem",
                            color: isBest ? accentColors.success : "text.secondary",
                            fontWeight: isBest ? 700 : 400,
                          }}
                        >
                          {typeof value === "number" ? value.toFixed(3) : "-"}
                        </Typography>
                      </Box>
                    );
                  })}
                </Box>
              </Box>
            </Grid>
          );
        })}
      </Grid>
    </Box>
  );
};

export const AdapterComparisonPage: React.FC = () => {
  const navigate = useNavigate();
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedAdapters = useMemo(
    () => adapters.filter((a) => selectedIds.includes(a.id)),
    [adapters, selectedIds]
  );

  useEffect(() => {
    fetchAdapters();
  }, []);

  const fetchAdapters = async () => {
    try {
      const res = await fetch("/api/v1/finetune/adapters/registry");
      if (res.ok) {
        const data = await res.json();
        setAdapters(data.adapters || []);
      }
    } catch {
      setError("Failed to load adapters");
    }
  };

  const toggleAdapter = (id: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((i) => i !== id);
      }
      if (prev.length >= 4) {
        return prev; // Max 4 adapters for comparison
      }
      return [...prev, id];
    });
  };

  const runComparison = async () => {
    if (selectedIds.length < 2) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/v1/finetune/adapters/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selectedIds),
      });
      if (res.ok) {
        const data = await res.json();
        setComparison(data);
      } else {
        const err = await res.json();
        setError(err.detail || "Comparison failed");
      }
    } catch {
      setError("Failed to compare adapters");
    } finally {
      setLoading(false);
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
          alignItems: "center",
          justifyContent: "space-between",
          mb: 3,
        }}
      >
        <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
          <IconButton
            onClick={() => navigate("/finetune")}
            sx={{
              bgcolor: "rgba(255, 255, 255, 0.05)",
              "&:hover": { bgcolor: "rgba(255, 255, 255, 0.1)" },
            }}
          >
            <BackIcon />
          </IconButton>
          <Box>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              Adapter Comparison
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Compare configurations, metrics, and benchmark results
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: "flex", gap: 1 }}>
          <Tooltip title="Refresh">
            <IconButton onClick={fetchAdapters}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Adapter Selection */}
        <Grid item xs={12} md={4}>
          <SectionCard
            title="Select Adapters"
            subtitle={`${selectedIds.length}/4 selected`}
            icon={<AdapterIcon />}
            accentColor={accentColors.purple}
            action={
              <Button
                variant="contained"
                size="small"
                startIcon={<CompareIcon />}
                onClick={runComparison}
                disabled={selectedIds.length < 2 || loading}
                sx={{
                  background: `linear-gradient(135deg, ${accentColors.info} 0%, ${accentColors.purple} 100%)`,
                }}
              >
                Compare
              </Button>
            }
          >
            {adapters.length === 0 ? (
              <Box sx={{ py: 4, textAlign: "center" }}>
                <AdapterIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                <Typography color="text.secondary">No registered adapters found</Typography>
              </Box>
            ) : (
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                {adapters.map((adapter) => {
                  const isSelected = selectedIds.includes(adapter.id);
                  return (
                    <Box
                      key={adapter.id}
                      onClick={() => toggleAdapter(adapter.id)}
                      sx={{
                        p: 1.5,
                        borderRadius: 2,
                        bgcolor: isSelected ? alpha(accentColors.success, 0.1) : "rgba(255, 255, 255, 0.02)",
                        border: `1px solid ${isSelected ? accentColors.success : "rgba(255, 255, 255, 0.06)"}`,
                        cursor: "pointer",
                        transition: "all 0.2s",
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5,
                        "&:hover": {
                          bgcolor: alpha(accentColors.primary, 0.08),
                        },
                      }}
                    >
                      <Checkbox
                        checked={isSelected}
                        size="small"
                        sx={{
                          p: 0,
                          color: "rgba(255, 255, 255, 0.3)",
                          "&.Mui-checked": { color: accentColors.success },
                        }}
                      />
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
                          <Typography
                            variant="subtitle2"
                            sx={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis" }}
                          >
                            {adapter.name}
                          </Typography>
                          <StatusChip status={adapter.status} />
                        </Box>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: "0.7rem" }}>
                          {adapter.base_model.split("/").pop()}
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            )}
          </SectionCard>
        </Grid>

        {/* Comparison Results */}
        <Grid item xs={12} md={8}>
          {loading ? (
            <SectionCard title="Comparing..." icon={<CompareIcon />} accentColor={accentColors.info}>
              <LinearProgress />
            </SectionCard>
          ) : comparison && selectedAdapters.length >= 2 ? (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {/* Selected Adapters Header */}
              <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
                {selectedAdapters.map((adapter, idx) => (
                  <Box
                    key={adapter.id}
                    sx={{
                      flex: 1,
                      minWidth: 150,
                      p: 1.5,
                      borderRadius: 2,
                      bgcolor: alpha(
                        [accentColors.info, accentColors.purple, accentColors.success, accentColors.warning][idx],
                        0.1
                      ),
                      border: `2px solid ${[accentColors.info, accentColors.purple, accentColors.success, accentColors.warning][idx]}`,
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                      {adapter.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {adapter.base_model.split("/").pop()}
                    </Typography>
                  </Box>
                ))}
              </Box>

              {/* Config Differences */}
              <SectionCard
                title="Configuration Differences"
                subtitle="LoRA and Training parameters that differ"
                icon={<ConfigIcon />}
                accentColor={accentColors.info}
              >
                {Object.keys(comparison.config_diffs.lora_config).length === 0 &&
                Object.keys(comparison.config_diffs.training_config).length === 0 ? (
                  <Alert severity="info" sx={{ bgcolor: alpha(accentColors.info, 0.1) }}>
                    All configurations are identical
                  </Alert>
                ) : (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                    {/* LoRA Config */}
                    {Object.keys(comparison.config_diffs.lora_config).length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5, color: accentColors.info }}>
                          LoRA Configuration
                        </Typography>
                        {Object.entries(comparison.config_diffs.lora_config).map(([key, values]) => (
                          <Box key={key} sx={{ mb: 1.5 }}>
                            <Typography variant="caption" sx={{ color: "text.secondary", textTransform: "capitalize" }}>
                              {key.replace(/_/g, " ")}
                            </Typography>
                            <ComparisonCell
                              values={values}
                              adapters={selectedAdapters}
                              highlight={key === "rank" || key === "alpha" ? "higher" : "none"}
                            />
                          </Box>
                        ))}
                      </Box>
                    )}

                    {/* Training Config */}
                    {Object.keys(comparison.config_diffs.training_config).length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5, color: accentColors.purple }}>
                          Training Configuration
                        </Typography>
                        {Object.entries(comparison.config_diffs.training_config).map(([key, values]) => (
                          <Box key={key} sx={{ mb: 1.5 }}>
                            <Typography variant="caption" sx={{ color: "text.secondary", textTransform: "capitalize" }}>
                              {key.replace(/_/g, " ")}
                            </Typography>
                            <ComparisonCell
                              values={values}
                              adapters={selectedAdapters}
                              format={(v) => (typeof v === "number" && v < 0.01 ? v.toExponential(1) : String(v ?? "-"))}
                            />
                          </Box>
                        ))}
                      </Box>
                    )}
                  </Box>
                )}
              </SectionCard>

              {/* Training Metrics */}
              <SectionCard
                title="Training Metrics Comparison"
                subtitle="Side-by-side metric visualization"
                icon={<MetricsIcon />}
                accentColor={accentColors.success}
              >
                <MetricsOverlay adapters={selectedAdapters} />
              </SectionCard>

              {/* Benchmark Results */}
              <SectionCard
                title="Benchmark Comparison"
                subtitle="Performance across evaluation benchmarks"
                icon={<BenchmarkIcon />}
                accentColor={accentColors.warning}
              >
                <BenchmarkChart benchmarks={comparison.benchmark_comparisons} adapters={selectedAdapters} />
              </SectionCard>
            </Box>
          ) : (
            <SectionCard
              title="Comparison Results"
              subtitle="Select at least 2 adapters to compare"
              icon={<CompareIcon />}
              accentColor={accentColors.info}
            >
              <Box sx={{ py: 6, textAlign: "center" }}>
                <CompareIcon sx={{ fontSize: 64, color: "text.secondary", opacity: 0.2, mb: 2 }} />
                <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                  Select Adapters to Compare
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Choose 2-4 adapters from the list to see detailed comparisons of their configurations, training
                  metrics, and benchmark results.
                </Typography>
              </Box>
            </SectionCard>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default AdapterComparisonPage;
