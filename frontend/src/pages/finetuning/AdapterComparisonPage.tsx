import React, { useEffect, useState, useMemo, useRef } from "react";
import {
  Box,
  Typography,
  IconButton,
  Button,
  Grid,
  Alert,
  LinearProgress,
  Checkbox,
  Tooltip,
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
  Star as BestIcon,
} from "@mui/icons-material";
import { useLocation, useNavigate } from "react-router-dom";
import { getRoutePath } from "@/config/navigation";
import type { AdapterComparisonRouteState, AdapterHandoff } from "@/types/trainingHandoff";
import { adapterHandoffsKey, findAdapterForHandoff } from "@/types/trainingHandoff";
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from "@/components/common";
import { accentColors } from "@/utils/theme";

// Types
type Adapter = {
  id: string;
  name: string;
  base_model: string;
  status: string;
  training_job_id?: string;
  job_id?: string;
  adapter_path?: string;
  path?: string;
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

const ADAPTER_COMPARISON_STATUS_CONFIG: StatusChipConfigMap = {
  compared: { color: accentColors.success, label: "Compared" },
  comparing: { color: accentColors.info, label: "Comparing" },
  comparison: { color: accentColors.purple, label: "Adapter comparison" },
  ready: { color: accentColors.success, label: "Ready" },
  selecting: { color: accentColors.warning, label: "Select adapters" },
  selected: { color: accentColors.success, label: "Selected" },
  setup: { color: accentColors.info, label: "Setup" },
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
              borderRadius: "8px",
              bgcolor: isBest ? alpha(accentColors.success, 0.1) : "background.paper",
              border: "1px solid",
              borderColor: isBest ? alpha(accentColors.success, 0.3) : "divider",
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
                          bgcolor: "action.hover",
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
                  borderRadius: "8px",
                  bgcolor: "background.paper",
                  border: "1px solid",
                  borderColor: "divider",
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
                            bgcolor: "action.hover",
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
  const location = useLocation();
  const trainingPath = getRoutePath("finetuning");
  const routeState = location.state as AdapterComparisonRouteState | null;
  const incomingAdapters = useMemo(() => {
    const adaptersFromState = routeState?.adapters ?? [];
    return [...(routeState?.adapter ? [routeState.adapter] : []), ...adaptersFromState].filter(
      (adapter): adapter is AdapterHandoff => Boolean(adapter)
    );
  }, [routeState]);
  const incomingAdapterKey = useMemo(
    () => adapterHandoffsKey(incomingAdapters),
    [incomingAdapters]
  );
  const appliedHandoffRef = useRef("");
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [handoffNotice, setHandoffNotice] = useState<string | null>(null);

  const selectedAdapters = useMemo(
    () => adapters.filter((a) => selectedIds.includes(a.id)),
    [adapters, selectedIds]
  );

  useEffect(() => {
    fetchAdapters();
  }, []);

  useEffect(() => {
    if (incomingAdapters.length === 0 || adapters.length === 0 || !incomingAdapterKey) return;
    if (appliedHandoffRef.current === incomingAdapterKey) return;

    const matchedAdapters = incomingAdapters
      .map((handoff) => findAdapterForHandoff(handoff, adapters))
      .filter((adapter): adapter is Adapter => Boolean(adapter));
    const matchedIds = Array.from(new Set(matchedAdapters.map((adapter) => adapter.id))).slice(0, 4);

    appliedHandoffRef.current = incomingAdapterKey;
    if (matchedIds.length === 0) {
      setHandoffNotice("The incoming adapter is not in the comparison registry yet");
      return;
    }

    setSelectedIds((current) => Array.from(new Set([...current, ...matchedIds])).slice(0, 4));
    setHandoffNotice(
      matchedIds.length === 1
        ? `${matchedAdapters[0].name} added to the comparison set`
        : `${matchedIds.length} adapters added to the comparison set`
    );
  }, [adapters, incomingAdapterKey, incomingAdapters]);

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

  const readyAdapters = adapters.filter((adapter) => adapter.status === "ready").length;
  const adaptersWithMetrics = adapters.filter((adapter) => Object.keys(adapter.metrics || {}).length > 0).length;
  const adaptersWithBenchmarks = adapters.filter((adapter) => Object.keys(adapter.benchmark_results || {}).length > 0).length;
  const baseModelCount = new Set(adapters.map((adapter) => adapter.base_model).filter(Boolean)).size;
  const hasAdapters = adapters.length > 0;
  const selectionLabel = selectedAdapters.length > 0
    ? selectedAdapters.map((adapter) => adapter.name).join(", ")
    : "No adapters selected";
  const comparisonStatus = loading
    ? "comparing"
    : comparison
      ? "compared"
      : selectedIds.length >= 2
        ? "ready"
        : "selecting";
  const showComparisonPane = loading || Boolean(comparison) || selectedIds.length > 0;
  const showComparisonSummary = hasAdapters || showComparisonPane;
  const stretchSelectionPane = hasAdapters || showComparisonPane;
  const selectionGridMd = showComparisonPane ? 4 : 12;
  const selectionGridLg = showComparisonPane ? 4 : 12;
  const selectionSummary = selectedIds.length > 0
    ? selectionLabel
    : `${baseModelCount} base model${baseModelCount === 1 ? "" : "s"}`;
  const commandIconButtonSx = {
    width: 34,
    height: 34,
    borderRadius: "8px",
    border: "1px solid",
    borderColor: "divider",
    bgcolor: "action.hover",
    "&:hover": { bgcolor: "action.selected" },
    "&.Mui-disabled": {
      bgcolor: "action.disabledBackground",
      color: "action.disabled",
    },
  };
  const comparisonHeaderActions = (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexWrap: "wrap", justifyContent: "flex-end" }}>
      <Tooltip title="Back to training">
        <IconButton
          size="small"
          aria-label="Back to training"
          onClick={() => navigate(trainingPath)}
          sx={commandIconButtonSx}
        >
          <BackIcon sx={{ fontSize: 17 }} />
        </IconButton>
      </Tooltip>
      {adapters.length > 0 && (
        <Tooltip title="Compare selected adapters">
          <span>
            <IconButton
              size="small"
              aria-label="Compare selected adapters"
              color="primary"
              onClick={runComparison}
              disabled={selectedIds.length < 2 || loading}
              sx={{
                ...commandIconButtonSx,
                bgcolor: "primary.main",
                color: "primary.contrastText",
                "&:hover": { bgcolor: "primary.dark" },
                "&.Mui-disabled": {
                  bgcolor: "action.disabledBackground",
                  color: "action.disabled",
                },
              }}
            >
              <CompareIcon sx={{ fontSize: 17 }} />
            </IconButton>
          </span>
        </Tooltip>
      )}
      <Tooltip title="Refresh adapters">
        <IconButton
          size="small"
          aria-label="Refresh adapters"
          onClick={fetchAdapters}
          sx={commandIconButtonSx}
        >
          <RefreshIcon sx={{ fontSize: 17 }} />
        </IconButton>
      </Tooltip>
    </Box>
  );
  const comparisonSummaryRow = (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap", mb: 1 }}>
      <StatusChip status={comparisonStatus} statusConfig={ADAPTER_COMPARISON_STATUS_CONFIG} height={22} />
      <StatusChip
        status="comparison"
        label={`${adapters.length} adapter${adapters.length === 1 ? "" : "s"}`}
        statusConfig={ADAPTER_COMPARISON_STATUS_CONFIG}
        height={22}
      />
      {adapters.length > 0 && (
        <StatusChip
          status="selected"
          label={`${selectedIds.length}/4 selected`}
          statusConfig={ADAPTER_COMPARISON_STATUS_CONFIG}
          height={22}
        />
      )}
      {readyAdapters > 0 && (
        <StatusChip
          status="ready"
          label={`${readyAdapters} ready`}
          statusConfig={ADAPTER_COMPARISON_STATUS_CONFIG}
          height={22}
        />
      )}
      {(adaptersWithMetrics > 0 || adaptersWithBenchmarks > 0) && (
        <StatusChip
          status="setup"
          label={`${adaptersWithMetrics} metrics / ${adaptersWithBenchmarks} benchmarks`}
          statusConfig={ADAPTER_COMPARISON_STATUS_CONFIG}
          height={22}
        />
      )}
      {adapters.length > 0 && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        >
          {selectionSummary}
        </Typography>
      )}
    </Box>
  );

  return (
    <PageShell sx={{ py: 1.25 }}>
      {handoffNotice && (
        <Alert severity="info" sx={{ mb: 1.25 }} onClose={() => setHandoffNotice(null)}>
          {handoffNotice}
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={2} sx={{ minHeight: { md: stretchSelectionPane ? "calc(100dvh - 82px)" : 0 } }}>
        {/* Adapter Selection */}
        <Grid item xs={12} md={selectionGridMd} lg={selectionGridLg} sx={{ display: "flex", minHeight: 0 }}>
          <SectionCard
            title="Select Adapters"
            subtitle={adapters.length > 0 ? `${selectedIds.length}/4 selected` : undefined}
            icon={<AdapterIcon />}
            accentColor={accentColors.purple}
            action={comparisonHeaderActions}
            fullHeight={stretchSelectionPane}
            sx={{
              flex: 1,
              ...(stretchSelectionPane ? {
                minHeight: 0,
                display: "flex",
                flexDirection: "column",
                "& .MuiCardContent-root": {
                  flex: 1,
                  minHeight: 0,
                  display: "flex",
                  flexDirection: "column",
                },
                "& .MuiCardContent-root > .MuiBox-root:last-child": {
                  flex: 1,
                  minHeight: 0,
                  display: "flex",
                  flexDirection: "column",
                },
              } : {}),
            }}
          >
            {showComparisonSummary && comparisonSummaryRow}
            {adapters.length === 0 ? (
              <Box
                sx={{
                  border: "1px dashed",
                  borderColor: "divider",
                  borderRadius: "8px",
                  bgcolor: "action.hover",
                  display: "grid",
                  gridTemplateColumns: { xs: "1fr", sm: "minmax(0, 1fr) auto" },
                  alignItems: "center",
                  gap: 1.25,
                  px: 1.5,
                  py: 1.35,
                }}
              >
                <Box sx={{ minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: 800, mb: 0.5 }}>No adapters available</Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                    Create one from Fine-Tuning to compare training metrics and benchmark results.
                  </Typography>
                </Box>
                <Button
                  size="small"
                  variant="contained"
                  startIcon={<AdapterIcon sx={{ fontSize: 16 }} />}
                  onClick={() => navigate(trainingPath)}
                  sx={{
                    height: 32,
                    borderRadius: "8px",
                    fontWeight: 800,
                    boxShadow: "none",
                    justifySelf: { xs: "flex-start", sm: "flex-end" },
                    "&:hover": { boxShadow: "none" },
                  }}
                >
                  Create adapter
                </Button>
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
                        borderRadius: "8px",
                        bgcolor: isSelected ? alpha(accentColors.success, 0.1) : "background.paper",
                        border: "1px solid",
                        borderColor: isSelected ? accentColors.success : "divider",
                        cursor: "pointer",
                        transition: "all 0.2s",
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5,
                        "&:hover": {
                          bgcolor: alpha(accentColors.primary, 0.08),
                          borderColor: isSelected ? accentColors.success : alpha(accentColors.primary, 0.3),
                        },
                      }}
                    >
                      <Checkbox
                        checked={isSelected}
                        size="small"
                        sx={{
                          p: 0,
                          color: "text.disabled",
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
                          <StatusChip status={adapter.status} height={22} />
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
        {showComparisonPane && (
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
                      borderRadius: "8px",
                      bgcolor: alpha(
                        [accentColors.info, accentColors.purple, accentColors.success, accentColors.warning][idx],
                        0.1
                      ),
                      border: "1px solid",
                      borderColor: [accentColors.info, accentColors.purple, accentColors.success, accentColors.warning][idx],
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
              title={selectedIds.length >= 2 ? "Ready to compare" : "Select one more adapter"}
              icon={<CompareIcon />}
              accentColor={accentColors.info}
            >
              <Box sx={{ py: 1.25 }}>
                <Typography variant="body2" color="text.secondary">
                  {selectedIds.length >= 2
                    ? "Run the comparison to inspect configuration, metrics, and benchmark differences."
                    : "Choose at least two adapters to compare configuration, metrics, and benchmark differences."}
                </Typography>
              </Box>
            </SectionCard>
          )}
        </Grid>
        )}
      </Grid>
    </PageShell>
  );
};

export default AdapterComparisonPage;
