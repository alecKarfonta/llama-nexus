import React, { useEffect, useState } from "react";
import {
  Box,
  Typography,
  TextField,
  IconButton,
  Alert,
  Tooltip,
  LinearProgress,
  Divider,
  alpha,
  useMediaQuery,
} from "@mui/material";
import {
  RocketLaunch as RocketIcon,
  ArrowBack as BackIcon,
  Refresh as RefreshIcon,
  Schedule as TimeIcon,
  Storage as DataIcon,
  School as LearnIcon,
  Star as StarIcon,
  Code as CodeIcon,
  Psychology as AIIcon,
  Lightbulb as TipIcon,
  PlayArrow as StartIcon,
  Edit as EditIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { getRoutePath } from "@/config/navigation";
import { PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from "@/components/common";
import { accentColors } from "@/utils/theme";

// Types
type WorkflowTemplate = {
  id: string;
  name: string;
  description: string;
  use_case: string;
  difficulty: string;
  sample_prompts: string[];
  prompt_guidelines: string;
  recommended_base_models: string[];
  training_notes: string;
  expected_quality: string;
  typical_dataset_size: string;
  estimated_time: string;
  distillation_config: {
    teacher_provider: string;
    teacher_model: string;
    strategy: string;
    target_examples: number;
  };
};

// Difficulty config
const difficultyConfig: Record<string, { color: string; label: string; icon: React.ReactNode }> = {
  beginner: { color: accentColors.success, label: "Beginner", icon: <StarIcon /> },
  intermediate: { color: accentColors.warning, label: "Intermediate", icon: <LearnIcon /> },
  advanced: { color: accentColors.rose, label: "Advanced", icon: <CodeIcon /> },
};

const WORKFLOW_TEMPLATE_STATUS_CONFIG: StatusChipConfigMap = {
  advanced: { color: accentColors.rose, label: "Advanced" },
  beginner: { color: accentColors.success, label: "Beginner" },
  browsing: { color: accentColors.warning, label: "Choose template" },
  custom: { color: accentColors.primary, label: "Custom prompts" },
  intermediate: { color: accentColors.warning, label: "Intermediate" },
  quickstart: { color: accentColors.success, label: "Quick start" },
  selected: { color: accentColors.info, label: "Template selected" },
  workflow: { color: accentColors.purple, label: "One-click workflow" },
};

const workflowTemplatePanelSx = {
  borderRadius: "8px",
  bgcolor: "background.paper",
  border: "1px solid",
  borderColor: "divider",
} as const;

const workflowTemplateActionButtonSx = {
  width: 34,
  height: 34,
  borderRadius: "8px",
  border: "1px solid",
  borderColor: "divider",
  bgcolor: "action.hover",
  "&:hover": {
    bgcolor: "action.selected",
  },
} as const;

// Template Card Component
interface TemplateCardProps {
  template: WorkflowTemplate;
  selected: boolean;
  loading?: boolean;
  onClick: () => void;
  onStart: () => void;
}

const TemplateCard: React.FC<TemplateCardProps> = ({ template, selected, loading = false, onClick, onStart }) => {
  const difficulty = difficultyConfig[template.difficulty] || difficultyConfig.beginner;

  return (
    <Box
      onClick={onClick}
      sx={{
        p: 1,
        borderRadius: "8px",
        bgcolor: selected ? alpha(accentColors.success, 0.1) : "background.paper",
        border: "1px solid",
        borderColor: selected ? accentColors.success : "divider",
        cursor: "pointer",
        transition: "background-color 0.18s ease, border-color 0.18s ease",
        "&:hover": {
          bgcolor: alpha(accentColors.primary, 0.08),
          borderColor: selected ? accentColors.success : alpha(accentColors.primary, 0.3),
        },
      }}
    >
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 1, mb: 0.5 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, minWidth: 0 }} noWrap>
          {template.name}
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, flexShrink: 0 }}>
          <Tooltip title={loading ? "Starting workflow" : "Start template"}>
            <span>
              <IconButton
                size="small"
                aria-label={`Start ${template.name}`}
                disabled={loading}
                onClick={(event) => {
                  event.stopPropagation();
                  onStart();
                }}
                sx={{
                  width: 28,
                  height: 28,
                  borderRadius: "7px",
                  color: selected ? accentColors.success : "text.secondary",
                  bgcolor: selected ? alpha(accentColors.success, 0.12) : "action.hover",
                  "&:hover": {
                    bgcolor: alpha(accentColors.success, 0.18),
                  },
                }}
              >
                <RocketIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </span>
          </Tooltip>
          <StatusChip
            status={template.difficulty}
            statusConfig={WORKFLOW_TEMPLATE_STATUS_CONFIG}
            icon={difficulty.icon as React.ReactElement}
            height={26}
            sx={{
              "& .MuiChip-icon": { color: difficulty.color, fontSize: 14 },
            }}
          />
        </Box>
      </Box>

      <Typography
        variant="body2"
        color="text.secondary"
        sx={{
          mb: 1,
          lineHeight: 1.35,
          display: "-webkit-box",
          WebkitLineClamp: 1,
          WebkitBoxOrient: "vertical",
          overflow: "hidden",
        }}
      >
        {template.description}
      </Typography>

      <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, minWidth: 0 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, minWidth: 0 }}>
          <DataIcon sx={{ fontSize: 16, color: accentColors.info }} />
          <Typography variant="caption" color="text.secondary" noWrap>
            {template.typical_dataset_size}
          </Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, minWidth: 0 }}>
          <TimeIcon sx={{ fontSize: 16, color: accentColors.warning }} />
          <Typography variant="caption" color="text.secondary" noWrap>
            {template.estimated_time}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export const WorkflowTemplatesPage: React.FC = () => {
  const navigate = useNavigate();
  const isNarrow = useMediaQuery("(max-width:599.95px)");
  const isShortViewport = useMediaQuery("(max-height:799.95px)");
  const distillationPath = getRoutePath("finetuning-distillation");
  const trainingPath = getRoutePath("finetuning");
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<WorkflowTemplate | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [customPrompts, setCustomPrompts] = useState("");

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    try {
      const res = await fetch("/api/v1/finetune/workflow-templates");
      if (res.ok) {
        const data = await res.json();
        const loadedTemplates = data.templates || [];
        setTemplates(loadedTemplates);
        setSelectedTemplate((current) => current || loadedTemplates[0] || null);
      }
    } catch (err) {
      setError("Failed to load workflow templates");
    }
  };

  const startWorkflow = async (templateId: string, useCustomPrompts: boolean = false) => {
    setLoading(true);
    setError(null);

    try {
      const customizations: any = {};

      if (useCustomPrompts && customPrompts.trim()) {
        const promptList = customPrompts.split("\n").filter((p) => p.trim());
        customizations.prompts = promptList;
        customizations.target_examples = promptList.length;
      }

      const res = await fetch(`/api/v1/finetune/workflow-templates/${templateId}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(customizations),
      });

      if (res.ok) {
        const data = await res.json();
        navigate(`${distillationPath}?job=${data.distillation_job.id}`);
      } else {
        const errorData = await res.json();
        setError(errorData.detail || "Failed to start workflow");
      }
    } catch (err) {
      setError("Failed to start workflow");
    } finally {
      setLoading(false);
    }
  };

  const customPromptCount = customPrompts.split("\n").filter((p) => p.trim()).length;
  const templateListActions = (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
      <Tooltip title="Back to training">
        <IconButton
          size="small"
          aria-label="Back to training"
          onClick={() => navigate(trainingPath)}
          sx={workflowTemplateActionButtonSx}
        >
          <BackIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
      <Tooltip title="Refresh templates">
        <IconButton
          size="small"
          aria-label="Refresh templates"
          onClick={fetchTemplates}
          sx={workflowTemplateActionButtonSx}
        >
          <RefreshIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
    </Box>
  );

  return (
    <PageShell sx={{ py: 1.25 }}>
      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 1 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading */}
      {loading && <LinearProgress sx={{ mb: 1, borderRadius: 1 }} />}

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", lg: selectedTemplate ? "minmax(360px, 0.9fr) minmax(0, 1.6fr)" : "1fr" },
          gap: 1.5,
          alignItems: "start",
        }}
      >
        {/* Templates List */}
        <SectionCard
          title="Quick Start"
          subtitle={`${templates.length} template${templates.length === 1 ? "" : "s"}`}
          icon={<AIIcon />}
          accentColor={accentColors.purple}
          action={templateListActions}
        >
          {templates.length === 0 ? (
            <Box sx={{ py: 5, textAlign: "center" }}>
              <RocketIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
              <Typography color="text.secondary">Loading templates...</Typography>
            </Box>
          ) : (
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {templates.map((template) => (
                <TemplateCard
                  key={template.id}
                  template={template}
                  selected={selectedTemplate?.id === template.id}
                  onClick={() => setSelectedTemplate(template)}
                  onStart={() => startWorkflow(template.id, false)}
                  loading={loading}
                />
              ))}
            </Box>
          )}
        </SectionCard>

        {/* Template Details */}
        {selectedTemplate && !isNarrow && (
            <SectionCard
              title={selectedTemplate.name}
              subtitle={selectedTemplate.use_case}
              icon={<RocketIcon />}
              accentColor={accentColors.success}
              collapsible
              defaultExpanded={!isNarrow}
              action={
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                  <Tooltip title={loading ? "Starting workflow" : "Start with sample prompts"}>
                    <span>
                      <IconButton
                        size="small"
                        aria-label="Start with sample prompts"
                        onClick={() => startWorkflow(selectedTemplate.id, false)}
                        disabled={loading}
                        sx={workflowTemplateActionButtonSx}
                      >
                        <RocketIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </span>
                  </Tooltip>
                  {customPromptCount > 0 && (
                    <Tooltip title={loading ? "Starting workflow" : "Start with custom prompts"}>
                      <span>
                        <IconButton
                          size="small"
                          aria-label="Start with custom prompts"
                          onClick={() => startWorkflow(selectedTemplate.id, true)}
                          disabled={loading}
                          sx={workflowTemplateActionButtonSx}
                        >
                          <StartIcon sx={{ fontSize: 18 }} />
                        </IconButton>
                      </span>
                    </Tooltip>
                  )}
                </Box>
              }
            >
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
                  {/* Configuration */}
                  <Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1, display: "flex", alignItems: "center", gap: 1 }}>
                      <CodeIcon sx={{ fontSize: 18 }} /> Configuration
                    </Typography>
                    <Box sx={{ p: 1.25, ...workflowTemplatePanelSx }}>
                      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(3, minmax(0, 1fr))" }, gap: 1.25 }}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Teacher Model
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 600, color: accentColors.info }}>
                            {selectedTemplate.distillation_config.teacher_model}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Strategy
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 600, color: accentColors.purple }}>
                            {selectedTemplate.distillation_config.strategy}
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Target Examples
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 600, color: accentColors.success }}>
                            {selectedTemplate.distillation_config.target_examples}
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  </Box>

                  {/* Sample Prompts */}
                  <Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1, display: "flex", alignItems: "center", gap: 1 }}>
                      <TipIcon sx={{ fontSize: 18 }} /> Sample Prompts
                    </Typography>
                    <Box sx={{ p: 1.1, display: "grid", gap: 0.5, ...workflowTemplatePanelSx }}>
                      {selectedTemplate.sample_prompts.map((prompt, i) => (
                        <Box
                          key={i}
                          sx={{
                            pl: 1.5,
                            py: 0.35,
                            borderLeft: `2px solid ${accentColors.info}`,
                            fontSize: "0.85rem",
                            color: "text.secondary",
                          }}
                        >
                          {prompt}
                        </Box>
                      ))}
                    </Box>
                  </Box>

                {/* Custom Prompts */}
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.75, display: "flex", alignItems: "center", gap: 1 }}>
                    <EditIcon sx={{ fontSize: 18 }} /> Custom Prompts
                  </Typography>
                  <TextField
                    value={customPrompts}
                    onChange={(e) => setCustomPrompts(e.target.value)}
                    placeholder="Enter your custom prompts here, one per line..."
                    multiline
                    rows={3}
                    fullWidth
                    size="small"
                    sx={{
                      "& textarea": { fontFamily: "monospace", fontSize: "0.85rem" },
                    }}
                  />
                  {customPromptCount > 0 && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: "block" }}>
                      {customPromptCount} custom prompt{customPromptCount !== 1 ? "s" : ""}
                    </Typography>
                  )}
                </Box>

                {!isShortViewport && (
                  <>
                    <Divider sx={{ borderColor: "divider" }} />

                    {/* Expected Results & Training Notes */}
                    <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(2, minmax(0, 1fr))" }, gap: 1.5 }}>
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                          Expected Results
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.85rem" }}>
                          {selectedTemplate.expected_quality}
                        </Typography>
                      </Box>
                      <Box>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                          Training Notes
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.85rem" }}>
                          {selectedTemplate.training_notes}
                        </Typography>
                      </Box>
                    </Box>
                  </>
                )}

              </Box>
            </SectionCard>
        )}
      </Box>
    </PageShell>
  );
};

export default WorkflowTemplatesPage;
