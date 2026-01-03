import React, { useEffect, useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  IconButton,
  Chip,
  Grid,
  Alert,
  Tooltip,
  LinearProgress,
  Divider,
  alpha,
} from "@mui/material";
import {
  RocketLaunch as RocketIcon,
  ArrowBack as BackIcon,
  Refresh as RefreshIcon,
  Schedule as TimeIcon,
  Storage as DataIcon,
  GpsFixed as TargetIcon,
  School as LearnIcon,
  Star as StarIcon,
  Code as CodeIcon,
  Psychology as AIIcon,
  Support as SupportIcon,
  Create as CreateIcon,
  Lightbulb as TipIcon,
  PlayArrow as StartIcon,
  Edit as EditIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

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

// Accent colors
const accentColors = {
  primary: "#6366f1",
  success: "#10b981",
  warning: "#f59e0b",
  info: "#06b6d4",
  purple: "#8b5cf6",
  rose: "#f43f5e",
};

// Difficulty config
const difficultyConfig: Record<string, { color: string; label: string; icon: React.ReactNode }> = {
  beginner: { color: accentColors.success, label: "Beginner", icon: <StarIcon /> },
  intermediate: { color: accentColors.warning, label: "Intermediate", icon: <LearnIcon /> },
  advanced: { color: accentColors.rose, label: "Advanced", icon: <CodeIcon /> },
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

// Template Card Component
interface TemplateCardProps {
  template: WorkflowTemplate;
  selected: boolean;
  onClick: () => void;
}

const TemplateCard: React.FC<TemplateCardProps> = ({ template, selected, onClick }) => {
  const difficulty = difficultyConfig[template.difficulty] || difficultyConfig.beginner;

  return (
    <Box
      onClick={onClick}
      sx={{
        p: 2.5,
        borderRadius: 2,
        bgcolor: selected ? alpha(accentColors.success, 0.1) : "rgba(255, 255, 255, 0.02)",
        border: `2px solid ${selected ? accentColors.success : "rgba(255, 255, 255, 0.06)"}`,
        cursor: "pointer",
        transition: "all 0.3s ease-in-out",
        "&:hover": {
          bgcolor: alpha(accentColors.primary, 0.08),
          borderColor: alpha(accentColors.primary, 0.3),
          transform: "translateY(-2px)",
        },
      }}
    >
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", mb: 1.5 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, fontSize: "1.1rem" }}>
          {template.name}
        </Typography>
        <Chip
          icon={difficulty.icon as React.ReactElement}
          label={difficulty.label}
          size="small"
          sx={{
            bgcolor: alpha(difficulty.color, 0.1),
            color: difficulty.color,
            border: `1px solid ${alpha(difficulty.color, 0.3)}`,
            fontWeight: 600,
            fontSize: "0.7rem",
            height: 26,
            "& .MuiChip-icon": { color: difficulty.color, fontSize: 14 },
          }}
        />
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, lineHeight: 1.5 }}>
        {template.description}
      </Typography>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <DataIcon sx={{ fontSize: 16, color: accentColors.info }} />
          <Typography variant="caption" color="text.secondary">
            {template.typical_dataset_size}
          </Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <TimeIcon sx={{ fontSize: 16, color: accentColors.warning }} />
          <Typography variant="caption" color="text.secondary">
            {template.estimated_time}
          </Typography>
        </Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <TargetIcon sx={{ fontSize: 16, color: accentColors.purple }} />
          <Typography variant="caption" color="text.secondary">
            {template.use_case}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export const WorkflowTemplatesPage: React.FC = () => {
  const navigate = useNavigate();
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
        setTemplates(data.templates || []);
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
        navigate(`/finetuning/distillation?job=${data.distillation_job.id}`);
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
              Quick Start Templates
            </Typography>
            <Chip
              icon={<RocketIcon />}
              label="1-Click Deploy"
              size="small"
              sx={{
                height: 26,
                bgcolor: alpha(accentColors.success, 0.1),
                border: `1px solid ${alpha(accentColors.success, 0.2)}`,
                color: accentColors.success,
                fontWeight: 600,
                fontSize: "0.7rem",
                "& .MuiChip-icon": { color: accentColors.success, fontSize: 16 },
              }}
            />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.875rem", maxWidth: 600 }}>
            Pre-configured workflows for common AI use cases. Each template includes optimized distillation settings,
            sample prompts, and training recommendations.
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
              onClick={fetchTemplates}
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

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading */}
      {loading && <LinearProgress sx={{ mb: 3, borderRadius: 1 }} />}

      <Grid container spacing={3}>
        {/* Templates List */}
        <Grid item xs={12} lg={selectedTemplate ? 5 : 12}>
          <SectionCard
            title="Choose a Template"
            subtitle={`${templates.length} templates available`}
            icon={<AIIcon />}
            accentColor={accentColors.purple}
          >
            {templates.length === 0 ? (
              <Box sx={{ p: 4, textAlign: "center" }}>
                <RocketIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                <Typography color="text.secondary">Loading templates...</Typography>
              </Box>
            ) : (
              <Grid container spacing={2}>
                {templates.map((template) => (
                  <Grid item xs={12} md={selectedTemplate ? 12 : 6} lg={selectedTemplate ? 12 : 4} key={template.id}>
                    <TemplateCard
                      template={template}
                      selected={selectedTemplate?.id === template.id}
                      onClick={() => setSelectedTemplate(template)}
                    />
                  </Grid>
                ))}
              </Grid>
            )}
          </SectionCard>
        </Grid>

        {/* Template Details */}
        {selectedTemplate && (
          <Grid item xs={12} lg={7}>
            <SectionCard
              title={selectedTemplate.name}
              subtitle={selectedTemplate.use_case}
              icon={<RocketIcon />}
              accentColor={accentColors.success}
            >
              <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                {/* Configuration */}
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5, display: "flex", alignItems: "center", gap: 1 }}>
                    <CodeIcon sx={{ fontSize: 18 }} /> Configuration
                  </Typography>
                  <Box
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      bgcolor: "rgba(0, 0, 0, 0.3)",
                      border: "1px solid rgba(255, 255, 255, 0.06)",
                    }}
                  >
                    <Grid container spacing={2}>
                      <Grid item xs={4}>
                        <Typography variant="caption" color="text.secondary">
                          Teacher Model
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600, color: accentColors.info }}>
                          {selectedTemplate.distillation_config.teacher_model}
                        </Typography>
                      </Grid>
                      <Grid item xs={4}>
                        <Typography variant="caption" color="text.secondary">
                          Strategy
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600, color: accentColors.purple }}>
                          {selectedTemplate.distillation_config.strategy}
                        </Typography>
                      </Grid>
                      <Grid item xs={4}>
                        <Typography variant="caption" color="text.secondary">
                          Target Examples
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600, color: accentColors.success }}>
                          {selectedTemplate.distillation_config.target_examples}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                </Box>

                {/* Sample Prompts */}
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5, display: "flex", alignItems: "center", gap: 1 }}>
                    <TipIcon sx={{ fontSize: 18 }} /> Sample Prompts
                  </Typography>
                  <Box
                    sx={{
                      maxHeight: 180,
                      overflow: "auto",
                      p: 2,
                      borderRadius: 2,
                      bgcolor: "rgba(0, 0, 0, 0.3)",
                      border: "1px solid rgba(255, 255, 255, 0.06)",
                    }}
                  >
                    {selectedTemplate.sample_prompts.map((prompt, i) => (
                      <Box
                        key={i}
                        sx={{
                          pl: 1.5,
                          py: 0.75,
                          mb: 1,
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
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1, display: "flex", alignItems: "center", gap: 1 }}>
                    <EditIcon sx={{ fontSize: 18 }} /> Custom Prompts (Optional)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1.5 }}>
                    Override the sample prompts with your own (one per line)
                  </Typography>
                  <TextField
                    value={customPrompts}
                    onChange={(e) => setCustomPrompts(e.target.value)}
                    placeholder="Enter your custom prompts here, one per line..."
                    multiline
                    rows={4}
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

                <Divider sx={{ borderColor: "rgba(255, 255, 255, 0.06)" }} />

                {/* Expected Results & Training Notes */}
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                      Expected Results
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.85rem" }}>
                      {selectedTemplate.expected_quality}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                      Training Notes
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.85rem" }}>
                      {selectedTemplate.training_notes}
                    </Typography>
                  </Grid>
                </Grid>

                <Divider sx={{ borderColor: "rgba(255, 255, 255, 0.06)" }} />

                {/* Action Buttons */}
                <Box sx={{ display: "flex", gap: 2 }}>
                  <Button
                    variant="contained"
                    size="large"
                    startIcon={<RocketIcon />}
                    onClick={() => startWorkflow(selectedTemplate.id, false)}
                    disabled={loading}
                    sx={{
                      flex: 1,
                      py: 1.5,
                      fontWeight: 700,
                      background: `linear-gradient(135deg, ${accentColors.success} 0%, ${accentColors.info} 100%)`,
                      boxShadow: `0 4px 14px ${alpha(accentColors.success, 0.4)}`,
                    }}
                  >
                    {loading ? "Starting..." : "Start with Sample Prompts"}
                  </Button>

                  {customPromptCount > 0 && (
                    <Button
                      variant="contained"
                      size="large"
                      startIcon={<StartIcon />}
                      onClick={() => startWorkflow(selectedTemplate.id, true)}
                      disabled={loading}
                      sx={{
                        flex: 1,
                        py: 1.5,
                        fontWeight: 700,
                        background: `linear-gradient(135deg, ${accentColors.primary} 0%, ${accentColors.purple} 100%)`,
                        boxShadow: `0 4px 14px ${alpha(accentColors.primary, 0.4)}`,
                      }}
                    >
                      {loading ? "Starting..." : "Start with Custom Prompts"}
                    </Button>
                  )}
                </Box>

                <Alert severity="info" icon={<TipIcon />} sx={{ mt: 1 }}>
                  <Typography variant="body2">
                    This will start distillation and automatically create a dataset. You can then start training with one
                    click when distillation completes.
                  </Typography>
                </Alert>
              </Box>
            </SectionCard>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default WorkflowTemplatesPage;
