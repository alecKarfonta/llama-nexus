import React, { useEffect, useState, useCallback } from "react";
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
    LinearProgress,
    Slider,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Checkbox,
    FormControlLabel,
    IconButton,
    alpha,
} from "@mui/material";
import {
    AutoStories as BookIcon,
    Storage as DatasetIcon,
    PlayArrow as GenerateIcon,
    Visibility as PreviewIcon,
    Refresh as RefreshIcon,
    ArrowBack as BackIcon,
    Settings as ConfigIcon,
    TextFields as TextIcon,
    Calculate as EstimateIcon,
    CheckCircle as SuccessIcon,
    Psychology as AIIcon,
    QuestionAnswer as QAIcon,
    Summarize as SummaryIcon,
    AttachMoney as CostIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

// Types
type Domain = {
    id: string;
    name: string;
    description?: string;
    document_count: number;
    total_chunks: number;
};

type Document = {
    id: string;
    name: string;
    domain_id: string;
    chunk_count: number;
    status: string;
};

type GeneratedExample = {
    prompt: string;
    completion: string;
    metadata?: Record<string, any>;
};

type DatasetEstimate = {
    total_documents: number;
    valid_documents: number;
    total_chunks: number;
    estimated_examples: number;
};

type DistillationEstimate = {
    total_documents: number;
    total_chunks: number;
    estimated_examples: number;
    estimated_cost_usd: number;
    model: string;
};

type DistillationJob = {
    id: string;
    status: string;
    progress: {
        total_chunks: number;
        processed_chunks: number;
        generated_examples: number;
        progress_percent: number;
        current_document: string;
    };
    output_dataset_id?: string;
    error?: string;
};

type GenerationMode = "continuation" | "distillation";

const TEACHER_PROVIDERS = [
    { value: "openai", label: "OpenAI (GPT-4o)" },
    { value: "anthropic", label: "Anthropic (Claude)" },
    { value: "google", label: "Google (Gemini)" },
    { value: "local", label: "Local Model" },
];

const GENERATION_MODES = [
    { value: "qa_generation", label: "Q&A Pairs", icon: <QAIcon /> },
    { value: "summarization", label: "Summaries", icon: <SummaryIcon /> },
    { value: "explanation", label: "Explanations", icon: <AIIcon /> },
    { value: "mixed", label: "Mixed (All)", icon: <DatasetIcon /> },
];

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

// Stat Box Component
interface StatBoxProps {
    label: string;
    value: string | number;
    icon: React.ReactNode;
    color: string;
}

const StatBox: React.FC<StatBoxProps> = ({ label, value, icon, color }) => (
    <Box
        sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: alpha(color, 0.08),
            border: `1px solid ${alpha(color, 0.15)}`,
            textAlign: "center",
            transition: "all 0.2s",
            "&:hover": { bgcolor: alpha(color, 0.12), transform: "translateY(-2px)" },
        }}
    >
        <Box sx={{ mb: 1, color, "& .MuiSvgIcon-root": { fontSize: 28 } }}>{icon}</Box>
        <Typography sx={{ fontWeight: 700, fontSize: "1.5rem", color: "text.primary", lineHeight: 1 }}>
            {value}
        </Typography>
        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.7rem" }}>
            {label}
        </Typography>
    </Box>
);


export const BookDatasetPage: React.FC = () => {
    const navigate = useNavigate();

    // Domain and document state
    const [domains, setDomains] = useState<Domain[]>([]);
    const [selectedDomainId, setSelectedDomainId] = useState<string>("");
    const [documents, setDocuments] = useState<Document[]>([]);
    const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

    // Configuration state
    const [name, setName] = useState<string>("");
    const [description, setDescription] = useState<string>("");
    const [contextChunks, setContextChunks] = useState<number>(3);
    const [outputChunks, setOutputChunks] = useState<number>(1);
    const [stride, setStride] = useState<number>(1);
    const [includeMetadata, setIncludeMetadata] = useState<boolean>(false);
    const [systemPrompt, setSystemPrompt] = useState<string>("");
    const [maxExamples, setMaxExamples] = useState<number | null>(null);

    // UI state
    const [estimate, setEstimate] = useState<DatasetEstimate | null>(null);
    const [preview, setPreview] = useState<GeneratedExample[]>(null);
    const [loading, setLoading] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    // Generation mode state
    const [generationMode, setGenerationMode] = useState<GenerationMode>("continuation");

    // Distillation-specific state
    const [teacherProvider, setTeacherProvider] = useState("openai");
    const [teacherModel, setTeacherModel] = useState("gpt-4o-mini");
    const [distillationMode, setDistillationMode] = useState("qa_generation");
    const [questionsPerChunk, setQuestionsPerChunk] = useState(3);
    const [maxChunks, setMaxChunks] = useState<number | null>(null);
    const [distillEstimate, setDistillEstimate] = useState<DistillationEstimate | null>(null);
    const [distillJob, setDistillJob] = useState<DistillationJob | null>(null);

    // Fetch domains on mount
    useEffect(() => {
        fetch("/api/v1/rag/domains")
            .then((res) => res.json())
            .then((data) => {
                const domainList = data.domains || data || [];
                setDomains(domainList);
            })
            .catch(() => setError("Failed to load domains"));
    }, []);

    // Fetch documents when domain changes
    useEffect(() => {
        if (!selectedDomainId) {
            setDocuments([]);
            setSelectedDocIds([]);
            return;
        }

        fetch(`/api/v1/rag/documents?domain_id=${selectedDomainId}&limit=100`)
            .then((res) => res.json())
            .then((data) => {
                const docs = data.documents || data || [];
                setDocuments(docs.filter((d: Document) => d.status === "ready"));
            })
            .catch(() => setError("Failed to load documents"));
    }, [selectedDomainId]);

    // Fetch estimate when config changes
    const fetchEstimate = useCallback(async () => {
        if (!selectedDomainId) return;

        setLoading(true);
        try {
            const res = await fetch("/api/v1/finetune/datasets/book/estimate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    domain_id: selectedDomainId,
                    document_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
                    context_chunks: contextChunks,
                    output_chunks: outputChunks,
                    stride: stride,
                }),
            });
            if (res.ok) {
                const data = await res.json();
                setEstimate(data);
            }
        } catch {
            // Ignore estimate errors
        } finally {
            setLoading(false);
        }
    }, [selectedDomainId, selectedDocIds, contextChunks, outputChunks, stride]);

    useEffect(() => {
        fetchEstimate();
    }, [fetchEstimate]);

    // Fetch preview
    const fetchPreview = useCallback(async () => {
        if (!selectedDomainId) return;

        setLoading(true);
        setError(null);
        try {
            const res = await fetch("/api/v1/finetune/datasets/book/preview?num_examples=3", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: name || "preview",
                    domain_id: selectedDomainId,
                    document_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
                    context_chunks: contextChunks,
                    output_chunks: outputChunks,
                    stride: stride,
                    include_metadata: includeMetadata,
                    system_prompt: systemPrompt,
                }),
            });
            if (res.ok) {
                const data = await res.json();
                setPreview(data.examples || []);
            } else {
                const errData = await res.json();
                setError(errData.detail || "Failed to generate preview");
            }
        } catch {
            setError("Failed to generate preview");
        } finally {
            setLoading(false);
        }
    }, [selectedDomainId, selectedDocIds, contextChunks, outputChunks, stride, includeMetadata, systemPrompt, name]);

    // Generate dataset
    const handleGenerate = async () => {
        if (!selectedDomainId || !name) {
            setError("Please select a domain and provide a dataset name");
            return;
        }

        setGenerating(true);
        setError(null);
        setSuccess(null);

        try {
            const res = await fetch("/api/v1/finetune/datasets/book/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: name,
                    description: description,
                    domain_id: selectedDomainId,
                    document_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
                    context_chunks: contextChunks,
                    output_chunks: outputChunks,
                    stride: stride,
                    include_metadata: includeMetadata,
                    system_prompt: systemPrompt,
                    max_examples: maxExamples,
                }),
            });

            if (res.ok) {
                const data = await res.json();
                setSuccess(`Dataset "${data.dataset.name}" created with ${data.statistics.examples_generated} examples!`);
                // Clear form
                setName("");
                setDescription("");
                setPreview([]);
            } else {
                const errData = await res.json();
                setError(errData.detail || "Failed to generate dataset");
            }
        } catch {
            setError("Failed to generate dataset");
        } finally {
            setGenerating(false);
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
                            Book Dataset Generator
                        </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ fontSize: "0.875rem", maxWidth: 500 }}>
                        Generate training datasets from document chunks with configurable context windows
                    </Typography>
                </Box>

                <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                    <Button
                        variant="outlined"
                        size="small"
                        startIcon={<BackIcon />}
                        onClick={() => navigate("/finetuning/datasets")}
                        sx={{
                            borderColor: "rgba(255, 255, 255, 0.1)",
                            color: "text.secondary",
                            "&:hover": { borderColor: accentColors.primary, color: accentColors.primary },
                        }}
                    >
                        Back to Datasets
                    </Button>
                </Box>
            </Box>

            {/* Alerts */}
            {error && (
                <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}
            {success && (
                <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess(null)}>
                    {success}
                </Alert>
            )}

            <Grid container spacing={3}>
                {/* Configuration Panel */}
                <Grid item xs={12} md={5}>
                    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        {/* Source Selection */}
                        <SectionCard
                            title="Source Selection"
                            subtitle="Choose documents to include"
                            icon={<BookIcon />}
                            accentColor={accentColors.info}
                        >
                            <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                                <FormControl fullWidth size="small">
                                    <InputLabel>Domain</InputLabel>
                                    <Select
                                        value={selectedDomainId}
                                        label="Domain"
                                        onChange={(e) => setSelectedDomainId(e.target.value)}
                                    >
                                        {domains.map((domain) => (
                                            <MenuItem key={domain.id} value={domain.id}>
                                                {domain.name} ({domain.document_count} docs, {domain.total_chunks} chunks)
                                            </MenuItem>
                                        ))}
                                    </Select>
                                </FormControl>

                                {documents.length > 0 && (
                                    <Box>
                                        <Typography variant="caption" sx={{ color: "text.secondary", mb: 1, display: "block" }}>
                                            Documents (optional - leave empty for all)
                                        </Typography>
                                        <Box sx={{ maxHeight: 150, overflow: "auto" }}>
                                            {documents.map((doc) => (
                                                <FormControlLabel
                                                    key={doc.id}
                                                    control={
                                                        <Checkbox
                                                            size="small"
                                                            checked={selectedDocIds.includes(doc.id)}
                                                            onChange={(e) => {
                                                                if (e.target.checked) {
                                                                    setSelectedDocIds([...selectedDocIds, doc.id]);
                                                                } else {
                                                                    setSelectedDocIds(selectedDocIds.filter((id) => id !== doc.id));
                                                                }
                                                            }}
                                                        />
                                                    }
                                                    label={
                                                        <Typography variant="body2" sx={{ fontSize: "0.8rem" }}>
                                                            {doc.name} ({doc.chunk_count} chunks)
                                                        </Typography>
                                                    }
                                                />
                                            ))}
                                        </Box>
                                    </Box>
                                )}
                            </Box>
                        </SectionCard>

                        {/* Generation Mode Toggle */}
                        <SectionCard
                            title="Generation Mode"
                            subtitle="Choose how to create training examples"
                            icon={<AIIcon />}
                            accentColor={accentColors.info}
                        >
                            <Box sx={{ display: "flex", gap: 2 }}>
                                <Button
                                    variant={generationMode === "continuation" ? "contained" : "outlined"}
                                    onClick={() => setGenerationMode("continuation")}
                                    startIcon={<TextIcon />}
                                    sx={{
                                        flex: 1,
                                        bgcolor: generationMode === "continuation" ? accentColors.info : "transparent",
                                        borderColor: accentColors.info,
                                        color: generationMode === "continuation" ? "white" : accentColors.info,
                                        "&:hover": {
                                            bgcolor: generationMode === "continuation" ? accentColors.info : alpha(accentColors.info, 0.1),
                                        },
                                    }}
                                >
                                    Continuation
                                </Button>
                                <Button
                                    variant={generationMode === "distillation" ? "contained" : "outlined"}
                                    onClick={() => setGenerationMode("distillation")}
                                    startIcon={<QAIcon />}
                                    sx={{
                                        flex: 1,
                                        bgcolor: generationMode === "distillation" ? accentColors.rose : "transparent",
                                        borderColor: accentColors.rose,
                                        color: generationMode === "distillation" ? "white" : accentColors.rose,
                                        "&:hover": {
                                            bgcolor: generationMode === "distillation" ? accentColors.rose : alpha(accentColors.rose, 0.1),
                                        },
                                    }}
                                >
                                    Distillation (Q&A)
                                </Button>
                            </Box>
                            <Typography variant="caption" sx={{ color: "text.secondary", mt: 1, display: "block" }}>
                                {generationMode === "continuation"
                                    ? "Generates context → next chunk pairs for text continuation training"
                                    : "Uses an LLM to generate Q&A pairs, summaries, or explanations from your documents"
                                }
                            </Typography>
                        </SectionCard>

                        {/* Distillation Config - Only show when distillation mode is selected */}
                        {generationMode === "distillation" && (
                            <SectionCard
                                title="Distillation Settings"
                                subtitle="Configure the teacher model and generation type"
                                icon={<QAIcon />}
                                accentColor={accentColors.rose}
                            >
                                <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Teacher Provider</InputLabel>
                                        <Select
                                            value={teacherProvider}
                                            label="Teacher Provider"
                                            onChange={(e) => setTeacherProvider(e.target.value)}
                                        >
                                            {TEACHER_PROVIDERS.map((p) => (
                                                <MenuItem key={p.value} value={p.value}>{p.label}</MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>

                                    <TextField
                                        label="Model Name"
                                        value={teacherModel}
                                        onChange={(e) => setTeacherModel(e.target.value)}
                                        size="small"
                                        fullWidth
                                        helperText="e.g. gpt-4o-mini, claude-3-5-sonnet, gemini-1.5-pro"
                                    />

                                    <FormControl fullWidth size="small">
                                        <InputLabel>Generation Type</InputLabel>
                                        <Select
                                            value={distillationMode}
                                            label="Generation Type"
                                            onChange={(e) => setDistillationMode(e.target.value)}
                                        >
                                            {GENERATION_MODES.map((m) => (
                                                <MenuItem key={m.value} value={m.value}>
                                                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                                        {m.icon}
                                                        {m.label}
                                                    </Box>
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>

                                    <Box>
                                        <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                            Questions per Chunk: {questionsPerChunk}
                                        </Typography>
                                        <Slider
                                            value={questionsPerChunk}
                                            onChange={(_, v) => setQuestionsPerChunk(v as number)}
                                            min={1}
                                            max={10}
                                            step={1}
                                            marks
                                            sx={{ color: accentColors.rose }}
                                        />
                                    </Box>

                                    <TextField
                                        label="Max Chunks (optional)"
                                        value={maxChunks ?? ""}
                                        onChange={(e) => setMaxChunks(e.target.value ? parseInt(e.target.value) : null)}
                                        size="small"
                                        fullWidth
                                        type="number"
                                        helperText="Limit chunks to process (leave empty for all)"
                                    />

                                    {/* Distillation Estimate */}
                                    {distillEstimate && (
                                        <Alert severity="info" sx={{ bgcolor: alpha(accentColors.info, 0.1) }}>
                                            <Typography variant="body2">
                                                <strong>{distillEstimate.total_chunks}</strong> chunks → ~<strong>{distillEstimate.estimated_examples}</strong> examples
                                                {distillEstimate.estimated_cost_usd > 0 && (
                                                    <> (Est. cost: <strong>${distillEstimate.estimated_cost_usd}</strong>)</>
                                                )}
                                            </Typography>
                                        </Alert>
                                    )}

                                    {/* Distillation Job Progress */}
                                    {distillJob && distillJob.status === "running" && (
                                        <Box>
                                            <Typography variant="body2" sx={{ mb: 1 }}>
                                                Processing: {distillJob.progress.current_document}
                                            </Typography>
                                            <LinearProgress
                                                variant="determinate"
                                                value={distillJob.progress.progress_percent}
                                                sx={{ height: 8, borderRadius: 4, bgcolor: alpha(accentColors.rose, 0.2) }}
                                            />
                                            <Typography variant="caption" sx={{ color: "text.secondary", mt: 0.5, display: "block" }}>
                                                {distillJob.progress.processed_chunks}/{distillJob.progress.total_chunks} chunks •
                                                {distillJob.progress.generated_examples} examples generated
                                            </Typography>
                                        </Box>
                                    )}
                                </Box>
                            </SectionCard>
                        )}

                        {/* Configuration - Only show for continuation mode */}
                        {generationMode === "continuation" && (
                            <SectionCard
                                title="Configuration"
                                subtitle="Adjust context window settings"
                                icon={<ConfigIcon />}
                                accentColor={accentColors.purple}
                            >
                                <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                                    <TextField
                                        label="Dataset Name"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        size="small"
                                        fullWidth
                                        required
                                    />

                                    <TextField
                                        label="Description"
                                        value={description}
                                        onChange={(e) => setDescription(e.target.value)}
                                        size="small"
                                        fullWidth
                                        multiline
                                        rows={2}
                                    />

                                    <Box>
                                        <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                            Context Chunks: {contextChunks}
                                        </Typography>
                                        <Slider
                                            value={contextChunks}
                                            onChange={(_, v) => setContextChunks(v as number)}
                                            min={1}
                                            max={10}
                                            step={1}
                                            marks
                                            sx={{ color: accentColors.purple }}
                                        />
                                        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem" }}>
                                            Number of previous chunks to include in the prompt
                                        </Typography>
                                    </Box>

                                    <Box>
                                        <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                            Output Chunks: {outputChunks}
                                        </Typography>
                                        <Slider
                                            value={outputChunks}
                                            onChange={(_, v) => setOutputChunks(v as number)}
                                            min={1}
                                            max={5}
                                            step={1}
                                            marks
                                            sx={{ color: accentColors.purple }}
                                        />
                                        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem" }}>
                                            Number of chunks to predict (completion)
                                        </Typography>
                                    </Box>

                                    <Box>
                                        <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                            Stride: {stride}
                                        </Typography>
                                        <Slider
                                            value={stride}
                                            onChange={(_, v) => setStride(v as number)}
                                            min={1}
                                            max={5}
                                            step={1}
                                            marks
                                            sx={{ color: accentColors.purple }}
                                        />
                                        <Typography variant="caption" sx={{ color: "text.secondary", fontSize: "0.65rem" }}>
                                            Step size for sliding window (1 = max overlap)
                                        </Typography>
                                    </Box>

                                    <FormControlLabel
                                        control={
                                            <Checkbox
                                                size="small"
                                                checked={includeMetadata}
                                                onChange={(e) => setIncludeMetadata(e.target.checked)}
                                            />
                                        }
                                        label={<Typography variant="body2">Include metadata in examples</Typography>}
                                    />

                                    <TextField
                                        label="System Prompt (optional)"
                                        value={systemPrompt}
                                        onChange={(e) => setSystemPrompt(e.target.value)}
                                        size="small"
                                        fullWidth
                                        multiline
                                        rows={2}
                                        placeholder="Custom system prompt prefix..."
                                    />

                                    <TextField
                                        label="Max Examples (optional)"
                                        type="number"
                                        value={maxExamples || ""}
                                        onChange={(e) => setMaxExamples(e.target.value ? parseInt(e.target.value) : null)}
                                        size="small"
                                        fullWidth
                                        placeholder="Leave empty for all"
                                    />
                                </Box>
                            </SectionCard>
                        )}

                        {/* Estimate */}
                        {estimate && (
                            <SectionCard
                                title="Estimate"
                                subtitle="Predicted dataset size"
                                icon={<EstimateIcon />}
                                accentColor={accentColors.success}
                            >
                                <Grid container spacing={2}>
                                    <Grid item xs={6}>
                                        <StatBox
                                            label="Documents"
                                            value={estimate.valid_documents}
                                            icon={<BookIcon />}
                                            color={accentColors.info}
                                        />
                                    </Grid>
                                    <Grid item xs={6}>
                                        <StatBox
                                            label="Total Chunks"
                                            value={estimate.total_chunks.toLocaleString()}
                                            icon={<TextIcon />}
                                            color={accentColors.purple}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <StatBox
                                            label="Estimated Examples"
                                            value={estimate.estimated_examples.toLocaleString()}
                                            icon={<DatasetIcon />}
                                            color={accentColors.success}
                                        />
                                    </Grid>
                                </Grid>
                            </SectionCard>
                        )}

                        {/* Actions */}
                        <Box sx={{ display: "flex", gap: 2 }}>
                            <Button
                                variant="outlined"
                                startIcon={<PreviewIcon />}
                                onClick={fetchPreview}
                                disabled={!selectedDomainId || loading}
                                sx={{ flex: 1 }}
                            >
                                Preview
                            </Button>
                            <Button
                                variant="contained"
                                startIcon={generating ? undefined : <GenerateIcon />}
                                onClick={handleGenerate}
                                disabled={!selectedDomainId || !name || generating}
                                sx={{
                                    flex: 2,
                                    background: `linear-gradient(135deg, ${accentColors.success} 0%, ${accentColors.info} 100%)`,
                                }}
                            >
                                {generating ? "Generating..." : "Generate Dataset"}
                            </Button>
                        </Box>
                    </Box>
                </Grid>

                {/* Preview Panel */}
                <Grid item xs={12} md={7}>
                    <SectionCard
                        title="Preview"
                        subtitle={preview.length > 0 ? `${preview.length} example(s)` : "Generate a preview to see examples"}
                        icon={<PreviewIcon />}
                        accentColor={accentColors.warning}
                        action={
                            <Tooltip title="Refresh Preview">
                                <IconButton size="small" onClick={fetchPreview} disabled={!selectedDomainId || loading}>
                                    <RefreshIcon sx={{ fontSize: 18 }} />
                                </IconButton>
                            </Tooltip>
                        }
                    >
                        {loading && <LinearProgress sx={{ mb: 2 }} />}

                        {preview.length === 0 ? (
                            <Box sx={{ py: 6, textAlign: "center" }}>
                                <PreviewIcon sx={{ fontSize: 48, color: "text.secondary", opacity: 0.3, mb: 2 }} />
                                <Typography color="text.secondary">
                                    Select a domain and click "Preview" to see example training pairs
                                </Typography>
                            </Box>
                        ) : (
                            <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                                {preview.map((example, idx) => (
                                    <Box
                                        key={idx}
                                        sx={{
                                            p: 2,
                                            borderRadius: 2,
                                            bgcolor: "rgba(0, 0, 0, 0.2)",
                                            border: "1px solid rgba(255, 255, 255, 0.06)",
                                        }}
                                    >
                                        <Typography variant="caption" sx={{ color: accentColors.info, fontWeight: 600 }}>
                                            Example {idx + 1}
                                        </Typography>

                                        <Box sx={{ mt: 1 }}>
                                            <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                                PROMPT
                                            </Typography>
                                            <Box
                                                sx={{
                                                    mt: 0.5,
                                                    p: 1.5,
                                                    borderRadius: 1,
                                                    bgcolor: alpha(accentColors.info, 0.1),
                                                    border: `1px solid ${alpha(accentColors.info, 0.2)}`,
                                                    maxHeight: 150,
                                                    overflow: "auto",
                                                }}
                                            >
                                                <Typography
                                                    variant="body2"
                                                    sx={{ fontFamily: "monospace", fontSize: "0.75rem", whiteSpace: "pre-wrap" }}
                                                >
                                                    {example.prompt.length > 500 ? example.prompt.slice(0, 500) + "..." : example.prompt}
                                                </Typography>
                                            </Box>
                                        </Box>

                                        <Box sx={{ mt: 2 }}>
                                            <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                                COMPLETION
                                            </Typography>
                                            <Box
                                                sx={{
                                                    mt: 0.5,
                                                    p: 1.5,
                                                    borderRadius: 1,
                                                    bgcolor: alpha(accentColors.success, 0.1),
                                                    border: `1px solid ${alpha(accentColors.success, 0.2)}`,
                                                    maxHeight: 100,
                                                    overflow: "auto",
                                                }}
                                            >
                                                <Typography
                                                    variant="body2"
                                                    sx={{ fontFamily: "monospace", fontSize: "0.75rem", whiteSpace: "pre-wrap" }}
                                                >
                                                    {example.completion.length > 300 ? example.completion.slice(0, 300) + "..." : example.completion}
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Box>
                                ))}
                            </Box>
                        )}
                    </SectionCard>
                </Grid>
            </Grid>
        </Box>
    );
};

export default BookDatasetPage;
