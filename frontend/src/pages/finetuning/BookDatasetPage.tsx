import React, { useEffect, useState, useCallback } from "react";
import {
    Box,
    Typography,
    Grid,
    Alert,
    Tooltip,
    LinearProgress,
    CircularProgress,
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
    ArrowBack as BackIcon,
    Settings as ConfigIcon,
    TextFields as TextIcon,
    Calculate as EstimateIcon,
    Psychology as AIIcon,
    QuestionAnswer as QAIcon,
    Summarize as SummaryIcon,
    Stop as CancelIcon,
} from "@mui/icons-material";
import type { SxProps, Theme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import { getRoutePath } from "@/config/navigation";
import { MetricTile, PageShell, SectionCard, StatusChip, type StatusChipConfigMap } from "@/components/common";
import { accentColors } from "@/utils/theme";

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
    prompt?: string;
    completion?: string;
    instruction?: string;
    input?: string;
    output?: string;
    response?: string;
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

const BOOK_DATASET_STATUS_CONFIG: StatusChipConfigMap = {
    books: { color: accentColors.success, label: "Book datasets" },
    continuation: { color: accentColors.purple, label: "Continuation" },
    distillation: { color: accentColors.rose, label: "Teacher distillation" },
    draft: { color: accentColors.warning, label: "Draft" },
    generating: { color: accentColors.info, label: "Generating" },
    previewed: { color: accentColors.success, label: "Previewed" },
    run: { color: accentColors.info, label: "Run setup" },
};

const bookDatasetPanelSx = {
    borderRadius: "8px",
    bgcolor: "background.paper",
    border: "1px solid",
    borderColor: "divider",
} as const;

const bookDatasetIconButtonSx = {
    width: 32,
    height: 32,
    borderRadius: "8px",
    border: "1px solid",
    borderColor: "divider",
    bgcolor: "action.hover",
    "&:hover": {
        bgcolor: "action.selected",
    },
} as const;

const bookDatasetModeButtonSx = {
    width: 28,
    height: 28,
    borderRadius: "7px",
} as const;

const compactSliderSx = (color: string) => ({
    color,
    py: 0.25,
    "& .MuiSlider-rail, & .MuiSlider-track": {
        height: 4,
    },
    "& .MuiSlider-thumb": {
        width: 14,
        height: 14,
    },
});

export const BookDatasetPage: React.FC = () => {
    const navigate = useNavigate();
    const datasetsPath = getRoutePath("finetuning-datasets");

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
    const [preview, setPreview] = useState<GeneratedExample[]>([]);
    const [loading, setLoading] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    // Generation mode state
    const [generationMode, setGenerationMode] = useState<GenerationMode>("continuation");

    // Distillation-specific state
    const [teacherProvider, setTeacherProvider] = useState("openai");
    const [teacherModel, setTeacherModel] = useState("gpt-4o-mini");
    const [teacherApiKey, setTeacherApiKey] = useState("");
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

    const buildDistillationRequest = useCallback((datasetName: string) => ({
        name: datasetName,
        description,
        domain_id: selectedDomainId,
        document_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
        generation_mode: distillationMode,
        teacher_provider: teacherProvider,
        teacher_model: teacherModel,
        teacher_api_key: teacherApiKey || undefined,
        questions_per_chunk: questionsPerChunk,
        include_source_context: true,
        max_chunks: maxChunks,
    }), [
        description,
        selectedDomainId,
        selectedDocIds,
        distillationMode,
        teacherProvider,
        teacherModel,
        teacherApiKey,
        questionsPerChunk,
        maxChunks,
    ]);

    const fetchDistillEstimate = useCallback(async () => {
        if (!selectedDomainId) return;

        setLoading(true);
        try {
            const res = await fetch("/api/v1/finetune/datasets/book/distill/estimate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    domain_id: selectedDomainId,
                    document_ids: selectedDocIds.length > 0 ? selectedDocIds : null,
                    generation_mode: distillationMode,
                    teacher_model: teacherModel,
                    questions_per_chunk: questionsPerChunk,
                    max_chunks: maxChunks,
                }),
            });
            if (res.ok) {
                const data = await res.json();
                setDistillEstimate(data);
            }
        } catch {
            // Ignore estimate errors
        } finally {
            setLoading(false);
        }
    }, [selectedDomainId, selectedDocIds, distillationMode, teacherModel, questionsPerChunk, maxChunks]);

    useEffect(() => {
        if (generationMode === "continuation") {
            fetchEstimate();
        } else {
            fetchDistillEstimate();
        }
    }, [generationMode, fetchEstimate, fetchDistillEstimate]);

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

    const fetchDistillationPreview = useCallback(async () => {
        if (!selectedDomainId) return;

        setLoading(true);
        setError(null);
        try {
            const res = await fetch("/api/v1/finetune/datasets/book/distill/preview?num_chunks=1", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(buildDistillationRequest(name || "distillation-preview")),
            });
            if (res.ok) {
                const data = await res.json();
                setPreview(data.examples || []);
            } else {
                const errData = await res.json();
                setError(errData.detail || "Failed to generate distillation preview");
            }
        } catch {
            setError("Failed to generate distillation preview");
        } finally {
            setLoading(false);
        }
    }, [selectedDomainId, buildDistillationRequest, name]);

    const handlePreview = () => {
        if (generationMode === "distillation") {
            fetchDistillationPreview();
        } else {
            fetchPreview();
        }
    };

    // Generate dataset
    const handleGenerate = async () => {
        if (!selectedDomainId || !name) {
            setError("Please select a domain and provide a dataset name");
            return;
        }

        if (generationMode === "distillation") {
            setGenerating(true);
            setError(null);
            setSuccess(null);

            try {
                const res = await fetch("/api/v1/finetune/datasets/book/distill", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(buildDistillationRequest(name)),
                });

                if (res.ok) {
                    const data = await res.json();
                    setDistillJob(data.job);
                    setSuccess("Book distillation job started. Progress will update here.");
                } else {
                    const errData = await res.json();
                    setError(errData.detail || "Failed to start distillation job");
                }
            } catch {
                setError("Failed to start distillation job");
            } finally {
                setGenerating(false);
            }
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

    const handleCancelDistillationJob = async () => {
        if (!distillJob) return;

        try {
            await fetch(`/api/v1/finetune/datasets/book/distill/${distillJob.id}/cancel`, { method: "POST" });
            setSuccess("Distillation cancellation requested");
        } catch {
            setError("Failed to cancel distillation job");
        }
    };

    useEffect(() => {
        const jobId = distillJob?.id;
        if (!jobId || !["pending", "running"].includes(distillJob.status)) return;

        const refreshJob = async () => {
            try {
                const res = await fetch(`/api/v1/finetune/datasets/book/distill/${jobId}`);
                if (!res.ok) return;
                const data = await res.json();
                setDistillJob(data.job);
                if (data.job?.status === "completed") {
                    setSuccess("Book distillation completed and saved as a dataset.");
                }
                if (data.job?.status === "failed") {
                    setError(data.job.error || "Book distillation failed");
                }
            } catch {
                // Keep the current job state if polling fails once.
            }
        };

        const interval = setInterval(refreshJob, 3000);
        refreshJob();
        return () => clearInterval(interval);
    }, [distillJob?.id, distillJob?.status]);

    const selectedDomain = domains.find((domain) => domain.id === selectedDomainId);
    const selectedDocumentCount = selectedDocIds.length > 0 ? selectedDocIds.length : documents.length;
    const estimateExamples = generationMode === "continuation"
        ? estimate?.estimated_examples
        : distillEstimate?.estimated_examples;
    const costLabel = generationMode === "distillation"
        ? `$${(distillEstimate?.estimated_cost_usd ?? 0).toFixed(2)}`
        : "Local";
    const activeModeStatus = generationMode === "continuation" ? "continuation" : "distillation";
    const runStatus = generating
        ? "generating"
        : distillJob?.status || (preview.length > 0 ? "previewed" : "draft");
    const showRunStatusChip = generating || Boolean(distillJob) || preview.length > 0;
    const hasEstimatedExamples = Boolean(estimateExamples && estimateExamples > 0);
    const previewPromptFor = (example: GeneratedExample) =>
        example.prompt || [example.instruction, example.input].filter(Boolean).join("\n\n") || "No prompt";
    const previewCompletionFor = (example: GeneratedExample) =>
        example.completion || example.output || example.response || "";
    const showPreviewPanel = Boolean(selectedDomainId) || preview.length > 0 || loading || generating || Boolean(distillJob);
    const setupPanelLayoutSx: SxProps<Theme> = showPreviewPanel
        ? { display: "flex", flexDirection: "column", gap: 1.25 }
        : {
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "minmax(300px, 0.72fr) minmax(0, 1.28fr)" },
            gap: 1.25,
            alignItems: "start",
        };

    return (
        <PageShell sx={{ py: 1.25 }}>
            {/* Alerts */}
            {error && (
                <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}
            {success && (
                <Alert severity="success" sx={{ mb: 1.25 }} onClose={() => setSuccess(null)}>
                    {success}
                </Alert>
            )}

            <Grid container spacing={2}>
                {/* Configuration Panel */}
                <Grid item xs={12} md={showPreviewPanel ? 5 : 12} lg={showPreviewPanel ? 5 : 12}>
                    <Box sx={setupPanelLayoutSx}>
                        {/* Source Selection */}
                        <SectionCard
                            title="Source Selection"
                            icon={<BookIcon />}
                            accentColor={accentColors.info}
                            action={
                                <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                    <Tooltip title="Back to datasets">
                                        <IconButton
                                            size="small"
                                            aria-label="Back to datasets"
                                            onClick={() => navigate(datasetsPath)}
                                            sx={bookDatasetIconButtonSx}
                                        >
                                            <BackIcon sx={{ fontSize: 17 }} />
                                        </IconButton>
                                    </Tooltip>
                                    <Box
                                        sx={{
                                            display: "flex",
                                            gap: 0.25,
                                            p: 0.25,
                                            borderRadius: "8px",
                                            border: "1px solid",
                                            borderColor: "divider",
                                            bgcolor: "action.hover",
                                        }}
                                    >
                                        <Tooltip title="Continuation examples from book chunks">
                                            <IconButton
                                                size="small"
                                                aria-label="Continuation mode"
                                                onClick={() => setGenerationMode("continuation")}
                                                sx={{
                                                    ...bookDatasetModeButtonSx,
                                                    bgcolor: generationMode === "continuation" ? accentColors.info : "transparent",
                                                    color: generationMode === "continuation" ? "white" : accentColors.info,
                                                    "&:hover": {
                                                        bgcolor: generationMode === "continuation" ? accentColors.info : alpha(accentColors.info, 0.12),
                                                    },
                                                }}
                                            >
                                                <TextIcon sx={{ fontSize: 16 }} />
                                            </IconButton>
                                        </Tooltip>
                                        <Tooltip title="Teacher-generated Q&A, summaries, or explanations">
                                            <IconButton
                                                size="small"
                                                aria-label="Distillation mode"
                                                onClick={() => setGenerationMode("distillation")}
                                                sx={{
                                                    ...bookDatasetModeButtonSx,
                                                    bgcolor: generationMode === "distillation" ? accentColors.rose : "transparent",
                                                    color: generationMode === "distillation" ? "white" : accentColors.rose,
                                                    "&:hover": {
                                                        bgcolor: generationMode === "distillation" ? accentColors.rose : alpha(accentColors.rose, 0.12),
                                                    },
                                                }}
                                            >
                                                <QAIcon sx={{ fontSize: 16 }} />
                                            </IconButton>
                                        </Tooltip>
                                    </Box>
                                </Box>
                            }
                        >
                            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25 }}>
                                <Box sx={{ display: "flex", alignItems: "center", gap: 0.6, flexWrap: "wrap" }}>
                                    <StatusChip status={activeModeStatus} statusConfig={BOOK_DATASET_STATUS_CONFIG} height={22} />
                                    {showRunStatusChip && <StatusChip status={runStatus} statusConfig={BOOK_DATASET_STATUS_CONFIG} height={22} />}
                                    {selectedDomain && (
                                        <StatusChip
                                            status="books"
                                            label={selectedDomain.name}
                                            statusConfig={BOOK_DATASET_STATUS_CONFIG}
                                            height={22}
                                        />
                                    )}
                                    {selectedDocumentCount > 0 && (
                                        <StatusChip
                                            status="books"
                                            label={`${selectedDocumentCount} docs`}
                                            statusConfig={BOOK_DATASET_STATUS_CONFIG}
                                            height={22}
                                        />
                                    )}
                                    {hasEstimatedExamples && (
                                        <StatusChip
                                            status="previewed"
                                            label={`${estimateExamples?.toLocaleString()} examples`}
                                            statusConfig={BOOK_DATASET_STATUS_CONFIG}
                                            height={22}
                                        />
                                    )}
                                    {selectedDomain && (
                                        <Typography variant="caption" color="text.secondary" sx={{ minWidth: 0 }}>
                                            {generationMode === "distillation" ? `${teacherModel} / ${costLabel}` : `${contextChunks} context / ${outputChunks} output / ${costLabel}`}
                                        </Typography>
                                    )}
                                </Box>
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

                        {/* Distillation Config - Only show when distillation mode is selected */}
                        {generationMode === "distillation" && (
                            <SectionCard
                                title="Distillation Settings"
                                icon={<QAIcon />}
                                accentColor={accentColors.rose}
                            >
                                <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
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
                                        placeholder="gpt-4o-mini"
                                    />

                                    {teacherProvider !== "local" && (
                                        <TextField
                                            label="API Key (optional)"
                                            value={teacherApiKey}
                                            onChange={(e) => setTeacherApiKey(e.target.value)}
                                            size="small"
                                            fullWidth
                                            type="password"
                                            placeholder="Use server key when empty"
                                        />
                                    )}

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
                                            size="small"
                                            sx={compactSliderSx(accentColors.rose)}
                                        />
                                    </Box>

                                    <TextField
                                        label="Max Chunks (optional)"
                                        value={maxChunks ?? ""}
                                        onChange={(e) => setMaxChunks(e.target.value ? parseInt(e.target.value) : null)}
                                        size="small"
                                        fullWidth
                                        type="number"
                                        placeholder="All chunks"
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
                                icon={<ConfigIcon />}
                                accentColor={accentColors.purple}
                            >
                                <Box
                                    sx={{
                                        display: "grid",
                                        gridTemplateColumns: { xs: "1fr", sm: "repeat(2, minmax(0, 1fr))" },
                                        gap: 1,
                                    }}
                                >
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
                                        rows={1}
                                    />

                                    <Box
                                        sx={{
                                            gridColumn: "1 / -1",
                                            display: "grid",
                                            gridTemplateColumns: { xs: "1fr", sm: "repeat(3, minmax(0, 1fr))" },
                                            gap: 1,
                                        }}
                                    >
                                        <Box sx={{ px: 0.5 }}>
                                            <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                                Context {contextChunks}
                                            </Typography>
                                            <Slider
                                                value={contextChunks}
                                                onChange={(_, v) => setContextChunks(v as number)}
                                                min={1}
                                                max={10}
                                                step={1}
                                                size="small"
                                                sx={compactSliderSx(accentColors.purple)}
                                            />
                                        </Box>

                                        <Box sx={{ px: 0.5 }}>
                                            <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                                Output {outputChunks}
                                            </Typography>
                                            <Slider
                                                value={outputChunks}
                                                onChange={(_, v) => setOutputChunks(v as number)}
                                                min={1}
                                                max={5}
                                                step={1}
                                                size="small"
                                                sx={compactSliderSx(accentColors.purple)}
                                            />
                                        </Box>

                                        <Box sx={{ px: 0.5 }}>
                                            <Typography variant="caption" sx={{ color: "text.secondary" }}>
                                                Stride {stride}
                                            </Typography>
                                            <Slider
                                                value={stride}
                                                onChange={(_, v) => setStride(v as number)}
                                                min={1}
                                                max={5}
                                                step={1}
                                                size="small"
                                                sx={compactSliderSx(accentColors.purple)}
                                            />
                                        </Box>
                                    </Box>

                                    <FormControlLabel
                                        sx={{ minHeight: 40, m: 0 }}
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
                                        label="Max Examples (optional)"
                                        type="number"
                                        value={maxExamples || ""}
                                        onChange={(e) => setMaxExamples(e.target.value ? parseInt(e.target.value) : null)}
                                        size="small"
                                        fullWidth
                                        placeholder="All examples"
                                    />

                                    <TextField
                                        label="System Prompt (optional)"
                                        value={systemPrompt}
                                        onChange={(e) => setSystemPrompt(e.target.value)}
                                        size="small"
                                        fullWidth
                                        multiline
                                        rows={1}
                                        placeholder="Custom system prompt prefix..."
                                        sx={{ gridColumn: "1 / -1" }}
                                    />
                                </Box>
                            </SectionCard>
                        )}

                        {/* Estimate */}
                        {generationMode === "continuation" && estimate && (
                            <SectionCard
                                title="Estimate"
                                subtitle="Predicted dataset size"
                                icon={<EstimateIcon />}
                                accentColor={accentColors.success}
                                sx={!showPreviewPanel ? { gridColumn: { md: "1 / -1" } } : undefined}
                            >
                                <Grid container spacing={2}>
                                    <Grid item xs={6}>
                                        <MetricTile
                                            variant="centered"
                                            label="Documents"
                                            value={estimate.valid_documents}
                                            icon={<BookIcon />}
                                            color={accentColors.info}
                                        />
                                    </Grid>
                                    <Grid item xs={6}>
                                        <MetricTile
                                            variant="centered"
                                            label="Total Chunks"
                                            value={estimate.total_chunks.toLocaleString()}
                                            icon={<TextIcon />}
                                            color={accentColors.purple}
                                        />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <MetricTile
                                            variant="centered"
                                            label="Estimated Examples"
                                            value={estimate.estimated_examples.toLocaleString()}
                                            icon={<DatasetIcon />}
                                            color={accentColors.success}
                                        />
                                    </Grid>
                                </Grid>
                            </SectionCard>
                        )}

                    </Box>
                </Grid>

                {/* Preview Panel */}
                {showPreviewPanel && (
                <Grid item xs={12} md={7}>
                    <SectionCard
                        title="Preview"
                        subtitle={preview.length > 0 ? `${preview.length} example(s)` : undefined}
                        icon={<PreviewIcon />}
                        accentColor={accentColors.warning}
                        action={
                            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                <Tooltip title="Preview examples">
                                    <span>
                                        <IconButton
                                            size="small"
                                            aria-label="Preview examples"
                                            onClick={handlePreview}
                                            disabled={!selectedDomainId || loading}
                                            sx={bookDatasetIconButtonSx}
                                        >
                                            {loading ? <CircularProgress size={15} /> : <PreviewIcon sx={{ fontSize: 17 }} />}
                                        </IconButton>
                                    </span>
                                </Tooltip>
                                <Tooltip title="Generate dataset">
                                    <span>
                                        <IconButton
                                            size="small"
                                            aria-label="Generate dataset"
                                            onClick={handleGenerate}
                                            disabled={!selectedDomainId || !name || generating}
                                            sx={{
                                                ...bookDatasetIconButtonSx,
                                                bgcolor: "primary.main",
                                                color: "primary.contrastText",
                                                borderColor: "primary.main",
                                                "&:hover": { bgcolor: "primary.dark" },
                                                "&.Mui-disabled": {
                                                    bgcolor: "action.disabledBackground",
                                                    color: "action.disabled",
                                                    borderColor: "divider",
                                                },
                                            }}
                                        >
                                            {generating ? <CircularProgress size={15} color="inherit" /> : <GenerateIcon sx={{ fontSize: 17 }} />}
                                        </IconButton>
                                    </span>
                                </Tooltip>
                                {distillJob && ["pending", "running"].includes(distillJob.status) && (
                                    <Tooltip title="Cancel distillation job">
                                        <IconButton
                                            size="small"
                                            color="error"
                                            aria-label="Cancel distillation job"
                                            onClick={handleCancelDistillationJob}
                                            sx={bookDatasetIconButtonSx}
                                        >
                                            <CancelIcon sx={{ fontSize: 17 }} />
                                        </IconButton>
                                    </Tooltip>
                                )}
                            </Box>
                        }
                    >
                        {loading && <LinearProgress sx={{ mb: 2 }} />}

                        {preview.length === 0 ? (
                            <Box
                                sx={{
                                    minHeight: 68,
                                    px: 1.25,
                                    py: 1,
                                    borderRadius: "8px",
                                    border: "1px solid",
                                    borderColor: "divider",
                                    bgcolor: alpha("#000", 0.12),
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    gap: 1,
                                }}
                            >
                                <Box sx={{ minWidth: 0 }}>
                                    <Typography variant="body2" sx={{ fontWeight: 800 }}>
                                        No preview yet
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        {selectedDomainId ? "Preview examples before generating the dataset." : "Select a domain to preview examples."}
                                    </Typography>
                                </Box>
                                <StatusChip
                                    status={selectedDomainId ? "draft" : "books"}
                                    label={selectedDomainId ? "Ready" : "No domain"}
                                    statusConfig={BOOK_DATASET_STATUS_CONFIG}
                                />
                            </Box>
                        ) : (
                            <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                                {preview.map((example, idx) => {
                                    const promptText = previewPromptFor(example);
                                    const completionText = previewCompletionFor(example);

                                    return (
                                        <Box
                                            key={idx}
                                            sx={{
                                                p: 2,
                                                ...bookDatasetPanelSx,
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
                                                        borderRadius: "8px",
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
                                                        {promptText.length > 500 ? `${promptText.slice(0, 500)}...` : promptText}
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
                                                        borderRadius: "8px",
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
                                                        {completionText.length > 300 ? `${completionText.slice(0, 300)}...` : completionText}
                                                    </Typography>
                                                </Box>
                                            </Box>
                                        </Box>
                                    );
                                })}
                            </Box>
                        )}
                    </SectionCard>
                </Grid>
                )}
            </Grid>
        </PageShell>
    );
};

export default BookDatasetPage;
