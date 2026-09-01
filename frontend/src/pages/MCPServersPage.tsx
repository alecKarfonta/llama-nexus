/**
 * MCP Servers Page
 * 
 * Management interface for MCP (Model Context Protocol) server connections.
 * Supports adding/removing servers, viewing tools/resources, and testing connections.
 */

import { useState, useEffect, useCallback } from 'react';
import {
    Box,
    Typography,
    Button,
    Grid,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Alert,
    CircularProgress,
    Tooltip,
    Tabs,
    Tab,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    ListItemSecondaryAction,
    Divider,
    Collapse,
    Switch,
    FormControlLabel,
} from '@mui/material';
import { alpha } from '@mui/material/styles';
import {
    Add as AddIcon,
    Refresh as RefreshIcon,
    Delete as DeleteIcon,
    PlayArrow as ConnectIcon,
    Stop as DisconnectIcon,
    Extension as ToolIcon,
    Storage as ResourceIcon,
    CheckCircle as ConnectedIcon,
    Error as ErrorIcon,
    Link as LinkIcon,
    ExpandMore as ExpandIcon,
    ExpandLess as CollapseIcon,
    ContentCopy as CopyIcon,
    Terminal as TerminalIcon,
    Language as HttpIcon,
    Close as CloseIcon,
    ArrowBack as BackIcon,
} from '@mui/icons-material';
import {
    ConfirmationDialog,
    PageShell,
    SectionCard,
    StatusChip,
    type StatusChipConfigMap,
} from '@/components/common';
import { useClipboard } from '@/hooks/useClipboard';
import { useConfirmableAction } from '@/hooks/useConfirmableAction';
import { mcpService } from '@/services/mcp';
import type { MCPServer, PopularMCPServer, MCPTool, MCPResource, TransportType } from '@/types/mcp';
import { accentColors } from '@/utils/theme';
import { debugLog } from '@/utils/debug';

// Tab panel component
interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    return (
        <div role="tabpanel" hidden={value !== index} {...other}>
            {value === index && <Box sx={{ py: 2 }}>{children}</Box>}
        </div>
    );
}

const MCP_STATUS_CONFIG: StatusChipConfigMap = {
    catalog: { color: accentColors.purple, label: 'Catalog' },
    checking: { color: accentColors.info, label: 'Checking servers' },
    connected: { color: accentColors.success, label: 'Connected' },
    disconnected: { color: accentColors.warning, label: 'Disconnected' },
    empty: { color: accentColors.rose, label: 'No servers' },
    http: { color: accentColors.info, label: 'HTTP' },
    latency: { color: accentColors.warning, label: 'Latency' },
    mime: { color: accentColors.slate, label: 'MIME' },
    resource: { color: accentColors.success, label: 'Resource' },
    resources: { color: accentColors.success, label: 'Resources' },
    schema: { color: accentColors.primary, label: 'Schema' },
    stdio: { color: accentColors.warning, label: 'STDIO' },
    tool: { color: accentColors.primary, label: 'Tool' },
    tool_servers: { color: accentColors.rose, label: 'Tool servers' },
    tools: { color: accentColors.primary, label: 'Tools' },
    transport: { color: accentColors.info, label: 'Transport' },
    unavailable: { color: accentColors.warning, label: 'API unavailable' },
};

const mcpPanelSx = {
    borderRadius: '8px',
    bgcolor: 'background.paper',
    border: '1px solid',
    borderColor: 'divider',
} as const;

export default function MCPServersPage() {
    const copyToClipboard = useClipboard();

    // State
    const [servers, setServers] = useState<MCPServer[]>([]);
    const [popularServers, setPopularServers] = useState<PopularMCPServer[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [selectedServer, setSelectedServer] = useState<MCPServer | null>(null);
    const [serverTools, setServerTools] = useState<MCPTool[]>([]);
    const [serverResources, setServerResources] = useState<MCPResource[]>([]);
    const [detailsTab, setDetailsTab] = useState(0);
    const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());

    // Dialog state
    const [addDialogOpen, setAddDialogOpen] = useState(false);
    const [addDialogTab, setAddDialogTab] = useState(0);
    const {
        confirmationAction: confirmAction,
        confirmationLoading: confirmingAction,
        requestConfirmation,
        closeConfirmation: closeConfirmDialog,
        confirmAction: handleConfirmAction,
    } = useConfirmableAction();
    const [newServerName, setNewServerName] = useState('');
    const [newServerTransport, setNewServerTransport] = useState<TransportType>('stdio');
    const [newServerCommand, setNewServerCommand] = useState('');
    const [newServerArgs, setNewServerArgs] = useState('');
    const [newServerUrl, setNewServerUrl] = useState('');
    const [newServerDescription, setNewServerDescription] = useState('');
    const [newServerAutoConnect, setNewServerAutoConnect] = useState(true);
    const [addingServer, setAddingServer] = useState(false);

    // Popular server dialog state
    const [selectedPopular, setSelectedPopular] = useState<PopularMCPServer | null>(null);
    const [popularPath, setPopularPath] = useState('');
    const [popularEnvVars, setPopularEnvVars] = useState<Record<string, string>>({});

    // Load servers
    const loadServers = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const serverList = await mcpService.listServers();
            setServers(serverList);
        } catch (err: any) {
            setError(err.message || 'Failed to load MCP servers');
        } finally {
            setLoading(false);
        }
    }, []);

    // Load popular servers
    const loadPopularServers = useCallback(async () => {
        try {
            const response = await mcpService.listPopularServers();
            setPopularServers(response.servers);
        } catch (err) {
            debugLog('MCPServersPage', 'Failed to load popular servers:', err);
        }
    }, []);

    useEffect(() => {
        loadServers();
        loadPopularServers();
    }, [loadServers, loadPopularServers]);

    // Load server details (tools/resources)
    const loadServerDetails = async (server: MCPServer) => {
        if (!server.connected) return;

        try {
            const [toolsResponse, resourcesResponse] = await Promise.all([
                mcpService.listServerTools(server.name),
                mcpService.listServerResources(server.name).catch(() => ({ resources: [] })),
            ]);
            setServerTools(toolsResponse.tools);
            setServerResources(resourcesResponse.resources);
        } catch (err) {
            debugLog('MCPServersPage', 'Failed to load server details:', err);
        }
    };

    // Select a server
    const handleSelectServer = async (server: MCPServer) => {
        setSelectedServer(server);
        setServerTools([]);
        setServerResources([]);
        setDetailsTab(0);
        await loadServerDetails(server);
    };

    // Connect to server
    const handleConnect = async (server: MCPServer) => {
        try {
            const result = await mcpService.connectServer(server.name);
            if (result.connected) {
                const updatedServer: MCPServer = {
                    ...server,
                    connected: true,
                    tools_count: result.tools_count ?? server.tools_count,
                    resources_count: result.resources_count ?? server.resources_count,
                    latency_ms: result.latency_ms ?? server.latency_ms,
                    error: null,
                };
                if (selectedServer?.name === server.name) {
                    setSelectedServer(updatedServer);
                    await loadServerDetails(updatedServer);
                }
                await loadServers();
            } else if (result.error) {
                setError(result.error);
            }
        } catch (err: any) {
            setError(err.message || `Failed to connect ${server.name}`);
        }
    };

    // Disconnect from server
    const handleDisconnect = async (server: MCPServer) => {
        try {
            await mcpService.disconnectServer(server.name);
            await loadServers();
            if (selectedServer?.name === server.name) {
                setSelectedServer({ ...server, connected: false });
                setServerTools([]);
                setServerResources([]);
            }
        } catch (err: any) {
            setError(err.message || `Failed to disconnect ${server.name}`);
        }
    };

    // Delete server
    const handleDelete = (server: MCPServer) => {
        requestConfirmation({
            title: 'Delete MCP server?',
            message: `Delete "${server.name}" from your MCP server list?`,
            confirmLabel: 'Delete server',
            destructive: true,
            onConfirm: async () => {
                try {
                    await mcpService.deleteServer(server.name);
                    await loadServers();
                    if (selectedServer?.name === server.name) {
                        setSelectedServer(null);
                    }
                } catch (err: any) {
                    setError(err.message || `Failed to delete ${server.name}`);
                }
            },
        });
    };

    // Add custom server
    const handleAddCustomServer = async () => {
        try {
            setAddingServer(true);
            const args = newServerArgs.split(' ').filter(a => a.trim());

            await mcpService.addServer({
                name: newServerName,
                transport: newServerTransport,
                command: newServerTransport === 'stdio' ? newServerCommand : undefined,
                args: newServerTransport === 'stdio' ? args : undefined,
                url: newServerTransport === 'http' ? newServerUrl : undefined,
                description: newServerDescription || undefined,
                auto_connect: newServerAutoConnect,
            });

            setAddDialogOpen(false);
            resetAddForm();
            await loadServers();
        } catch (err: any) {
            setError(err.message || 'Failed to add MCP server');
        } finally {
            setAddingServer(false);
        }
    };

    // Add popular server
    const handleAddPopularServer = async () => {
        if (!selectedPopular) return;

        try {
            setAddingServer(true);
            await mcpService.addPopularServer({
                server_id: selectedPopular.id,
                path: popularPath || undefined,
                env_vars: Object.keys(popularEnvVars).length > 0 ? popularEnvVars : undefined,
            });

            setAddDialogOpen(false);
            setSelectedPopular(null);
            setPopularPath('');
            setPopularEnvVars({});
            await loadServers();
        } catch (err: any) {
            setError(err.message || 'Failed to add popular MCP server');
        } finally {
            setAddingServer(false);
        }
    };

    // Reset add form
    const resetAddForm = () => {
        setNewServerName('');
        setNewServerTransport('stdio');
        setNewServerCommand('');
        setNewServerArgs('');
        setNewServerUrl('');
        setNewServerDescription('');
        setNewServerAutoConnect(true);
        setAddDialogTab(0);
        setSelectedPopular(null);
        setPopularPath('');
        setPopularEnvVars({});
    };

    // Toggle tool expansion
    const toggleToolExpanded = (toolName: string) => {
        setExpandedTools(prev => {
            const next = new Set(prev);
            if (next.has(toolName)) {
                next.delete(toolName);
            } else {
                next.add(toolName);
            }
            return next;
        });
    };

    // Get transport icon
    const TransportIcon = ({ transport }: { transport: TransportType }) => {
        return transport === 'stdio' ? <TerminalIcon fontSize="small" /> : <HttpIcon fontSize="small" />;
    };

    const connectedServers = servers.filter(server => server.connected);
    const disconnectedServers = servers.length - connectedServers.length;
    const totalTools = servers.reduce((total, server) => total + (server.tools_count || 0), 0);
    const totalResources = servers.reduce((total, server) => total + (server.resources_count || 0), 0);
    const latencySamples = connectedServers
        .map(server => server.latency_ms)
        .filter((latency): latency is number => typeof latency === 'number');
    const averageLatency = latencySamples.length
        ? `${Math.round(latencySamples.reduce((sum, latency) => sum + latency, 0) / latencySamples.length)}ms`
        : 'n/a';
    const popularPreviewServers = popularServers.slice(0, 4);
    const selectedServerCommand = selectedServer?.command
        ? `${selectedServer.command} ${selectedServer.args?.join(' ') || ''}`.trim()
        : '';
    const mcpApiUnavailable = Boolean(error && /not found|404/i.test(error));
    const serviceStatus = loading
        ? 'checking'
        : mcpApiUnavailable
            ? 'unavailable'
            : connectedServers.length > 0
                ? 'connected'
                : servers.length > 0
                    ? 'disconnected'
                    : 'empty';
    const serviceLabel = loading
        ? 'Checking servers'
        : mcpApiUnavailable
            ? 'API unavailable'
            : connectedServers.length > 0
                ? `${connectedServers.length} connected`
                : servers.length > 0
                    ? 'Disconnected'
                    : 'No servers';
    const openPopularServer = (server: PopularMCPServer) => {
        setSelectedPopular(server);
        setAddDialogTab(0);
        setAddDialogOpen(true);
    };
    const mcpHeaderActions = (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <Tooltip title="Refresh MCP servers">
                <span>
                    <IconButton
                        aria-label="Refresh MCP servers"
                        size="small"
                        onClick={loadServers}
                        disabled={loading}
                        sx={{
                            width: 34,
                            height: 34,
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: 'divider',
                            bgcolor: 'action.hover',
                            '&:hover': { bgcolor: alpha(accentColors.rose, 0.12) },
                        }}
                    >
                        {loading ? <CircularProgress size={16} /> : <RefreshIcon sx={{ fontSize: 18 }} />}
                    </IconButton>
                </span>
            </Tooltip>
            {!mcpApiUnavailable && (
                <Button
                    variant="contained"
                    size="small"
                    startIcon={<AddIcon />}
                    onClick={() => setAddDialogOpen(true)}
                    sx={{ height: 34, borderRadius: '8px', fontWeight: 800, px: 1.25 }}
                >
                    Add Server
                </Button>
            )}
        </Box>
    );
    const mcpSummaryRow = (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 1 }}>
            <StatusChip status={serviceStatus} label={serviceLabel} statusConfig={MCP_STATUS_CONFIG} height={22} />
            {servers.length > 0 && (
                <>
                    <StatusChip status="tools" label={`${totalTools.toLocaleString()} tools`} statusConfig={MCP_STATUS_CONFIG} height={22} />
                    <StatusChip status="resources" label={`${totalResources.toLocaleString()} resources`} statusConfig={MCP_STATUS_CONFIG} height={22} />
                </>
            )}
            {latencySamples.length > 0 && (
                <StatusChip status="latency" label={`Latency ${averageLatency}`} statusConfig={MCP_STATUS_CONFIG} height={22} />
            )}
            {popularServers.length > 0 && (
                <StatusChip status="catalog" label={`${popularServers.length.toLocaleString()} templates`} statusConfig={MCP_STATUS_CONFIG} height={22} />
            )}
            <Typography variant="caption" color="text.secondary">
                {mcpApiUnavailable
                    ? 'Backend route missing'
                    : `${servers.length.toLocaleString()} servers / ${disconnectedServers.toLocaleString()} offline`}
            </Typography>
        </Box>
    );

    return (
        <PageShell sx={{ py: 1.25 }}>
            {error && !mcpApiUnavailable && (
                <Alert severity="error" sx={{ mb: 1.25 }} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}

            <Grid container spacing={2.5}>
                <Grid item xs={12} lg={selectedServer ? 5 : 12}>
                    <SectionCard
                        title="Server inventory"
                        subtitle={
                            mcpApiUnavailable
                                ? 'Backend route unavailable'
                                : servers.length
                                ? `${servers.length} configured, ${connectedServers.length} connected`
                                : 'No servers configured'
                        }
                        icon={<ToolIcon />}
                        accentColor={accentColors.rose}
                        action={mcpHeaderActions}
                    >
                        {mcpSummaryRow}
                        {loading ? (
                            <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
                                <CircularProgress />
                            </Box>
                        ) : mcpApiUnavailable ? (
                            <Box
                                sx={{
                                    ...mcpPanelSx,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    gap: 1.5,
                                    flexWrap: 'wrap',
                                    p: 1.25,
                                    bgcolor: alpha(accentColors.warning, 0.06),
                                    borderColor: alpha(accentColors.warning, 0.24),
                                }}
                            >
                                <Box sx={{ minWidth: 0 }}>
                                    <Typography variant="body2" sx={{ fontWeight: 800 }}>
                                        MCP backend endpoint unavailable
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        `/api/v1/mcp/servers` returned Not Found.
                                    </Typography>
                                </Box>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    startIcon={<RefreshIcon />}
                                    onClick={loadServers}
                                    sx={{ height: 32, borderRadius: '8px', fontWeight: 800 }}
                                >
                                    Retry
                                </Button>
                            </Box>
                        ) : servers.length === 0 ? (
                            <Box
                                sx={{
                                    ...mcpPanelSx,
                                    display: 'grid',
                                    gap: 1,
                                    borderStyle: 'dashed',
                                    p: 1.25,
                                }}
                            >
                                <Box sx={{ minWidth: 0 }}>
                                    <Typography variant="body2" sx={{ fontWeight: 800 }}>
                                        No MCP servers configured
                                    </Typography>
                                    {popularPreviewServers.length > 0 && (
                                        <Typography variant="caption" color="text.secondary">
                                            Templates
                                        </Typography>
                                    )}
                                </Box>
                                {popularPreviewServers.length > 0 && (
                                    <Box
                                        sx={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            flexWrap: 'wrap',
                                            gap: 0.5,
                                        }}
                                    >
                                        {popularPreviewServers.map((server) => (
                                            <Box
                                                component="button"
                                                key={server.id}
                                                onClick={() => openPopularServer(server)}
                                                sx={{
                                                    appearance: 'none',
                                                    minWidth: 0,
                                                    flex: { xs: '1 1 calc(50% - 4px)', sm: '0 1 176px' },
                                                    maxWidth: { xs: 'calc(50% - 4px)', sm: 176 },
                                                    minHeight: 30,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'space-between',
                                                    gap: 0.65,
                                                    px: 0.85,
                                                    py: 0.45,
                                                    border: '1px solid',
                                                    borderColor: 'divider',
                                                    borderRadius: '8px',
                                                    bgcolor: 'background.paper',
                                                    color: 'text.primary',
                                                    cursor: 'pointer',
                                                    font: 'inherit',
                                                    textAlign: 'left',
                                                    '&:hover': {
                                                        borderColor: alpha(accentColors.rose, 0.42),
                                                        bgcolor: alpha(accentColors.rose, 0.07),
                                                    },
                                                    '&:focus-visible': {
                                                        outline: `2px solid ${alpha(accentColors.rose, 0.5)}`,
                                                        outlineOffset: 2,
                                                    },
                                                }}
                                            >
                                                <Box sx={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 0.65 }}>
                                                    <Typography aria-hidden sx={{ fontSize: 14, lineHeight: 1, flexShrink: 0 }}>
                                                        {server.icon}
                                                    </Typography>
                                                    <Typography component="span" variant="caption" noWrap sx={{ fontWeight: 800, color: 'text.primary' }}>
                                                        {server.name}
                                                    </Typography>
                                                </Box>
                                                <Typography component="span" variant="caption" color="text.secondary" sx={{ flexShrink: 0, fontSize: '0.68rem' }}>
                                                    {server.tools.length}
                                                </Typography>
                                            </Box>
                                        ))}
                                    </Box>
                                )}
                            </Box>
                        ) : (
                            <Box sx={{ display: 'grid', gap: 1.25 }}>
                                {servers.map(server => {
                                    const isSelected = selectedServer?.name === server.name;
                                    const rowColor = server.connected ? accentColors.success : server.error ? accentColors.rose : accentColors.warning;

                                    return (
                                        <Box
                                            key={server.name}
                                            onClick={() => handleSelectServer(server)}
                                            sx={{
                                                ...mcpPanelSx,
                                                p: 1.5,
                                                borderColor: isSelected ? alpha(accentColors.rose, 0.45) : 'divider',
                                                bgcolor: isSelected ? alpha(accentColors.rose, 0.1) : alpha(rowColor, 0.05),
                                                cursor: 'pointer',
                                                transition: 'border-color 0.2s ease, background-color 0.2s ease',
                                                '&:hover': {
                                                    borderColor: alpha(rowColor, 0.55),
                                                    bgcolor: alpha(rowColor, 0.09),
                                                },
                                            }}
                                        >
                                            <Box
                                                sx={{
                                                    display: 'flex',
                                                    alignItems: { xs: 'flex-start', sm: 'center' },
                                                    justifyContent: 'space-between',
                                                    flexDirection: { xs: 'column', sm: 'row' },
                                                    gap: 1,
                                                }}
                                            >
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0 }}>
                                                    {server.icon ? (
                                                        <Typography aria-hidden sx={{ fontSize: 20, lineHeight: 1 }}>
                                                            {server.icon}
                                                        </Typography>
                                                    ) : (
                                                        <ToolIcon sx={{ color: rowColor }} />
                                                    )}
                                                    <Box sx={{ minWidth: 0 }}>
                                                        <Typography variant="subtitle2" sx={{ fontWeight: 800 }} noWrap title={server.name}>
                                                            {server.name}
                                                        </Typography>
                                                        <Typography variant="caption" color="text.secondary" noWrap title={server.description || server.transport}>
                                                            {server.description || `${server.transport.toUpperCase()} server`}
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
                                                    <StatusChip
                                                        status={server.connected ? 'connected' : 'disconnected'}
                                                        icon={server.connected ? <ConnectedIcon /> : <ErrorIcon />}
                                                        label={server.connected ? 'Connected' : 'Disconnected'}
                                                        statusConfig={MCP_STATUS_CONFIG}
                                                    />
                                                    <StatusChip
                                                        status={server.transport}
                                                        icon={<TransportIcon transport={server.transport} />}
                                                        label={server.transport.toUpperCase()}
                                                        variant="outlined"
                                                        statusConfig={MCP_STATUS_CONFIG}
                                                    />
                                                    <StatusChip
                                                        status="tools"
                                                        icon={<ToolIcon />}
                                                        label={`${server.tools_count || 0} tools`}
                                                        variant="outlined"
                                                        statusConfig={MCP_STATUS_CONFIG}
                                                    />
                                                    {server.resources_count > 0 && (
                                                        <StatusChip
                                                            status="resources"
                                                            icon={<ResourceIcon />}
                                                            label={`${server.resources_count} resources`}
                                                            variant="outlined"
                                                            statusConfig={MCP_STATUS_CONFIG}
                                                        />
                                                    )}
                                                    {typeof server.latency_ms === 'number' && (
                                                        <StatusChip
                                                            status="latency"
                                                            label={`${Math.round(server.latency_ms)}ms`}
                                                            variant="outlined"
                                                            statusConfig={MCP_STATUS_CONFIG}
                                                        />
                                                    )}
                                                </Box>
                                            </Box>

                                            {server.error && (
                                                <Alert severity="error" sx={{ mt: 1, py: 0 }}>
                                                    {server.error}
                                                </Alert>
                                            )}

                                            <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 1 }}>
                                                {server.connected ? (
                                                    <Button
                                                        size="small"
                                                        startIcon={<DisconnectIcon />}
                                                        onClick={(e) => { e.stopPropagation(); handleDisconnect(server); }}
                                                    >
                                                        Disconnect
                                                    </Button>
                                                ) : (
                                                    <Button
                                                        size="small"
                                                        startIcon={<ConnectIcon />}
                                                        onClick={(e) => { e.stopPropagation(); handleConnect(server); }}
                                                        color="primary"
                                                    >
                                                        Connect
                                                    </Button>
                                                )}
                                                <Tooltip title="Delete server">
                                                    <IconButton
                                                        size="small"
                                                        onClick={(e) => { e.stopPropagation(); handleDelete(server); }}
                                                        color="error"
                                                    >
                                                        <DeleteIcon />
                                                    </IconButton>
                                                </Tooltip>
                                            </Box>
                                        </Box>
                                    );
                                })}
                            </Box>
                        )}
                    </SectionCard>
                </Grid>

                {selectedServer && (
                    <Grid item xs={12} lg={7}>
                        <SectionCard
                            title={selectedServer.name}
                            subtitle={selectedServer.connected ? `${serverTools.length} tools loaded` : 'Disconnected'}
                            icon={selectedServer.icon ? (
                                <Typography aria-hidden sx={{ fontSize: 22, lineHeight: 1 }}>
                                    {selectedServer.icon}
                                </Typography>
                            ) : (
                                <ToolIcon />
                            )}
                            accentColor={selectedServer.connected ? accentColors.success : accentColors.warning}
                            fullHeight
                            action={
                                <>
                                    <StatusChip
                                        status={selectedServer.connected ? 'connected' : 'disconnected'}
                                        icon={selectedServer.connected ? <ConnectedIcon /> : <ErrorIcon />}
                                        label={selectedServer.connected ? 'Connected' : 'Disconnected'}
                                        statusConfig={MCP_STATUS_CONFIG}
                                    />
                                    <Tooltip title="Close details">
                                        <IconButton size="small" onClick={() => setSelectedServer(null)}>
                                            <CloseIcon fontSize="small" />
                                        </IconButton>
                                    </Tooltip>
                                </>
                            }
                        >
                            <Tabs value={detailsTab} onChange={(_, v) => setDetailsTab(v)} sx={{ mb: 2 }}>
                                <Tab label={`Tools (${serverTools.length})`} icon={<ToolIcon />} iconPosition="start" />
                                <Tab label={`Resources (${serverResources.length})`} icon={<ResourceIcon />} iconPosition="start" />
                                <Tab label="Config" />
                            </Tabs>

                            <TabPanel value={detailsTab} index={0}>
                                {!selectedServer.connected ? (
                                    <Alert severity="info">
                                        Connect to the server to view available tools
                                    </Alert>
                                ) : serverTools.length === 0 ? (
                                    <Box sx={{ py: 4, textAlign: 'center' }}>
                                        <Typography color="text.secondary">No tools available</Typography>
                                    </Box>
                                ) : (
                                    <List dense sx={{ py: 0 }}>
                                        {serverTools.map(tool => (
                                            <Box key={tool.name}>
                                                <ListItem
                                                    button
                                                    onClick={() => toggleToolExpanded(tool.name)}
                                                    sx={{
                                                        ...mcpPanelSx,
                                                        borderRadius: '8px',
                                                        mb: 1,
                                                        bgcolor: alpha(accentColors.primary, 0.05),
                                                    }}
                                                >
                                                    <ListItemIcon>
                                                        <ToolIcon />
                                                    </ListItemIcon>
                                                    <ListItemText
                                                        primary={tool.name}
                                                        secondary={tool.description}
                                                    />
                                                    {expandedTools.has(tool.name) ? <CollapseIcon /> : <ExpandIcon />}
                                                </ListItem>
                                                <Collapse in={expandedTools.has(tool.name)}>
                                                    <Box sx={{ pl: 7, pr: 2, pb: 2 }}>
                                                        <Typography variant="caption" color="text.secondary">
                                                            Schema:
                                                        </Typography>
                                                        <Box sx={{ ...mcpPanelSx, p: 1, mt: 0.5, bgcolor: alpha(accentColors.primary, 0.04) }}>
                                                            <pre style={{ margin: 0, fontSize: '0.75rem', overflow: 'auto' }}>
                                                                {JSON.stringify(tool.inputSchema, null, 2)}
                                                            </pre>
                                                        </Box>
                                                    </Box>
                                                </Collapse>
                                            </Box>
                                        ))}
                                    </List>
                                )}
                            </TabPanel>

                            <TabPanel value={detailsTab} index={1}>
                                {!selectedServer.connected ? (
                                    <Alert severity="info">
                                        Connect to the server to view available resources
                                    </Alert>
                                ) : serverResources.length === 0 ? (
                                    <Box sx={{ py: 4, textAlign: 'center' }}>
                                        <Typography color="text.secondary">No resources available</Typography>
                                    </Box>
                                ) : (
                                    <List dense sx={{ py: 0 }}>
                                        {serverResources.map(resource => (
                                            <ListItem
                                                key={resource.uri}
                                                sx={{
                                                    ...mcpPanelSx,
                                                    borderRadius: '8px',
                                                    mb: 1,
                                                    bgcolor: alpha(accentColors.success, 0.05),
                                                }}
                                            >
                                                <ListItemIcon>
                                                    <ResourceIcon />
                                                </ListItemIcon>
                                                <ListItemText
                                                    primary={resource.name || resource.uri}
                                                    secondary={resource.description || resource.uri}
                                                />
                                                {resource.mimeType && (
                                                    <StatusChip
                                                        status="mime"
                                                        label={resource.mimeType}
                                                        variant="outlined"
                                                        statusConfig={MCP_STATUS_CONFIG}
                                                    />
                                                )}
                                            </ListItem>
                                        ))}
                                    </List>
                                )}
                            </TabPanel>

                            <TabPanel value={detailsTab} index={2}>
                                <List dense sx={{ py: 0 }}>
                                    <ListItem>
                                        <ListItemText primary="Transport" secondary={selectedServer.transport} />
                                    </ListItem>
                                    <Divider />
                                    {selectedServer.command && (
                                        <>
                                            <ListItem>
                                                <ListItemText
                                                    primary="Command"
                                                    secondary={selectedServerCommand}
                                                />
                                                <ListItemSecondaryAction>
                                                    <IconButton
                                                        size="small"
                                                        onClick={() => copyToClipboard(selectedServerCommand, {
                                                            successMessage: 'MCP command copied.',
                                                            successTitle: 'Command copied',
                                                        })}
                                                    >
                                                        <CopyIcon fontSize="small" />
                                                    </IconButton>
                                                </ListItemSecondaryAction>
                                            </ListItem>
                                            <Divider />
                                        </>
                                    )}
                                    {selectedServer.url && (
                                        <>
                                            <ListItem>
                                                <ListItemText primary="URL" secondary={selectedServer.url} />
                                                <ListItemSecondaryAction>
                                                    <IconButton
                                                        size="small"
                                                        onClick={() => copyToClipboard(selectedServer.url || '', {
                                                            successMessage: 'MCP server URL copied.',
                                                            successTitle: 'URL copied',
                                                        })}
                                                    >
                                                        <CopyIcon fontSize="small" />
                                                    </IconButton>
                                                </ListItemSecondaryAction>
                                            </ListItem>
                                            <Divider />
                                        </>
                                    )}
                                    <ListItem>
                                        <ListItemText primary="Auto-connect" secondary={selectedServer.auto_connect ? 'Yes' : 'No'} />
                                    </ListItem>
                                    <Divider />
                                    <ListItem>
                                        <ListItemText primary="Created" secondary={selectedServer.created_at || 'Unknown'} />
                                    </ListItem>
                                </List>
                            </TabPanel>
                        </SectionCard>
                    </Grid>
                )}
            </Grid>

            {/* Add Server Dialog */}
            <Dialog
                open={addDialogOpen}
                onClose={() => { setAddDialogOpen(false); resetAddForm(); }}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>Add MCP Server</DialogTitle>
                <DialogContent>
                    <Tabs value={addDialogTab} onChange={(_, v) => setAddDialogTab(v)} sx={{ mb: 2 }}>
                        <Tab label="Popular Servers" />
                        <Tab label="Custom Server" />
                    </Tabs>

                    <TabPanel value={addDialogTab} index={0}>
                        {selectedPopular ? (
                            <Box>
                                <Button startIcon={<BackIcon />} onClick={() => setSelectedPopular(null)} sx={{ mb: 2 }}>
                                    Back to list
                                </Button>
                                <Typography variant="h6" gutterBottom>
                                    {selectedPopular.icon} {selectedPopular.name}
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                    {selectedPopular.description}
                                </Typography>

                                {selectedPopular.requires_path && (
                                    <TextField
                                        fullWidth
                                        label="Path"
                                        placeholder="/path/to/directory"
                                        value={popularPath}
                                        onChange={(e) => setPopularPath(e.target.value)}
                                        sx={{ mb: 2 }}
                                        helperText="Required: Specify the path this server should access"
                                    />
                                )}

                                {selectedPopular.requires_env.length > 0 && (
                                    <Box sx={{ mb: 2 }}>
                                        <Typography variant="subtitle2" gutterBottom>
                                            Required Environment Variables:
                                        </Typography>
                                        {selectedPopular.requires_env.map(envVar => (
                                            <TextField
                                                key={envVar}
                                                fullWidth
                                                label={envVar}
                                                type="password"
                                                value={popularEnvVars[envVar] || ''}
                                                onChange={(e) => setPopularEnvVars({ ...popularEnvVars, [envVar]: e.target.value })}
                                                sx={{ mb: 1 }}
                                            />
                                        ))}
                                    </Box>
                                )}

                                {selectedPopular.tools.length > 0 && (
                                    <Box sx={{ mb: 2 }}>
                                        <Typography variant="subtitle2" gutterBottom>
                                            Available Tools:
                                        </Typography>
                                        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                            {selectedPopular.tools.map(tool => (
                                                <StatusChip
                                                    key={tool}
                                                    status="tool"
                                                    label={tool}
                                                    variant="outlined"
                                                    statusConfig={MCP_STATUS_CONFIG}
                                                />
                                            ))}
                                        </Box>
                                    </Box>
                                )}

                                {selectedPopular.docs_url && (
                                    <Button
                                        component="a"
                                        href={selectedPopular.docs_url}
                                        target="_blank"
                                        rel="noopener"
                                        startIcon={<LinkIcon />}
                                        size="small"
                                    >
                                        Documentation
                                    </Button>
                                )}
                            </Box>
                        ) : (
                            <Grid container spacing={2}>
                                {popularServers.map(server => (
                                    <Grid item xs={12} sm={6} key={server.id}>
                                        <Box
                                            onClick={() => setSelectedPopular(server)}
                                            sx={{
                                                ...mcpPanelSx,
                                                height: '100%',
                                                p: 2,
                                                cursor: 'pointer',
                                                transition: 'border-color 0.2s ease, background-color 0.2s ease',
                                                '&:hover': {
                                                    borderColor: alpha(accentColors.rose, 0.35),
                                                    bgcolor: alpha(accentColors.rose, 0.07),
                                                },
                                            }}
                                        >
                                            <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 800 }}>
                                                {server.icon} {server.name}
                                            </Typography>
                                            <Typography variant="body2" color="text.secondary">
                                                {server.description}
                                            </Typography>
                                            <Box sx={{ mt: 1 }}>
                                                <StatusChip
                                                    status="tools"
                                                    label={`${server.tools.length} tools`}
                                                    variant="outlined"
                                                    statusConfig={MCP_STATUS_CONFIG}
                                                />
                                            </Box>
                                        </Box>
                                    </Grid>
                                ))}
                            </Grid>
                        )}
                    </TabPanel>

                    <TabPanel value={addDialogTab} index={1}>
                        <Grid container spacing={2}>
                            <Grid item xs={12} sm={6}>
                                <TextField
                                    fullWidth
                                    label="Server Name"
                                    value={newServerName}
                                    onChange={(e) => setNewServerName(e.target.value)}
                                    placeholder="my-mcp-server"
                                    required
                                />
                            </Grid>
                            <Grid item xs={12} sm={6}>
                                <FormControl fullWidth>
                                    <InputLabel>Transport</InputLabel>
                                    <Select
                                        value={newServerTransport}
                                        onChange={(e) => setNewServerTransport(e.target.value as TransportType)}
                                        label="Transport"
                                    >
                                        <MenuItem value="stdio">Stdio (Local Command)</MenuItem>
                                        <MenuItem value="http">HTTP (Remote URL)</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>

                            {newServerTransport === 'stdio' ? (
                                <>
                                    <Grid item xs={12} sm={6}>
                                        <TextField
                                            fullWidth
                                            label="Command"
                                            value={newServerCommand}
                                            onChange={(e) => setNewServerCommand(e.target.value)}
                                            placeholder="npx"
                                            required
                                        />
                                    </Grid>
                                    <Grid item xs={12} sm={6}>
                                        <TextField
                                            fullWidth
                                            label="Arguments"
                                            value={newServerArgs}
                                            onChange={(e) => setNewServerArgs(e.target.value)}
                                            placeholder="-y @modelcontextprotocol/server-filesystem /tmp"
                                            helperText="Space-separated arguments"
                                        />
                                    </Grid>
                                </>
                            ) : (
                                <Grid item xs={12}>
                                    <TextField
                                        fullWidth
                                        label="URL"
                                        value={newServerUrl}
                                        onChange={(e) => setNewServerUrl(e.target.value)}
                                        placeholder="http://localhost:8000/mcp"
                                        required
                                    />
                                </Grid>
                            )}

                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label="Description"
                                    value={newServerDescription}
                                    onChange={(e) => setNewServerDescription(e.target.value)}
                                    placeholder="Optional description"
                                />
                            </Grid>

                            <Grid item xs={12}>
                                <FormControlLabel
                                    control={
                                        <Switch
                                            checked={newServerAutoConnect}
                                            onChange={(e) => setNewServerAutoConnect(e.target.checked)}
                                        />
                                    }
                                    label="Auto-connect on startup"
                                />
                            </Grid>
                        </Grid>
                    </TabPanel>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => { setAddDialogOpen(false); resetAddForm(); }}>
                        Cancel
                    </Button>
                    {addDialogTab === 0 ? (
                        <Button
                            variant="contained"
                            onClick={handleAddPopularServer}
                            disabled={!selectedPopular || addingServer}
                        >
                            {addingServer ? <CircularProgress size={20} /> : 'Add Server'}
                        </Button>
                    ) : (
                        <Button
                            variant="contained"
                            onClick={handleAddCustomServer}
                            disabled={!newServerName || addingServer}
                        >
                            {addingServer ? <CircularProgress size={20} /> : 'Add Server'}
                        </Button>
                    )}
                </DialogActions>
            </Dialog>

            <ConfirmationDialog
                action={confirmAction}
                loading={confirmingAction}
                onClose={closeConfirmDialog}
                onConfirm={handleConfirmAction}
            />
        </PageShell>
    );
}
