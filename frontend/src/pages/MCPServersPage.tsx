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
    Paper,
    Grid,
    Card,
    CardContent,
    CardActions,
    Chip,
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
} from '@mui/icons-material';
import { mcpService } from '@/services/mcp';
import type { MCPServer, PopularMCPServer, MCPTool, MCPResource, TransportType } from '@/types/mcp';

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

export default function MCPServersPage() {
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
            console.error('Failed to load popular servers:', err);
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
            console.error('Failed to load server details:', err);
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
                await loadServers();
                if (selectedServer?.name === server.name) {
                    await loadServerDetails({ ...server, connected: true });
                }
            }
        } catch (err: any) {
            console.error('Failed to connect:', err);
        }
    };

    // Disconnect from server
    const handleDisconnect = async (server: MCPServer) => {
        try {
            await mcpService.disconnectServer(server.name);
            await loadServers();
            if (selectedServer?.name === server.name) {
                setServerTools([]);
                setServerResources([]);
            }
        } catch (err: any) {
            console.error('Failed to disconnect:', err);
        }
    };

    // Delete server
    const handleDelete = async (server: MCPServer) => {
        if (!confirm(`Are you sure you want to delete "${server.name}"?`)) return;

        try {
            await mcpService.deleteServer(server.name);
            await loadServers();
            if (selectedServer?.name === server.name) {
                setSelectedServer(null);
            }
        } catch (err: any) {
            console.error('Failed to delete:', err);
        }
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
            console.error('Failed to add server:', err);
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
            console.error('Failed to add popular server:', err);
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

    return (
        <Box sx={{ p: 3, height: '100%', overflow: 'auto' }}>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="h4" fontWeight="bold">
                        MCP Servers
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        Manage Model Context Protocol server connections for extended tool capabilities
                    </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                        variant="outlined"
                        startIcon={<RefreshIcon />}
                        onClick={loadServers}
                        disabled={loading}
                    >
                        Refresh
                    </Button>
                    <Button
                        variant="contained"
                        startIcon={<AddIcon />}
                        onClick={() => setAddDialogOpen(true)}
                    >
                        Add Server
                    </Button>
                </Box>
            </Box>

            {/* Error Alert */}
            {error && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
                    {error}
                </Alert>
            )}

            {/* Main Content */}
            <Grid container spacing={3}>
                {/* Server List */}
                <Grid item xs={12} md={selectedServer ? 5 : 12}>
                    {loading ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                            <CircularProgress />
                        </Box>
                    ) : servers.length === 0 ? (
                        <Paper sx={{ p: 4, textAlign: 'center' }}>
                            <Typography variant="h6" color="text.secondary" gutterBottom>
                                No MCP Servers Configured
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                Add an MCP server to extend your LLM's capabilities with external tools
                            </Typography>
                            <Button
                                variant="contained"
                                startIcon={<AddIcon />}
                                onClick={() => setAddDialogOpen(true)}
                            >
                                Add Your First Server
                            </Button>
                        </Paper>
                    ) : (
                        <Grid container spacing={2}>
                            {servers.map(server => (
                                <Grid item xs={12} sm={selectedServer ? 12 : 6} lg={selectedServer ? 12 : 4} key={server.name}>
                                    <Card
                                        sx={{
                                            cursor: 'pointer',
                                            border: selectedServer?.name === server.name ? 2 : 1,
                                            borderColor: selectedServer?.name === server.name ? 'primary.main' : 'divider',
                                            transition: 'all 0.2s',
                                            '&:hover': {
                                                borderColor: 'primary.main',
                                                boxShadow: 2,
                                            },
                                        }}
                                        onClick={() => handleSelectServer(server)}
                                    >
                                        <CardContent>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                                <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
                                                    {server.icon || '🔧'} {server.name}
                                                </Typography>
                                                <Chip
                                                    size="small"
                                                    icon={server.connected ? <ConnectedIcon /> : <ErrorIcon />}
                                                    label={server.connected ? 'Connected' : 'Disconnected'}
                                                    color={server.connected ? 'success' : 'default'}
                                                    sx={{ ml: 'auto' }}
                                                />
                                            </Box>

                                            {server.description && (
                                                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                                    {server.description}
                                                </Typography>
                                            )}

                                            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                                <Chip
                                                    size="small"
                                                    icon={<TransportIcon transport={server.transport} />}
                                                    label={server.transport.toUpperCase()}
                                                    variant="outlined"
                                                />
                                                {server.connected && server.tools_count > 0 && (
                                                    <Chip
                                                        size="small"
                                                        icon={<ToolIcon />}
                                                        label={`${server.tools_count} tools`}
                                                        variant="outlined"
                                                        color="primary"
                                                    />
                                                )}
                                                {server.latency_ms && (
                                                    <Chip
                                                        size="small"
                                                        label={`${Math.round(server.latency_ms)}ms`}
                                                        variant="outlined"
                                                    />
                                                )}
                                            </Box>

                                            {server.error && (
                                                <Alert severity="error" sx={{ mt: 1, py: 0 }}>
                                                    {server.error}
                                                </Alert>
                                            )}
                                        </CardContent>

                                        <CardActions sx={{ justifyContent: 'flex-end' }}>
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
                                            <IconButton
                                                size="small"
                                                onClick={(e) => { e.stopPropagation(); handleDelete(server); }}
                                                color="error"
                                            >
                                                <DeleteIcon />
                                            </IconButton>
                                        </CardActions>
                                    </Card>
                                </Grid>
                            ))}
                        </Grid>
                    )}
                </Grid>

                {/* Server Details Panel */}
                {selectedServer && (
                    <Grid item xs={12} md={7}>
                        <Paper sx={{ p: 2, height: 'calc(100vh - 250px)', overflow: 'auto' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <Typography variant="h6">
                                    {selectedServer.icon || '🔧'} {selectedServer.name}
                                </Typography>
                                <IconButton
                                    size="small"
                                    sx={{ ml: 'auto' }}
                                    onClick={() => setSelectedServer(null)}
                                >
                                    ✕
                                </IconButton>
                            </Box>

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
                                    <Typography color="text.secondary">No tools available</Typography>
                                ) : (
                                    <List dense>
                                        {serverTools.map(tool => (
                                            <Box key={tool.name}>
                                                <ListItem
                                                    button
                                                    onClick={() => toggleToolExpanded(tool.name)}
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
                                                        <Paper variant="outlined" sx={{ p: 1, mt: 0.5, bgcolor: 'background.default' }}>
                                                            <pre style={{ margin: 0, fontSize: '0.75rem', overflow: 'auto' }}>
                                                                {JSON.stringify(tool.inputSchema, null, 2)}
                                                            </pre>
                                                        </Paper>
                                                    </Box>
                                                </Collapse>
                                                <Divider />
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
                                    <Typography color="text.secondary">No resources available</Typography>
                                ) : (
                                    <List dense>
                                        {serverResources.map(resource => (
                                            <ListItem key={resource.uri}>
                                                <ListItemIcon>
                                                    <ResourceIcon />
                                                </ListItemIcon>
                                                <ListItemText
                                                    primary={resource.name || resource.uri}
                                                    secondary={resource.description || resource.uri}
                                                />
                                                {resource.mimeType && (
                                                    <Chip size="small" label={resource.mimeType} variant="outlined" />
                                                )}
                                            </ListItem>
                                        ))}
                                    </List>
                                )}
                            </TabPanel>

                            <TabPanel value={detailsTab} index={2}>
                                <List dense>
                                    <ListItem>
                                        <ListItemText primary="Transport" secondary={selectedServer.transport} />
                                    </ListItem>
                                    <Divider />
                                    {selectedServer.command && (
                                        <>
                                            <ListItem>
                                                <ListItemText
                                                    primary="Command"
                                                    secondary={`${selectedServer.command} ${selectedServer.args?.join(' ') || ''}`}
                                                />
                                                <ListItemSecondaryAction>
                                                    <IconButton
                                                        size="small"
                                                        onClick={() => navigator.clipboard.writeText(
                                                            `${selectedServer.command} ${selectedServer.args?.join(' ') || ''}`
                                                        )}
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
                                                    <IconButton size="small" onClick={() => navigator.clipboard.writeText(selectedServer.url || '')}>
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
                        </Paper>
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
                                <Button onClick={() => setSelectedPopular(null)} sx={{ mb: 2 }}>
                                    ← Back to list
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
                                                <Chip key={tool} size="small" label={tool} variant="outlined" />
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
                                        <Card
                                            sx={{ cursor: 'pointer', '&:hover': { borderColor: 'primary.main' } }}
                                            variant="outlined"
                                            onClick={() => setSelectedPopular(server)}
                                        >
                                            <CardContent>
                                                <Typography variant="h6" sx={{ fontSize: '1rem' }}>
                                                    {server.icon} {server.name}
                                                </Typography>
                                                <Typography variant="body2" color="text.secondary">
                                                    {server.description}
                                                </Typography>
                                                <Box sx={{ mt: 1 }}>
                                                    <Chip size="small" label={`${server.tools.length} tools`} variant="outlined" />
                                                </Box>
                                            </CardContent>
                                        </Card>
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
        </Box>
    );
}
