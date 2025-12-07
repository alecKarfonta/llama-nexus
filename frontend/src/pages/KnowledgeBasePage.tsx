import React, { useState, useCallback } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  IconButton,
  TextField,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Tooltip,
  alpha,
  LinearProgress,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Menu,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  Tabs,
  Tab,
  InputAdornment,
} from '@mui/material'
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  CloudUpload as UploadIcon,
  Search as SearchIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreIcon,
  Description as DocIcon,
  Folder as FolderIcon,
  Link as LinkIcon,
  Storage as StorageIcon,
  Psychology as EmbedIcon,
  ContentCopy as CopyIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material'

// Document interface
interface Document {
  id: string
  name: string
  type: 'pdf' | 'txt' | 'md' | 'html' | 'url'
  size: number
  chunks: number
  status: 'processing' | 'ready' | 'error'
  uploadedAt: string
  collection: string
  metadata?: Record<string, string>
}

// Collection interface
interface Collection {
  id: string
  name: string
  description: string
  documentCount: number
  totalChunks: number
  embeddingModel: string
  createdAt: string
}

// Sample data
const sampleCollections: Collection[] = [
  {
    id: 'col-1',
    name: 'Documentation',
    description: 'Product documentation and guides',
    documentCount: 12,
    totalChunks: 245,
    embeddingModel: 'nomic-embed-text',
    createdAt: '2025-12-01',
  },
  {
    id: 'col-2',
    name: 'Research Papers',
    description: 'AI/ML research papers',
    documentCount: 8,
    totalChunks: 156,
    embeddingModel: 'nomic-embed-text',
    createdAt: '2025-12-05',
  },
]

const sampleDocuments: Document[] = [
  {
    id: 'doc-1',
    name: 'Getting Started Guide.pdf',
    type: 'pdf',
    size: 2450000,
    chunks: 24,
    status: 'ready',
    uploadedAt: '2025-12-07T10:00:00Z',
    collection: 'col-1',
  },
  {
    id: 'doc-2',
    name: 'API Reference.md',
    type: 'md',
    size: 156000,
    chunks: 18,
    status: 'ready',
    uploadedAt: '2025-12-07T11:30:00Z',
    collection: 'col-1',
  },
  {
    id: 'doc-3',
    name: 'Attention Is All You Need.pdf',
    type: 'pdf',
    size: 890000,
    chunks: 32,
    status: 'ready',
    uploadedAt: '2025-12-06T14:00:00Z',
    collection: 'col-2',
  },
  {
    id: 'doc-4',
    name: 'New Document.txt',
    type: 'txt',
    size: 45000,
    chunks: 0,
    status: 'processing',
    uploadedAt: '2025-12-07T15:00:00Z',
    collection: 'col-1',
  },
]

// Format file size
const formatSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// Document type icons and colors
const typeConfig: Record<string, { icon: React.ReactNode; color: string }> = {
  pdf: { icon: <DocIcon />, color: '#ef4444' },
  txt: { icon: <DocIcon />, color: '#6366f1' },
  md: { icon: <DocIcon />, color: '#10b981' },
  html: { icon: <LinkIcon />, color: '#f59e0b' },
  url: { icon: <LinkIcon />, color: '#06b6d4' },
}

// Upload Dialog
interface UploadDialogProps {
  open: boolean
  onClose: () => void
  collections: Collection[]
}

const UploadDialog: React.FC<UploadDialogProps> = ({ open, onClose, collections }) => {
  const [selectedCollection, setSelectedCollection] = useState('')
  const [uploadType, setUploadType] = useState<'file' | 'url' | 'text'>('file')
  const [url, setUrl] = useState('')
  const [text, setText] = useState('')
  const [files, setFiles] = useState<File[]>([])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const handleUpload = () => {
    // Handle upload logic here
    console.log('Uploading:', { selectedCollection, uploadType, url, text, files })
    onClose()
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <UploadIcon sx={{ color: '#6366f1' }} />
        Add Documents
      </DialogTitle>
      <DialogContent>
        <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <FormControl fullWidth size="small">
            <InputLabel>Collection</InputLabel>
            <Select
              value={selectedCollection}
              label="Collection"
              onChange={(e) => setSelectedCollection(e.target.value)}
            >
              {collections.map((col) => (
                <MenuItem key={col.id} value={col.id}>{col.name}</MenuItem>
              ))}
            </Select>
          </FormControl>

          <Tabs value={uploadType} onChange={(_, v) => setUploadType(v)} sx={{ mb: 2 }}>
            <Tab value="file" label="Upload File" />
            <Tab value="url" label="From URL" />
            <Tab value="text" label="Paste Text" />
          </Tabs>

          {uploadType === 'file' && (
            <Box
              sx={{
                p: 4,
                border: '2px dashed rgba(255, 255, 255, 0.1)',
                borderRadius: 2,
                textAlign: 'center',
                cursor: 'pointer',
                '&:hover': { borderColor: alpha('#6366f1', 0.5), bgcolor: alpha('#6366f1', 0.05) },
              }}
              component="label"
            >
              <input
                type="file"
                hidden
                multiple
                accept=".pdf,.txt,.md,.html"
                onChange={handleFileChange}
              />
              <UploadIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                Drop files here or click to browse
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Supported: PDF, TXT, MD, HTML
              </Typography>
              {files.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  {files.map((file, i) => (
                    <Chip key={i} label={file.name} size="small" sx={{ m: 0.5 }} />
                  ))}
                </Box>
              )}
            </Box>
          )}

          {uploadType === 'url' && (
            <TextField
              fullWidth
              label="URL"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/document.pdf"
              size="small"
            />
          )}

          {uploadType === 'text' && (
            <TextField
              fullWidth
              label="Text Content"
              value={text}
              onChange={(e) => setText(e.target.value)}
              multiline
              rows={6}
              placeholder="Paste your text content here..."
              size="small"
            />
          )}
        </Box>
      </DialogContent>
      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          onClick={handleUpload}
          disabled={!selectedCollection || (uploadType === 'file' && files.length === 0)}
        >
          Upload
        </Button>
      </DialogActions>
    </Dialog>
  )
}

// New Collection Dialog
interface NewCollectionDialogProps {
  open: boolean
  onClose: () => void
  onCreate: (collection: Partial<Collection>) => void
}

const NewCollectionDialog: React.FC<NewCollectionDialogProps> = ({ open, onClose, onCreate }) => {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [embeddingModel, setEmbeddingModel] = useState('nomic-embed-text')

  const handleCreate = () => {
    onCreate({ name, description, embeddingModel })
    setName('')
    setDescription('')
    onClose()
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <FolderIcon sx={{ color: '#10b981' }} />
        New Collection
      </DialogTitle>
      <DialogContent>
        <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <TextField
            fullWidth
            label="Collection Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            size="small"
          />
          <TextField
            fullWidth
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            multiline
            rows={2}
            size="small"
          />
          <FormControl fullWidth size="small">
            <InputLabel>Embedding Model</InputLabel>
            <Select
              value={embeddingModel}
              label="Embedding Model"
              onChange={(e) => setEmbeddingModel(e.target.value)}
            >
              <MenuItem value="nomic-embed-text">nomic-embed-text</MenuItem>
              <MenuItem value="all-MiniLM-L6-v2">all-MiniLM-L6-v2</MenuItem>
              <MenuItem value="bge-large-en-v1.5">bge-large-en-v1.5</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </DialogContent>
      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleCreate} disabled={!name}>
          Create Collection
        </Button>
      </DialogActions>
    </Dialog>
  )
}

// Search Dialog
interface SearchDialogProps {
  open: boolean
  onClose: () => void
  collections: Collection[]
}

const SearchDialog: React.FC<SearchDialogProps> = ({ open, onClose, collections }) => {
  const [query, setQuery] = useState('')
  const [selectedCollection, setSelectedCollection] = useState('')
  const [topK, setTopK] = useState(5)
  const [results, setResults] = useState<any[]>([])
  const [searching, setSearching] = useState(false)

  const handleSearch = async () => {
    setSearching(true)
    // Simulate search
    await new Promise((r) => setTimeout(r, 1000))
    setResults([
      { id: 1, content: 'This is a sample search result matching your query...', score: 0.95, document: 'Getting Started Guide.pdf' },
      { id: 2, content: 'Another relevant chunk of text from the knowledge base...', score: 0.87, document: 'API Reference.md' },
      { id: 3, content: 'A third result that might be useful for context...', score: 0.82, document: 'Getting Started Guide.pdf' },
    ])
    setSearching(false)
  }

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <SearchIcon sx={{ color: '#06b6d4' }} />
        Semantic Search
      </DialogTitle>
      <DialogContent>
        <Box sx={{ pt: 1, display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              fullWidth
              label="Search Query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your search query..."
              size="small"
            />
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>Collection</InputLabel>
              <Select
                value={selectedCollection}
                label="Collection"
                onChange={(e) => setSelectedCollection(e.target.value)}
              >
                <MenuItem value="">All Collections</MenuItem>
                {collections.map((col) => (
                  <MenuItem key={col.id} value={col.id}>{col.name}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              type="number"
              label="Top K"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              size="small"
              sx={{ width: 100 }}
              inputProps={{ min: 1, max: 20 }}
            />
            <Button
              variant="contained"
              onClick={handleSearch}
              disabled={!query || searching}
              sx={{ minWidth: 100 }}
            >
              {searching ? 'Searching...' : 'Search'}
            </Button>
          </Box>

          {results.length > 0 && (
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Results ({results.length})
              </Typography>
              {results.map((result) => (
                <Card
                  key={result.id}
                  sx={{
                    mb: 1,
                    bgcolor: 'rgba(255, 255, 255, 0.02)',
                    border: '1px solid rgba(255, 255, 255, 0.06)',
                  }}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        {result.document}
                      </Typography>
                      <Chip
                        label={`Score: ${result.score.toFixed(2)}`}
                        size="small"
                        sx={{
                          height: 20,
                          bgcolor: alpha('#10b981', 0.1),
                          color: '#34d399',
                          fontSize: '0.625rem',
                        }}
                      />
                    </Box>
                    <Typography variant="body2">{result.content}</Typography>
                  </CardContent>
                </Card>
              ))}
            </Box>
          )}
        </Box>
      </DialogContent>
      <DialogActions sx={{ p: 2 }}>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  )
}

// Main Knowledge Base Page
const KnowledgeBasePage: React.FC = () => {
  const [collections, setCollections] = useState<Collection[]>(sampleCollections)
  const [documents, setDocuments] = useState<Document[]>(sampleDocuments)
  const [selectedCollection, setSelectedCollection] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [newCollectionDialogOpen, setNewCollectionDialogOpen] = useState(false)
  const [searchDialogOpen, setSearchDialogOpen] = useState(false)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null)

  const filteredDocuments = documents.filter((doc) => {
    if (selectedCollection && doc.collection !== selectedCollection) return false
    if (searchQuery && !doc.name.toLowerCase().includes(searchQuery.toLowerCase())) return false
    return true
  })

  const handleCreateCollection = (collection: Partial<Collection>) => {
    const newCollection: Collection = {
      id: `col-${Date.now()}`,
      name: collection.name || 'New Collection',
      description: collection.description || '',
      documentCount: 0,
      totalChunks: 0,
      embeddingModel: collection.embeddingModel || 'nomic-embed-text',
      createdAt: new Date().toISOString().split('T')[0],
    }
    setCollections([...collections, newCollection])
  }

  const handleDeleteDocument = (docId: string) => {
    setDocuments(documents.filter((d) => d.id !== docId))
    setAnchorEl(null)
  }

  return (
    <Box
      sx={{
        width: '100%',
        maxWidth: '100vw',
        overflow: 'hidden',
        px: { xs: 2, sm: 3, md: 4 },
        py: 3,
        boxSizing: 'border-box',
      }}
    >
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
          <Typography
            variant="h1"
            sx={{
              fontWeight: 700,
              fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2rem' },
              background: 'linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Knowledge Base
          </Typography>
          <Chip
            label="RAG"
            size="small"
            sx={{
              height: 22,
              bgcolor: alpha('#06b6d4', 0.1),
              border: `1px solid ${alpha('#06b6d4', 0.2)}`,
              color: '#22d3ee',
              fontWeight: 600,
              fontSize: '0.6875rem',
            }}
          />
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 600 }}>
          Upload and manage documents for retrieval-augmented generation. Search your knowledge base semantically.
        </Typography>
      </Box>

      {/* Actions Bar */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          onClick={() => setUploadDialogOpen(true)}
        >
          Upload Documents
        </Button>
        <Button
          variant="outlined"
          startIcon={<FolderIcon />}
          onClick={() => setNewCollectionDialogOpen(true)}
        >
          New Collection
        </Button>
        <Button
          variant="outlined"
          startIcon={<SearchIcon />}
          onClick={() => setSearchDialogOpen(true)}
          sx={{ borderColor: '#06b6d4', color: '#06b6d4' }}
        >
          Semantic Search
        </Button>
        <Box sx={{ flex: 1 }} />
        <TextField
          size="small"
          placeholder="Filter documents..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <FilterIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
              </InputAdornment>
            ),
          }}
          sx={{ width: 250 }}
        />
      </Box>

      <Grid container spacing={3}>
        {/* Collections Sidebar */}
        <Grid item xs={12} md={3}>
          <Card
            sx={{
              background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: 2,
            }}
          >
            <CardContent sx={{ p: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
                Collections
              </Typography>
              <Box
                sx={{
                  p: 1.5,
                  mb: 1,
                  borderRadius: 1,
                  cursor: 'pointer',
                  bgcolor: !selectedCollection ? alpha('#6366f1', 0.1) : 'transparent',
                  border: !selectedCollection ? `1px solid ${alpha('#6366f1', 0.3)}` : '1px solid transparent',
                  '&:hover': { bgcolor: alpha('#6366f1', 0.08) },
                }}
                onClick={() => setSelectedCollection(null)}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <StorageIcon sx={{ fontSize: 18, color: '#6366f1' }} />
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>All Documents</Typography>
                </Box>
                <Typography variant="caption" color="text.secondary">
                  {documents.length} documents
                </Typography>
              </Box>

              {collections.map((col) => (
                <Box
                  key={col.id}
                  sx={{
                    p: 1.5,
                    mb: 1,
                    borderRadius: 1,
                    cursor: 'pointer',
                    bgcolor: selectedCollection === col.id ? alpha('#10b981', 0.1) : 'transparent',
                    border: selectedCollection === col.id ? `1px solid ${alpha('#10b981', 0.3)}` : '1px solid transparent',
                    '&:hover': { bgcolor: alpha('#10b981', 0.08) },
                  }}
                  onClick={() => setSelectedCollection(col.id)}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <FolderIcon sx={{ fontSize: 18, color: '#10b981' }} />
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>{col.name}</Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {col.documentCount} docs, {col.totalChunks} chunks
                  </Typography>
                </Box>
              ))}
            </CardContent>
          </Card>

          {/* Stats Card */}
          <Card
            sx={{
              mt: 2,
              background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: 2,
            }}
          >
            <CardContent sx={{ p: 2 }}>
              <Typography variant="subtitle2" sx={{ mb: 2, color: 'text.secondary' }}>
                Statistics
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="caption" color="text.secondary">Total Documents</Typography>
                  <Typography variant="body2" fontWeight={600}>{documents.length}</Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="caption" color="text.secondary">Total Chunks</Typography>
                  <Typography variant="body2" fontWeight={600}>
                    {documents.reduce((acc, d) => acc + d.chunks, 0)}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="caption" color="text.secondary">Collections</Typography>
                  <Typography variant="body2" fontWeight={600}>{collections.length}</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Documents Table */}
        <Grid item xs={12} md={9}>
          <Card
            sx={{
              background: 'linear-gradient(145deg, rgba(30, 30, 63, 0.6) 0%, rgba(26, 26, 46, 0.8) 100%)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: 2,
            }}
          >
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Document</TableCell>
                    <TableCell>Collection</TableCell>
                    <TableCell align="right">Size</TableCell>
                    <TableCell align="right">Chunks</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Uploaded</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredDocuments.map((doc) => {
                    const typeConf = typeConfig[doc.type] || typeConfig.txt
                    const collection = collections.find((c) => c.id === doc.collection)
                    return (
                      <TableRow key={doc.id} hover>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <Box sx={{ color: typeConf.color }}>{typeConf.icon}</Box>
                            <Typography variant="body2">{doc.name}</Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={collection?.name || 'Unknown'}
                            size="small"
                            sx={{
                              height: 22,
                              bgcolor: alpha('#10b981', 0.1),
                              color: '#34d399',
                              fontSize: '0.6875rem',
                            }}
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" color="text.secondary">
                            {formatSize(doc.size)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" color="text.secondary">
                            {doc.chunks}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          {doc.status === 'processing' ? (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <LinearProgress sx={{ width: 60, height: 4 }} />
                              <Typography variant="caption" color="text.secondary">
                                Processing
                              </Typography>
                            </Box>
                          ) : doc.status === 'ready' ? (
                            <Chip
                              label="Ready"
                              size="small"
                              sx={{
                                height: 20,
                                bgcolor: alpha('#10b981', 0.1),
                                color: '#34d399',
                                fontSize: '0.625rem',
                              }}
                            />
                          ) : (
                            <Chip
                              label="Error"
                              size="small"
                              sx={{
                                height: 20,
                                bgcolor: alpha('#ef4444', 0.1),
                                color: '#f87171',
                                fontSize: '0.625rem',
                              }}
                            />
                          )}
                        </TableCell>
                        <TableCell>
                          <Typography variant="caption" color="text.secondary">
                            {new Date(doc.uploadedAt).toLocaleDateString()}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <IconButton
                            size="small"
                            onClick={(e) => {
                              setAnchorEl(e.currentTarget)
                              setSelectedDoc(doc)
                            }}
                          >
                            <MoreIcon fontSize="small" />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                  {filteredDocuments.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                        <Typography color="text.secondary">
                          No documents found. Upload some documents to get started.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Card>
        </Grid>
      </Grid>

      {/* Document Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        <MenuItem onClick={() => setAnchorEl(null)}>
          <ViewIcon sx={{ mr: 1, fontSize: 18 }} /> View Chunks
        </MenuItem>
        <MenuItem onClick={() => setAnchorEl(null)}>
          <DownloadIcon sx={{ mr: 1, fontSize: 18 }} /> Download
        </MenuItem>
        <MenuItem onClick={() => setAnchorEl(null)}>
          <EmbedIcon sx={{ mr: 1, fontSize: 18 }} /> Re-embed
        </MenuItem>
        <MenuItem onClick={() => selectedDoc && handleDeleteDocument(selectedDoc.id)} sx={{ color: '#ef4444' }}>
          <DeleteIcon sx={{ mr: 1, fontSize: 18 }} /> Delete
        </MenuItem>
      </Menu>

      {/* Dialogs */}
      <UploadDialog
        open={uploadDialogOpen}
        onClose={() => setUploadDialogOpen(false)}
        collections={collections}
      />
      <NewCollectionDialog
        open={newCollectionDialogOpen}
        onClose={() => setNewCollectionDialogOpen(false)}
        onCreate={handleCreateCollection}
      />
      <SearchDialog
        open={searchDialogOpen}
        onClose={() => setSearchDialogOpen(false)}
        collections={collections}
      />
    </Box>
  )
}

export default KnowledgeBasePage
