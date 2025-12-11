/**
 * WorkflowCanvas - Main React Flow canvas for workflow editing
 */
import React, { useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  NodeTypes,
  EdgeTypes,
  OnConnect,
  OnNodesChange,
  OnEdgesChange,
  ReactFlowInstance,
  XYPosition,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Box, alpha } from '@mui/material';
import { BaseNode } from './BaseNode';
import { AnimatedEdge } from './AnimatedEdge';
import { 
  WorkflowNode, 
  WorkflowConnection, 
  getNodeTypeDefinition,
  BUILTIN_NODE_TYPES,
  WorkflowNodeData,
  ExecutionStatus,
  NodeExecution,
} from '@/types/workflow';

interface WorkflowCanvasProps {
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  nodeExecutions?: Record<string, NodeExecution>;
  onNodesChange: (nodes: WorkflowNode[]) => void;
  onConnectionsChange: (connections: WorkflowConnection[]) => void;
  onNodeSelect: (node: WorkflowNode | null) => void;
  selectedNodeId?: string | null;
}

// Convert our workflow types to React Flow types
function toReactFlowNode(node: WorkflowNode, executionStatus?: NodeExecution): Node<WorkflowNodeData> {
  return {
    id: node.id,
    type: 'workflow', // Use our custom node type
    position: node.position,
    data: {
      ...node.data,
      status: executionStatus?.status,
      error: executionStatus?.error,
      executionTime: executionStatus?.duration,
    },
    selected: false,
  };
}

function toReactFlowEdge(conn: WorkflowConnection): Edge {
  return {
    id: conn.id,
    source: conn.source,
    sourceHandle: conn.sourceHandle,
    target: conn.target,
    targetHandle: conn.targetHandle,
    type: 'animated',
    animated: false,
  };
}

function fromReactFlowNode(node: Node<WorkflowNodeData>): WorkflowNode {
  return {
    id: node.id,
    type: node.data.nodeType,
    position: node.position,
    data: node.data,
  };
}

function fromReactFlowEdge(edge: Edge): WorkflowConnection {
  return {
    id: edge.id,
    source: edge.source,
    sourceHandle: edge.sourceHandle || '',
    target: edge.target,
    targetHandle: edge.targetHandle || '',
  };
}

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  nodes,
  connections,
  nodeExecutions,
  onNodesChange,
  onConnectionsChange,
  onNodeSelect,
  selectedNodeId,
}) => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = React.useState<ReactFlowInstance | null>(null);

  // Convert to React Flow format
  const rfNodes = useMemo(() => 
    nodes.map((n) => toReactFlowNode(n, nodeExecutions?.[n.id])),
    [nodes, nodeExecutions]
  );
  
  const rfEdges = useMemo(() => 
    connections.map(toReactFlowEdge),
    [connections]
  );

  const [flowNodes, setFlowNodes, onFlowNodesChange] = useNodesState(rfNodes);
  const [flowEdges, setFlowEdges, onFlowEdgesChange] = useEdgesState(rfEdges);

  // Sync external changes
  React.useEffect(() => {
    setFlowNodes(rfNodes);
  }, [rfNodes, setFlowNodes]);

  React.useEffect(() => {
    setFlowEdges(rfEdges);
  }, [rfEdges, setFlowEdges]);

  // Custom node types
  const nodeTypes: NodeTypes = useMemo(() => ({
    workflow: BaseNode,
  }), []);

  // Custom edge types
  const edgeTypes: EdgeTypes = useMemo(() => ({
    animated: AnimatedEdge,
  }), []);

  // Handle node changes
  const handleNodesChange: OnNodesChange = useCallback((changes) => {
    onFlowNodesChange(changes);
    // Debounce update to parent
    setTimeout(() => {
      setFlowNodes((nds) => {
        onNodesChange(nds.map(fromReactFlowNode));
        return nds;
      });
    }, 0);
  }, [onFlowNodesChange, onNodesChange, setFlowNodes]);

  // Handle edge changes
  const handleEdgesChange: OnEdgesChange = useCallback((changes) => {
    onFlowEdgesChange(changes);
    setTimeout(() => {
      setFlowEdges((eds) => {
        onConnectionsChange(eds.map(fromReactFlowEdge));
        return eds;
      });
    }, 0);
  }, [onFlowEdgesChange, onConnectionsChange, setFlowEdges]);

  // Handle new connections
  const handleConnect: OnConnect = useCallback((params: Connection) => {
    if (!params.source || !params.target) return;

    // Validate connection types could be done here
    const newEdge: Edge = {
      id: `e-${params.source}-${params.sourceHandle}-${params.target}-${params.targetHandle}`,
      source: params.source,
      sourceHandle: params.sourceHandle,
      target: params.target,
      targetHandle: params.targetHandle,
      type: 'animated',
    };

    setFlowEdges((eds) => {
      const updated = addEdge(newEdge, eds);
      onConnectionsChange(updated.map(fromReactFlowEdge));
      return updated;
    });
  }, [setFlowEdges, onConnectionsChange]);

  // Handle node selection
  const handleNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    onNodeSelect(fromReactFlowNode(node as Node<WorkflowNodeData>));
  }, [onNodeSelect]);

  const handlePaneClick = useCallback(() => {
    onNodeSelect(null);
  }, [onNodeSelect]);

  // Handle drag and drop from palette
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();

    const nodeType = event.dataTransfer.getData('application/workflow-node');
    if (!nodeType || !reactFlowInstance || !reactFlowWrapper.current) return;

    const typeDef = getNodeTypeDefinition(nodeType);
    if (!typeDef) return;

    // Get drop position
    const bounds = reactFlowWrapper.current.getBoundingClientRect();
    const position = reactFlowInstance.screenToFlowPosition({
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });

    // Create new node
    const newNode: WorkflowNode = {
      id: `node-${Date.now()}`,
      type: nodeType,
      position,
      data: {
        label: typeDef.displayName,
        nodeType: nodeType,
        config: {},
        inputs: typeDef.inputs,
        outputs: typeDef.outputs,
      },
    };

    onNodesChange([...nodes, newNode]);
  }, [reactFlowInstance, nodes, onNodesChange]);

  return (
    <Box
      ref={reactFlowWrapper}
      sx={{
        width: '100%',
        height: '100%',
        bgcolor: 'background.default',
        '& .react-flow__node': {
          cursor: 'pointer',
        },
        '& .react-flow__handle': {
          width: 10,
          height: 10,
          borderRadius: '50%',
          border: '2px solid',
          borderColor: 'primary.main',
          bgcolor: 'background.paper',
          '&:hover': {
            bgcolor: 'primary.main',
          },
        },
        '& .react-flow__handle-left': {
          left: -6,
        },
        '& .react-flow__handle-right': {
          right: -6,
        },
        '& .react-flow__edge-path': {
          stroke: alpha('#6366f1', 0.6),
          strokeWidth: 2,
        },
        '& .react-flow__edge.selected .react-flow__edge-path': {
          stroke: '#6366f1',
          strokeWidth: 3,
        },
        '& .react-flow__controls': {
          borderRadius: 2,
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          boxShadow: 2,
        },
        '& .react-flow__controls-button': {
          bgcolor: 'background.paper',
          border: 'none',
          borderBottom: '1px solid',
          borderColor: 'divider',
          '&:hover': {
            bgcolor: 'action.hover',
          },
          '& svg': {
            fill: 'currentColor',
          },
        },
        '& .react-flow__minimap': {
          borderRadius: 2,
          bgcolor: 'background.paper',
          border: '1px solid',
          borderColor: 'divider',
          boxShadow: 2,
        },
      }}
    >
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        onConnect={handleConnect}
        onNodeClick={handleNodeClick}
        onPaneClick={handlePaneClick}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onInit={setReactFlowInstance}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        snapToGrid
        snapGrid={[16, 16]}
        defaultEdgeOptions={{
          type: 'animated',
        }}
      >
        <Controls />
        <MiniMap 
          nodeStrokeWidth={3}
          zoomable
          pannable
        />
        <Background 
          variant={BackgroundVariant.Dots} 
          gap={16} 
          size={1}
          color="rgba(255, 255, 255, 0.1)"
        />
      </ReactFlow>
    </Box>
  );
};

export default WorkflowCanvas;
