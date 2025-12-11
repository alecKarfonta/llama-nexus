/**
 * AnimatedEdge - Custom edge component with animation support
 */
import React from 'react';
import { EdgeProps, getBezierPath, EdgeLabelRenderer } from 'reactflow';
import { alpha, keyframes } from '@mui/material';

const flowAnimation = keyframes`
  0% {
    stroke-dashoffset: 24;
  }
  100% {
    stroke-dashoffset: 0;
  }
`;

export const AnimatedEdge: React.FC<EdgeProps> = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  selected,
  data,
}) => {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const isAnimated = data?.animated || false;
  const isRunning = data?.status === 'running';
  const hasError = data?.status === 'error';

  const strokeColor = hasError 
    ? '#ef4444' 
    : selected 
      ? '#6366f1' 
      : alpha('#6366f1', 0.6);

  return (
    <>
      {/* Background path for better visibility */}
      <path
        id={`${id}-bg`}
        className="react-flow__edge-path"
        d={edgePath}
        style={{
          stroke: alpha('#000', 0.2),
          strokeWidth: 6,
          fill: 'none',
        }}
      />
      
      {/* Main path */}
      <path
        id={id}
        className="react-flow__edge-path"
        d={edgePath}
        style={{
          stroke: strokeColor,
          strokeWidth: selected ? 3 : 2,
          fill: 'none',
          transition: 'stroke 0.2s, stroke-width 0.2s',
          ...(isAnimated || isRunning ? {
            strokeDasharray: '8 8',
            animation: `${flowAnimation} 0.5s linear infinite`,
          } : {}),
          ...style,
        }}
        markerEnd={markerEnd}
      />

      {/* Optional label */}
      {data?.label && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              background: '#1e1e3f',
              padding: '2px 6px',
              borderRadius: 4,
              fontSize: 10,
              fontWeight: 500,
              color: '#fff',
              border: '1px solid rgba(255,255,255,0.1)',
              pointerEvents: 'all',
            }}
          >
            {data.label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default AnimatedEdge;
