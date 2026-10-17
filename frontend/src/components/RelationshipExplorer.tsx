import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { getGraphSimilarity } from '../api/client';
import type {
  SearchResponse,
  GraphNode as BackendGraphNode,
  GraphEdge as BackendGraphEdge,
  PatentResult,
} from '../types/patent';

import {
  Share2,
  Sliders,
  AlertCircle,
  RotateCcw,
  Search,
  Copy,
  Check,
  Compass,
  Tag,
  Sparkles,
  Info,
  Loader2,
  X,
} from 'lucide-react';

interface RelationshipExplorerProps {
  searchResponse: SearchResponse | null;
  onNavigateToSearch: () => void;
}

// Color palette mapping based on IPC Class string
const IPC_PALETTES: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  '16': { bg: 'rgba(6, 182, 212, 0.15)', border: '#06b6d4', text: '#22d3ee', glow: 'rgba(6, 182, 212, 0.3)' },
  '12': { bg: 'rgba(14, 165, 233, 0.15)', border: '#0ea5e9', text: '#38bdf8', glow: 'rgba(14, 165, 233, 0.3)' },
  '8': { bg: 'rgba(16, 185, 129, 0.15)', border: '#10b981', text: '#34d399', glow: 'rgba(16, 185, 129, 0.3)' },
  '4': { bg: 'rgba(245, 158, 11, 0.15)', border: '#f59e0b', text: '#fbbf24', glow: 'rgba(245, 158, 11, 0.3)' },
  '20': { bg: 'rgba(99, 102, 241, 0.15)', border: '#6366f1', text: '#818cf8', glow: 'rgba(99, 102, 241, 0.3)' },
  default: { bg: 'rgba(148, 163, 184, 0.12)', border: '#64748b', text: '#cbd5e1', glow: 'rgba(148, 163, 184, 0.25)' },
};

function getIpcColor(ipcClass: string | null) {
  if (!ipcClass) return IPC_PALETTES.default;
  return IPC_PALETTES[ipcClass] || IPC_PALETTES.default;
}

export const RelationshipExplorer: React.FC<RelationshipExplorerProps> = ({
  searchResponse,
  onNavigateToSearch,
}) => {
  const [similarityThreshold, setSimilarityThreshold] = useState<number>(0.65);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Raw graph elements from backend
  const [backendNodes, setBackendNodes] = useState<BackendGraphNode[]>([]);
  const [backendEdges, setBackendEdges] = useState<BackendGraphEdge[]>([]);

  // Selected node state
  const [selectedPatentId, setSelectedPatentId] = useState<string | null>(null);
  const [hasCopiedId, setHasCopiedId] = useState<boolean>(false);

  // Fast map to lookup original search result info (rank, hybrid score, chunk)
  const searchResultMap = useMemo(() => {
    const map = new Map<string, { result: PatentResult; rank: number }>();
    if (searchResponse?.results) {
      searchResponse.results.forEach((r, idx) => {
        map.set(r.patent_id, { result: r, rank: idx + 1 });
      });
    }
    return map;
  }, [searchResponse]);

  // Extract patent IDs to send to backend
  const patentIdsToExplore = useMemo(() => {
    if (!searchResponse || !searchResponse.results || searchResponse.results.length < 2) {
      return [];
    }
    // Limit to max 25 as per backend schema
    return searchResponse.results.slice(0, 25).map((r) => r.patent_id);
  }, [searchResponse]);

  // Manual trigger for refresh and slider updates
  const handleManualRefresh = useCallback(async () => {
    if (patentIdsToExplore.length < 2) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await getGraphSimilarity({
        patent_ids: patentIdsToExplore,
        similarity_threshold: similarityThreshold,
      });

      setBackendNodes(response.nodes);
      setBackendEdges(response.edges);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to retrieve relationship graph.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [patentIdsToExplore, similarityThreshold]);

  // Fetch on mount or when patent IDs or threshold change
  useEffect(() => {
    if (patentIdsToExplore.length < 2) return;

    let isMounted = true;
    const loadGraph = async () => {
      try {
        const response = await getGraphSimilarity({
          patent_ids: patentIdsToExplore,
          similarity_threshold: similarityThreshold,
        });

        if (isMounted) {
          setBackendNodes(response.nodes);
          setBackendEdges(response.edges);
          setIsLoading(false);
          setError(null);
        }
      } catch (err: unknown) {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Failed to retrieve relationship graph.';
          setError(msg);
          setIsLoading(false);
        }
      }
    };

    void loadGraph();

    return () => {
      isMounted = false;
    };
  }, [patentIdsToExplore, similarityThreshold]);

  // Convert backend nodes & edges into React Flow elements with circular arrangement
  const flowNodes: Node[] = useMemo(() => {
    const n = backendNodes.length;
    if (n === 0) return [];

    const radiusX = Math.max(280, n * 24);
    const radiusY = Math.max(200, n * 18);
    const centerX = 360;
    const centerY = 240;

    return backendNodes.map((bNode, idx) => {
      const angle = (2 * Math.PI * idx) / n - Math.PI / 2;
      const x = centerX + radiusX * Math.cos(angle);
      const y = centerY + radiusY * Math.sin(angle);

      const isSelected = selectedPatentId === bNode.patent_id;
      const colorScheme = getIpcColor(bNode.ipc_class);
      const searchItem = searchResultMap.get(bNode.patent_id);

      return {
        id: bNode.id,
        position: { x, y },
        data: {
          label: (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '4px',
                textAlign: 'left',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '8px' }}>
                <span
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    fontFamily: 'var(--font-mono)',
                    color: isSelected ? '#ffffff' : colorScheme.text,
                  }}
                >
                  {bNode.patent_id}
                </span>
                {searchItem && (
                  <span
                    style={{
                      fontSize: '9px',
                      fontWeight: 700,
                      padding: '1px 4px',
                      borderRadius: '3px',
                      backgroundColor: 'rgba(255, 255, 255, 0.1)',
                      color: '#cbd5e1',
                    }}
                  >
                    #{searchItem.rank}
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--text-muted)' }}>
                <span>IPC {bNode.ipc_class ?? 'N/A'}</span>
                {bNode.subclass && <span>&bull; {bNode.subclass}</span>}
              </div>
            </div>
          ),
        },
        style: {
          background: isSelected ? 'var(--bg-card-hover)' : colorScheme.bg,
          border: `1.5px solid ${isSelected ? '#22d3ee' : colorScheme.border}`,
          borderRadius: '10px',
          padding: '10px 14px',
          color: 'var(--text-primary)',
          cursor: 'pointer',
          width: 160,
          boxShadow: isSelected
            ? `0 0 16px ${colorScheme.glow}, 0 4px 12px rgba(0, 0, 0, 0.6)`
            : '0 2px 8px rgba(0, 0, 0, 0.4)',
          transition: 'all 0.2s ease',
        },
      };
    });
  }, [backendNodes, selectedPatentId, searchResultMap]);

  const flowEdges: Edge[] = useMemo(() => {
    return backendEdges.map((bEdge, idx) => {
      const isConnectedToSelected =
        selectedPatentId &&
        (bEdge.source === selectedPatentId || bEdge.target === selectedPatentId);

      const strokeColor = isConnectedToSelected ? '#22d3ee' : 'rgba(56, 189, 248, 0.45)';
      const strokeWidth = isConnectedToSelected ? 2.5 : 1.5;

      return {
        id: `edge-${bEdge.source}-${bEdge.target}-${idx}`,
        source: bEdge.source,
        target: bEdge.target,
        animated: isConnectedToSelected || bEdge.similarity >= 0.8,
        label: bEdge.similarity.toFixed(3),
        labelStyle: {
          fill: isConnectedToSelected ? '#f8fafc' : '#94a3b8',
          fontWeight: 600,
          fontSize: 10,
          fontFamily: 'var(--font-mono)',
        },
        labelBgStyle: {
          fill: '#0b1120',
          fillOpacity: 0.9,
          rx: 4,
          ry: 4,
        },
        labelBgPadding: [4, 2] as [number, number],
        style: {
          stroke: strokeColor,
          strokeWidth,
        },
      };
    });
  }, [backendEdges, selectedPatentId]);

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  // Sync flowNodes and flowEdges when data changes
  useEffect(() => {
    setNodes(flowNodes);
  }, [flowNodes, setNodes]);

  useEffect(() => {
    setEdges(flowEdges);
  }, [flowEdges, setEdges]);

  // Selected patent details
  const selectedNodeData = useMemo(() => {
    if (!selectedPatentId) return null;
    return backendNodes.find((n) => n.patent_id === selectedPatentId) || null;
  }, [selectedPatentId, backendNodes]);

  const selectedSearchResult = useMemo(() => {
    if (!selectedPatentId) return null;
    return searchResultMap.get(selectedPatentId) || null;
  }, [selectedPatentId, searchResultMap]);

  // Connected patents to selected node
  const connectedEdges = useMemo(() => {
    if (!selectedPatentId) return [];
    return backendEdges
      .filter((e) => e.source === selectedPatentId || e.target === selectedPatentId)
      .map((e) => ({
        otherPatentId: e.source === selectedPatentId ? e.target : e.source,
        similarity: e.similarity,
      }))
      .sort((a, b) => b.similarity - a.similarity);
  }, [selectedPatentId, backendEdges]);

  const handleCopyId = (pid: string) => {
    navigator.clipboard.writeText(pid);
    setHasCopiedId(true);
    setTimeout(() => setHasCopiedId(false), 2000);
  };

  // State 1: No prior search performed (Empty state)
  if (!searchResponse || searchResponse.results.length < 2) {
    return (
      <div style={{ maxWidth: '840px', margin: '40px auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div
          style={{
            padding: '48px 36px',
            backgroundColor: 'var(--bg-card)',
            borderRadius: 'var(--radius-xl)',
            border: '1px dashed var(--border-card)',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '16px',
            boxShadow: 'var(--shadow-md)',
          }}
        >
          <div
            style={{
              width: '54px',
              height: '54px',
              borderRadius: '14px',
              backgroundColor: 'var(--accent-cyan-muted)',
              color: 'var(--accent-cyan-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px solid rgba(6, 182, 212, 0.3)',
            }}
          >
            <Share2 size={26} />
          </div>

          <div>
            <h3 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '8px' }}>
              No Active Patent Candidates to Explore
            </h3>
            <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '580px' }}>
              The Relationship Explorer computes real pairwise vector cosine similarities between candidate patents
              retrieved from your semantic query. Run a search to populate the graph with candidate embeddings.
            </p>
          </div>

          <div style={{ paddingTop: '8px' }}>
            <button
              onClick={onNavigateToSearch}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '10px 22px',
                borderRadius: 'var(--radius-md)',
                background: 'linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)',
                color: '#ffffff',
                fontWeight: 600,
                fontSize: '14px',
                cursor: 'pointer',
                boxShadow: '0 4px 14px rgba(6, 182, 212, 0.3)',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.transform = 'translateY(-1px)')}
              onMouseLeave={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
            >
              <Search size={16} />
              <span>Go to Semantic Search</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 110px)', gap: '16px' }}>
      {/* Top Controls Toolbar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '16px',
          padding: '14px 20px',
          borderRadius: 'var(--radius-lg)',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-card)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              padding: '6px',
              borderRadius: '8px',
              backgroundColor: 'var(--accent-cyan-muted)',
              color: 'var(--accent-cyan-light)',
            }}
          >
            <Share2 size={16} />
          </div>
          <div>
            <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span>Pairwise Patent Similarity Graph</span>
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  fontFamily: 'var(--font-mono)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  backgroundColor: 'rgba(56, 189, 248, 0.1)',
                  color: 'var(--accent-sky)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                }}
              >
                {backendNodes.length} Patents &bull; {backendEdges.length} Edges
              </span>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Source Query: &ldquo;{searchResponse.query}&rdquo;
            </div>
          </div>
        </div>

        {/* Threshold Slider Control */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Sliders size={14} style={{ color: 'var(--accent-cyan)' }} />
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              Similarity Threshold:
            </span>
            <input
              type="range"
              min={0.0}
              max={1.0}
              step={0.05}
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              disabled={isLoading}
              style={{ width: '110px', accentColor: 'var(--accent-cyan)' }}
            />
            <span
              style={{
                fontSize: '13px',
                fontWeight: 700,
                fontFamily: 'var(--font-mono)',
                color: 'var(--accent-cyan-light)',
                minWidth: '36px',
              }}
            >
              {similarityThreshold.toFixed(2)}
            </span>
          </div>

          <button
            onClick={() => void handleManualRefresh()}
            disabled={isLoading}
            title="Recalculate pairwise similarities"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-card)',
              color: 'var(--text-secondary)',
              fontSize: '12px',
              fontWeight: 600,
              cursor: isLoading ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {isLoading ? (
              <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} />
            ) : (
              <RotateCcw size={13} />
            )}
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div
          style={{
            padding: '12px 16px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--danger-bg)',
            border: '1px solid rgba(239, 68, 68, 0.35)',
            color: '#fca5a5',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '13px',
          }}
        >
          <AlertCircle size={16} style={{ color: 'var(--danger)', flexShrink: 0 }} />
          <div style={{ flex: 1 }}>{error}</div>
          <button
            onClick={() => void handleManualRefresh()}
            style={{ color: '#ffffff', textDecoration: 'underline', fontWeight: 600, fontSize: '12px' }}
          >
            Retry
          </button>
        </div>
      )}

      {/* Zero Edges Banner */}
      {!isLoading && !error && backendNodes.length > 0 && backendEdges.length === 0 && (
        <div
          style={{
            padding: '10px 16px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            fontSize: '12px',
          }}
        >
          <Info size={15} style={{ color: 'var(--accent-sky)', flexShrink: 0 }} />
          <span>
            No patent pairs meet the current similarity threshold (&ge; {similarityThreshold.toFixed(2)}).
            Lower the slider to reveal weaker cosine relationships between the candidates.
          </span>
        </div>
      )}

      {/* Main Graph Canvas + Side Details Drawer */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0, gap: '16px' }}>
        {/* Canvas Area */}
        <div
          style={{
            flex: 1,
            height: '100%',
            backgroundColor: '#070b14',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-card)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {isLoading && (
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(8, 12, 21, 0.65)',
                backdropFilter: 'blur(4px)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
                gap: '10px',
                color: 'var(--accent-cyan-light)',
                fontSize: '13px',
              }}
            >
              <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
              <span>Computing Pairwise Vector Cosine Similarities...</span>
            </div>
          )}

          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={(_, node) => setSelectedPatentId(node.id)}
            onPaneClick={() => setSelectedPatentId(null)}
            fitView
            style={{ width: '100%', height: '100%' }}
          >
            <Background color="#1e293b" gap={20} size={1} />
            <Controls
              style={{
                backgroundColor: '#0f172a',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                fill: '#94a3b8',
              }}
            />
            <MiniMap
              nodeColor={(node) => {
                const bNode = backendNodes.find((b) => b.id === node.id);
                return getIpcColor(bNode?.ipc_class ?? null).border;
              }}
              style={{
                backgroundColor: '#0b1120',
                border: '1px solid rgba(148, 163, 184, 0.15)',
                borderRadius: '8px',
                width: 140,
                height: 90,
              }}
            />
          </ReactFlow>
        </div>

        {/* Selected Patent Detail Side Panel */}
        {selectedNodeData ? (
          <aside
            style={{
              width: '340px',
              minWidth: '340px',
              backgroundColor: 'var(--bg-card)',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border-card)',
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '16px',
              overflowY: 'auto',
              boxShadow: 'var(--shadow-md)',
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span
                  style={{
                    fontSize: '10px',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    color: 'var(--accent-sky)',
                    letterSpacing: '0.5px',
                  }}
                >
                  Selected Patent
                </span>
                {selectedSearchResult && (
                  <span
                    style={{
                      fontSize: '10px',
                      fontWeight: 700,
                      padding: '1px 6px',
                      borderRadius: '4px',
                      backgroundColor: 'rgba(6, 182, 212, 0.15)',
                      color: 'var(--accent-cyan-light)',
                    }}
                  >
                    Rank #{selectedSearchResult.rank}
                  </span>
                )}
              </div>

              <button
                onClick={() => setSelectedPatentId(null)}
                style={{ color: 'var(--text-muted)', cursor: 'pointer' }}
                title="Deselect node"
              >
                <X size={16} />
              </button>
            </div>

            {/* Patent ID row */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '18px', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                {selectedNodeData.patent_id}
              </span>
              <button
                onClick={() => handleCopyId(selectedNodeData.patent_id)}
                title="Copy Patent ID"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  color: hasCopiedId ? 'var(--success)' : 'var(--text-muted)',
                  backgroundColor: hasCopiedId ? 'var(--success-bg)' : 'rgba(255, 255, 255, 0.05)',
                  cursor: 'pointer',
                }}
              >
                {hasCopiedId ? <Check size={12} /> : <Copy size={12} />}
                <span>{hasCopiedId ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            {/* Classification badges */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  fontFamily: 'var(--font-mono)',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  backgroundColor: 'rgba(56, 189, 248, 0.1)',
                  color: 'var(--accent-sky)',
                  border: '1px solid rgba(56, 189, 248, 0.25)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                <Compass size={11} />
                IPC Class: {selectedNodeData.ipc_class ?? 'N/A'}
              </span>

              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  fontFamily: 'var(--font-mono)',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  backgroundColor: 'rgba(148, 163, 184, 0.1)',
                  color: 'var(--text-secondary)',
                  border: '1px solid rgba(148, 163, 184, 0.2)',
                }}
              >
                Subclass: {selectedNodeData.subclass ?? 'N/A'}
              </span>

              {selectedNodeData.label !== null && (
                <span
                  style={{
                    fontSize: '11px',
                    fontFamily: 'var(--font-mono)',
                    padding: '3px 8px',
                    borderRadius: '4px',
                    backgroundColor: 'rgba(148, 163, 184, 0.08)',
                    color: 'var(--text-muted)',
                    border: '1px solid rgba(148, 163, 184, 0.12)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '3px',
                  }}
                >
                  <Tag size={10} />
                  Dataset Label: {selectedNodeData.label}
                </span>
              )}
            </div>

            {/* Search scores if available */}
            {selectedSearchResult && (
              <div
                style={{
                  padding: '12px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: 'var(--bg-input)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                  fontSize: '12px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Search Hybrid Score:</span>
                  <strong style={{ color: 'var(--accent-cyan-light)', fontFamily: 'var(--font-mono)' }}>
                    {selectedSearchResult.result.hybrid_score.toFixed(4)}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Semantic Similarity:</span>
                  <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    {selectedSearchResult.result.similarity.toFixed(4)}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Chunk Similarity:</span>
                  <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    {selectedSearchResult.result.chunk_similarity.toFixed(4)}
                  </span>
                </div>
              </div>
            )}

            {/* Pairwise similarity links to other patents */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Direct Similarity Connections ({connectedEdges.length})
              </div>

              {connectedEdges.length === 0 ? (
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  No connections meeting &ge; {similarityThreshold.toFixed(2)}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '180px', overflowY: 'auto' }}>
                  {connectedEdges.map((conn, i) => (
                    <button
                      key={i}
                      onClick={() => setSelectedPatentId(conn.otherPatentId)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '8px 10px',
                        borderRadius: 'var(--radius-sm)',
                        backgroundColor: 'var(--bg-surface)',
                        border: '1px solid var(--border-subtle)',
                        fontSize: '12px',
                        cursor: 'pointer',
                        textAlign: 'left',
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'rgba(6, 182, 212, 0.4)')}
                      onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-subtle)')}
                    >
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                        {conn.otherPatentId}
                      </span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan-light)', fontWeight: 600 }}>
                        {conn.similarity.toFixed(4)}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Matched Chunk excerpt if from search results */}
            {selectedSearchResult?.result.matched_chunk && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-sky)' }}>
                  Matched Text Chunk
                </div>
                <div
                  style={{
                    padding: '10px 12px',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: 'var(--bg-input)',
                    border: '1px solid var(--border-subtle)',
                    fontSize: '12px',
                    lineHeight: 1.5,
                    color: 'var(--text-muted)',
                    maxHeight: '130px',
                    overflowY: 'auto',
                  }}
                >
                  &ldquo;{selectedSearchResult.result.matched_chunk}&rdquo;
                </div>
              </div>
            )}
          </aside>
        ) : (
          <aside
            style={{
              width: '280px',
              minWidth: '280px',
              backgroundColor: 'var(--bg-card)',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border-card)',
              padding: '24px 20px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              gap: '12px',
              color: 'var(--text-muted)',
            }}
          >
            <Sparkles size={22} style={{ color: 'var(--accent-cyan)', opacity: 0.7 }} />
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Interactive Node Inspector
            </div>
            <p style={{ fontSize: '12px', lineHeight: 1.5 }}>
              Click any patent node in the graph to inspect its IPC classification, search scores, and pairwise cosine connections.
            </p>
          </aside>
        )}
      </div>
    </div>
  );
};
