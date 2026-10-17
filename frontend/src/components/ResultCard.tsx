import React, { useState } from 'react';
import type { PatentResult } from '../types/patent';
import {
  FileText,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Tag,
  Compass,
  Sliders,
  Sparkles,
} from 'lucide-react';

interface ResultCardProps {
  result: PatentResult;
  rank: number;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, rank }) => {
  const [isExcerptExpanded, setIsExcerptExpanded] = useState(false);
  const [isDetailsExpanded, setIsDetailsExpanded] = useState(false);
  const [hasCopiedId, setHasCopiedId] = useState(false);
  const [hasCopiedChunk, setHasCopiedChunk] = useState(false);

  const copyPatentId = () => {
    navigator.clipboard.writeText(result.patent_id);
    setHasCopiedId(true);
    setTimeout(() => setHasCopiedId(false), 2000);
  };

  const copyChunk = () => {
    if (result.matched_chunk) {
      navigator.clipboard.writeText(result.matched_chunk);
      setHasCopiedChunk(true);
      setTimeout(() => setHasCopiedChunk(false), 2000);
    }
  };

  const matchedChunkText = result.matched_chunk || '';
  const isExcerptLong = matchedChunkText.length > 280;

  return (
    <article
      style={{
        backgroundColor: 'var(--bg-card)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-card)',
        padding: '20px 24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        transition: 'all 0.2s ease',
        boxShadow: 'var(--shadow-sm)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.35)';
        e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)';
        e.currentTarget.style.boxShadow = 'var(--shadow-md)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--border-card)';
        e.currentTarget.style.backgroundColor = 'var(--bg-card)';
        e.currentTarget.style.boxShadow = 'var(--shadow-sm)';
      }}
    >
      {/* Top Header Row: Rank, Patent ID, Primary Hybrid Score */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          {/* Rank Badge */}
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '28px',
              height: '28px',
              borderRadius: '8px',
              backgroundColor: 'rgba(6, 182, 212, 0.15)',
              color: 'var(--accent-cyan-light)',
              border: '1px solid rgba(6, 182, 212, 0.35)',
              fontSize: '13px',
              fontWeight: 700,
              fontFamily: 'var(--font-mono)',
            }}
          >
            #{rank}
          </span>

          {/* Patent ID with Copy Button */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                fontSize: '17px',
                fontWeight: 700,
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-mono)',
                letterSpacing: '-0.2px',
              }}
            >
              {result.patent_id}
            </span>
            <button
              onClick={copyPatentId}
              title="Copy Patent ID to clipboard"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '4px',
                borderRadius: '4px',
                color: hasCopiedId ? 'var(--success)' : 'var(--text-muted)',
                backgroundColor: hasCopiedId ? 'var(--success-bg)' : 'transparent',
                transition: 'all 0.15s ease',
              }}
            >
              {hasCopiedId ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>

          {/* Explicit Metadata Badges without Invented Names */}
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
            {result.ipc_class && (
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
                IPC Class: {result.ipc_class}
              </span>
            )}

            {result.subclass && (
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
                Subclass: {result.subclass}
              </span>
            )}

            {result.label !== null && result.label !== undefined && (
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
                Label ID: {result.label}
              </span>
            )}
          </div>
        </div>

        {/* Primary Relevance Indicator (Hybrid Score) */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '6px 14px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'rgba(6, 182, 212, 0.08)',
            border: '1px solid rgba(6, 182, 212, 0.25)',
          }}
        >
          <div>
            <div style={{ fontSize: '10px', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 600 }}>
              Hybrid Score
            </div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: 700,
                color: 'var(--accent-cyan-light)',
                fontFamily: 'var(--font-mono)',
                lineHeight: 1.1,
              }}
            >
              {result.hybrid_score.toFixed(4)}
            </div>
          </div>

          {/* Miniature Visual Score Bar */}
          <div
            style={{
              width: '48px',
              height: '6px',
              borderRadius: '3px',
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${Math.min(Math.max(result.hybrid_score * 100, 0), 100)}%`,
                background: 'linear-gradient(90deg, #0284c7, #22d3ee)',
                borderRadius: '3px',
              }}
            />
          </div>
        </div>
      </div>

      {/* Score Components Breakdown Row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '24px',
          flexWrap: 'wrap',
          padding: '8px 12px',
          borderRadius: 'var(--radius-sm)',
          backgroundColor: 'rgba(15, 23, 42, 0.5)',
          border: '1px solid var(--border-subtle)',
          fontSize: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: 'var(--text-muted)' }}>Semantic Similarity:</span>
          <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {result.similarity.toFixed(4)}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: 'var(--text-muted)' }}>Deep Chunk Similarity:</span>
          <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            {result.chunk_similarity.toFixed(4)}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '11px' }}>
          <Sliders size={12} />
          <span>Formula: 0.5 &times; Sim + 0.5 &times; Chunk</span>
        </div>
      </div>

      {/* Matched Text Excerpt Box */}
      {matchedChunkText ? (
        <div
          style={{
            padding: '14px 16px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--bg-input)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '12px',
              fontWeight: 600,
              color: 'var(--accent-sky)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <FileText size={14} />
              <span>Highest-Scoring Text Chunk</span>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 400 }}>
                ({matchedChunkText.length} chars)
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <button
                onClick={copyChunk}
                title="Copy chunk excerpt"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  color: hasCopiedChunk ? 'var(--success)' : 'var(--text-muted)',
                  fontSize: '11px',
                  cursor: 'pointer',
                }}
              >
                {hasCopiedChunk ? <Check size={12} /> : <Copy size={12} />}
                <span>{hasCopiedChunk ? 'Copied' : 'Copy'}</span>
              </button>

              {isExcerptLong && (
                <button
                  onClick={() => setIsExcerptExpanded(!isExcerptExpanded)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    color: 'var(--text-secondary)',
                    fontSize: '11px',
                    cursor: 'pointer',
                  }}
                >
                  <span>{isExcerptExpanded ? 'Collapse' : 'Expand full chunk'}</span>
                  {isExcerptExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                </button>
              )}
            </div>
          </div>

          <p
            className={!isExcerptExpanded && isExcerptLong ? 'line-clamp-3' : 'line-clamp-none'}
            style={{
              fontSize: '13.5px',
              lineHeight: 1.65,
              color: 'var(--text-secondary)',
              fontStyle: 'normal',
              wordBreak: 'break-word',
            }}
          >
            &ldquo;{matchedChunkText}&rdquo;
          </p>
        </div>
      ) : (
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
          No chunk excerpt text available for this patent candidate.
        </div>
      )}

      {/* Bottom Technical Details Drawer Toggle */}
      <div>
        <button
          onClick={() => setIsDetailsExpanded(!isDetailsExpanded)}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px',
            color: 'var(--text-muted)',
            cursor: 'pointer',
            padding: '2px 0',
            transition: 'color 0.15s ease',
          }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--accent-cyan-light)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
        >
          <Sparkles size={13} />
          <span>{isDetailsExpanded ? 'Hide Technical Metadata' : 'View Technical Metadata'}</span>
          {isDetailsExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {isDetailsExpanded && (
          <div
            style={{
              marginTop: '10px',
              padding: '12px 14px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'rgba(11, 17, 32, 0.7)',
              border: '1px solid var(--border-subtle)',
              fontSize: '12px',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '12px',
            }}
          >
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Patent ID: </span>
              <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>{result.patent_id}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>IPC Class: </span>
              <span style={{ color: 'var(--text-primary)' }}>{result.ipc_class ?? 'Unspecified'}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Subclass: </span>
              <span style={{ color: 'var(--text-primary)' }}>{result.subclass ?? 'Unspecified'}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Dataset Label: </span>
              <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                {result.label !== null && result.label !== undefined ? result.label : 'None'}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Similarity Delta: </span>
              <span style={{ color: 'var(--accent-cyan-light)', fontFamily: 'var(--font-mono)' }}>
                {(result.chunk_similarity - result.similarity >= 0 ? '+' : '') +
                  (result.chunk_similarity - result.similarity).toFixed(4)}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Chunk Length: </span>
              <span style={{ color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                {matchedChunkText.length} characters
              </span>
            </div>
          </div>
        )}
      </div>
    </article>
  );
};
