import React from 'react';
import {
  Layers,
  Sliders,
  ArrowRight,
  Zap,
} from 'lucide-react';

interface OverviewViewProps {
  onNavigateToSearch: () => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ onNavigateToSearch }) => {
  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '32px' }}>
      {/* Hero Header */}
      <div
        style={{
          padding: '36px',
          borderRadius: 'var(--radius-xl)',
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(11, 17, 32, 0.8) 100%)',
          border: '1px solid var(--border-card)',
          boxShadow: 'var(--shadow-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontSize: '11px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.6px',
              padding: '3px 8px',
              borderRadius: '4px',
              backgroundColor: 'var(--accent-cyan-muted)',
              color: 'var(--accent-cyan-light)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
            }}
          >
            System Architecture
          </span>
          <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            High-Throughput Vector Indexing
          </span>
        </div>

        <h2 style={{ fontSize: '28px', fontWeight: 700, color: '#ffffff', letterSpacing: '-0.5px' }}>
          PatentBERT Research Platform
        </h2>

        <p style={{ fontSize: '15px', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '760px' }}>
          PatentBERT combines deep contextual sentence embeddings with localized chunk-level tokenization to
          perform high-accuracy patent prior-art discovery, claim verification, and semantic search over 200,380
          industrial patent records.
        </p>

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
              fontSize: '14px',
              fontWeight: 600,
              boxShadow: '0 4px 14px rgba(6, 182, 212, 0.3)',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.transform = 'translateY(-1px)')}
            onMouseLeave={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
          >
            <span>Open Semantic Search</span>
            <ArrowRight size={16} />
          </button>
        </div>
      </div>

      {/* Metric Cards Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '16px',
        }}
      >
        <div
          style={{
            padding: '20px',
            borderRadius: 'var(--radius-lg)',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-card)',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Indexed Patents
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            200,380
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            USPTO &amp; WIPO Corpus
          </div>
        </div>

        <div
          style={{
            padding: '20px',
            borderRadius: 'var(--radius-lg)',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-card)',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Embedding Dimensions
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--accent-cyan-light)', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            768
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            BAAI/bge-base-en-v1.5
          </div>
        </div>

        <div
          style={{
            padding: '20px',
            borderRadius: 'var(--radius-lg)',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-card)',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Index Type
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--accent-sky)', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            IndexFlatIP
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Exact Cosine Distance (Metric 0)
          </div>
        </div>

        <div
          style={{
            padding: '20px',
            borderRadius: 'var(--radius-lg)',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-card)',
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>
            Hybrid Fusion Ratio
          </div>
          <div style={{ fontSize: '24px', fontWeight: 700, color: '#10b981', fontFamily: 'var(--font-mono)', margin: '4px 0' }}>
            &alpha; = 0.5
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            50% Global + 50% Chunk Max
          </div>
        </div>
      </div>

      {/* Technical Workflow Explainer */}
      <div
        style={{
          padding: '28px',
          borderRadius: 'var(--radius-xl)',
          backgroundColor: 'var(--bg-card)',
          border: '1px solid var(--border-card)',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
        }}
      >
        <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)' }}>
          Retrieval &amp; Reranking Pipeline
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px' }}>
          <div
            style={{
              padding: '18px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-input)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-cyan-light)', fontWeight: 600, marginBottom: '8px' }}>
              <Zap size={16} />
              <span>1. Global Dense Retrieval</span>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              The search query is projected into the 768-dimensional BGE embedding space with unit normalization.
              FAISS executes an inner-product scan to identify the top candidate patents.
            </p>
          </div>

          <div
            style={{
              padding: '18px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-input)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-sky)', fontWeight: 600, marginBottom: '8px' }}>
              <Layers size={16} />
              <span>2. Chunk Tokenization</span>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Each candidate patent specification is tokenized into sliding windows of 400 tokens with 80 tokens of
              overlap, preserving dense semantic continuity across patent claims.
            </p>
          </div>

          <div
            style={{
              padding: '18px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-input)',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#10b981', fontWeight: 600, marginBottom: '8px' }}>
              <Sliders size={16} />
              <span>3. Max-Pooled Reranking</span>
            </div>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
              Chunk embeddings are multiplied by the query vector to identify the most relevant passage.
              The overall ranking is determined by a linear hybrid fusion: 0.5 &times; Global Sim + 0.5 &times; Chunk Sim.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
