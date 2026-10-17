import React from 'react';
import { Share2, BarChart3, Tag, ArrowRight, Clock } from 'lucide-react';
import type { ActiveView } from '../types/patent';

interface ComingSoonViewProps {
  view: ActiveView;
  onNavigateToSearch: () => void;
}

const MODULE_INFO: Record<
  string,
  {
    title: string;
    icon: React.ComponentType<{ size?: number; className?: string }>;
    phase: string;
    description: string;
    plannedFeatures: string[];
  }
> = {
  explorer: {
    title: 'Relationship Explorer',
    icon: Share2,
    phase: 'Phase 2 Architecture',
    description:
      'Pairwise patent cosine similarity networks, citation neighborhood projection, and vector graph topology will be integrated in Phase 2.',
    plannedFeatures: [
      'Interactive 2D/3D Force-Directed Patent Graph',
      'Pairwise Cosine Distance Matrix Calculation',
      'IPC Co-Occurrence & Citation Neighborhood Clustering',
      'Nearest-Neighbor Subgraph Traversal',
    ],
  },
  analytics: {
    title: 'Corpus-Wide Analytics',
    icon: BarChart3,
    phase: 'Phase 3 Milestone',
    description:
      'Global distribution analyses across all 200,380 patents in the index, including IPC section breakdowns, vocabulary metrics, and embedding density maps.',
    plannedFeatures: [
      'Corpus-wide IPC section & subclass density charts',
      'Embedding space dispersion & cluster coherence metrics',
      'Temporal distribution across patent filing splits',
      'Text chunk length distribution histograms',
    ],
  },
  classification: {
    title: 'Multi-Label Patent Classification',
    icon: Tag,
    phase: 'Phase 3 Milestone',
    description:
      'Automated IPC class and subclass prediction using PatentBERT contextual representation heads trained on patent claims.',
    plannedFeatures: [
      'Hierarchical IPC section/class/subclass prediction',
      'Confidence calibration for predicted class codes',
      'Multi-label claim classification head',
      'Out-of-distribution patent novelty detection',
    ],
  },
};

export const ComingSoonView: React.FC<ComingSoonViewProps> = ({
  view,
  onNavigateToSearch,
}) => {
  const info = MODULE_INFO[view] || {
    title: 'Planned Capability',
    icon: Clock,
    phase: 'Next Phase',
    description: 'This module is currently scheduled for subsequent development phases.',
    plannedFeatures: [],
  };

  const Icon = info.icon;

  return (
    <div
      style={{
        maxWidth: '800px',
        margin: '40px auto',
        padding: '36px',
        backgroundColor: 'var(--bg-card)',
        borderRadius: 'var(--radius-xl)',
        border: '1px solid var(--border-card)',
        boxShadow: 'var(--shadow-md)',
        display: 'flex',
        flexDirection: 'column',
        gap: '24px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            backgroundColor: 'var(--accent-cyan-muted)',
            color: 'var(--accent-cyan-light)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: '1px solid rgba(6, 182, 212, 0.3)',
          }}
        >
          <Icon size={24} />
        </div>
        <div>
          <div
            style={{
              fontSize: '11px',
              fontWeight: 600,
              textTransform: 'uppercase',
              color: 'var(--accent-sky)',
              letterSpacing: '0.6px',
            }}
          >
            {info.phase}
          </div>
          <h2 style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)' }}>
            {info.title}
          </h2>
        </div>
      </div>

      <p style={{ fontSize: '15px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
        {info.description}
      </p>

      {/* Planned Specs */}
      <div
        style={{
          padding: '20px',
          borderRadius: 'var(--radius-md)',
          backgroundColor: 'var(--bg-input)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <div
          style={{
            fontSize: '13px',
            fontWeight: 600,
            color: 'var(--text-primary)',
            marginBottom: '12px',
          }}
        >
          Planned Capabilities in Roadmap:
        </div>
        <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {info.plannedFeatures.map((feat, i) => (
            <li
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '13px',
                color: 'var(--text-muted)',
              }}
            >
              <span
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--accent-cyan)',
                }}
              />
              <span>{feat}</span>
            </li>
          ))}
        </ul>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '8px' }}>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Active Module in Phase 1: Semantic Search
        </span>

        <button
          onClick={onNavigateToSearch}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 18px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-card)',
            color: 'var(--accent-cyan-light)',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--accent-cyan-muted)';
            e.currentTarget.style.borderColor = 'rgba(6, 182, 212, 0.4)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = 'var(--bg-surface)';
            e.currentTarget.style.borderColor = 'var(--border-card)';
          }}
        >
          <span>Return to Semantic Search</span>
          <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
};
