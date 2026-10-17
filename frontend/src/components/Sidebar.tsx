import React from 'react';
import {
  Search,
  Share2,
  BarChart3,
  Tag,
  Layers,
  Database,
  Sparkles,
} from 'lucide-react';
import type { ActiveView } from '../types/patent';

interface SidebarProps {
  activeView: ActiveView;
  onSelectView: (view: ActiveView) => void;
}

interface NavItem {
  id: ActiveView;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string; style?: React.CSSProperties }>;
  isReady: boolean;
  badge?: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'search', label: 'Semantic Search', icon: Search, isReady: true },
  { id: 'explorer', label: 'Relationship Explorer', icon: Share2, isReady: true },
  { id: 'overview', label: 'Overview', icon: Layers, isReady: true },
  { id: 'analytics', label: 'Analytics', icon: BarChart3, isReady: false, badge: 'Soon' },
  { id: 'classification', label: 'Classification', icon: Tag, isReady: false, badge: 'Soon' },
];

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onSelectView }) => {
  return (
    <aside
      style={{
        width: '260px',
        minWidth: '260px',
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border-subtle)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        padding: '24px 16px',
        height: '100vh',
        position: 'sticky',
        top: 0,
      }}
    >
      <div>
        {/* Brand Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '4px 12px 24px 12px',
            borderBottom: '1px solid var(--border-subtle)',
            marginBottom: '20px',
          }}
        >
          <div
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 16px rgba(6, 182, 212, 0.35)',
              color: '#ffffff',
            }}
          >
            <Sparkles size={20} />
          </div>
          <div>
            <div
              style={{
                fontSize: '17px',
                fontWeight: 700,
                letterSpacing: '-0.3px',
                color: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              PatentBERT
              <span
                style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  backgroundColor: 'var(--accent-cyan-muted)',
                  color: 'var(--accent-cyan-light)',
                  border: '1px solid rgba(6, 182, 212, 0.3)',
                }}
              >
                v1.0
              </span>
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Neural Patent Intelligence
            </div>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <div
            style={{
              fontSize: '11px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.8px',
              color: 'var(--text-muted)',
              padding: '6px 12px',
            }}
          >
            Modules
          </div>

          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;

            return (
              <button
                key={item.id}
                onClick={() => onSelectView(item.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-md)',
                  backgroundColor: isActive ? 'var(--bg-surface)' : 'transparent',
                  color: isActive ? 'var(--accent-cyan-light)' : 'var(--text-secondary)',
                  border: isActive
                    ? '1px solid var(--border-cyan-glow)'
                    : '1px solid transparent',
                  fontWeight: isActive ? 600 : 500,
                  fontSize: '14px',
                  textAlign: 'left',
                  transition: 'all 0.15s ease',
                  cursor: 'pointer',
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
                    e.currentTarget.style.color = 'var(--text-primary)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = 'var(--text-secondary)';
                  }
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Icon
                    size={18}
                    style={{
                      color: isActive ? 'var(--accent-cyan-light)' : 'var(--text-muted)',
                    }}
                  />
                  <span>{item.label}</span>
                </div>

                {item.badge && (
                  <span
                    style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      backgroundColor: 'rgba(148, 163, 184, 0.1)',
                      color: 'var(--text-muted)',
                      border: '1px solid rgba(148, 163, 184, 0.15)',
                    }}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Index Metrics Footnote */}
      <div
        style={{
          padding: '14px',
          borderRadius: 'var(--radius-md)',
          backgroundColor: 'rgba(15, 23, 42, 0.7)',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '12px',
            fontWeight: 600,
            color: 'var(--text-primary)',
            marginBottom: '8px',
          }}
        >
          <Database size={14} style={{ color: 'var(--accent-cyan)' }} />
          <span>Active Vector Index</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Corpus Size:</span>
            <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>200,380 patents</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Embeddings:</span>
            <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>768-d (BGE v1.5)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>Index Structure:</span>
            <span style={{ color: 'var(--accent-cyan-light)', fontFamily: 'var(--font-mono)' }}>FAISS IndexFlatIP</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
