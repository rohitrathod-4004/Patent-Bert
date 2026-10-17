import React, { useEffect, useState, useCallback } from 'react';
import { getHealth } from '../api/client';
import { ShieldCheck, AlertCircle, RefreshCw, Loader2 } from 'lucide-react';
import type { ActiveView } from '../types/patent';

interface HeaderProps {
  activeView: ActiveView;
}

const VIEW_TITLES: Record<ActiveView, { title: string; breadcrumb: string }> = {
  search: { title: 'Semantic Patent Retrieval', breadcrumb: 'Search / Vector Match & Chunk Rerank' },
  overview: { title: 'System Overview & Architecture', breadcrumb: 'System / Architecture & Embeddings' },
  explorer: { title: 'Relationship Explorer', breadcrumb: 'Graph / Pairwise Patent Similarity' },
  analytics: { title: 'Corpus Analytics', breadcrumb: 'Analytics / Distribution & IPC Trends' },
  classification: { title: 'Patent Classification', breadcrumb: 'ML / IPC Class Prediction' },
};

export const Header: React.FC<HeaderProps> = ({ activeView }) => {
  const [healthStatus, setHealthStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking');
  const [isRetrying, setIsRetrying] = useState(false);

  const checkConnection = useCallback(async (isManual = false) => {
    if (isManual) setIsRetrying(true);
    try {
      const data = await getHealth();
      if (data && data.status === 'healthy') {
        setHealthStatus('healthy');
      } else {
        setHealthStatus('unhealthy');
      }
    } catch {
      setHealthStatus('unhealthy');
    } finally {
      if (isManual) setIsRetrying(false);
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    const poll = async () => {
      try {
        const data = await getHealth();
        if (isMounted) {
          setHealthStatus(data && data.status === 'healthy' ? 'healthy' : 'unhealthy');
        }
      } catch {
        if (isMounted) {
          setHealthStatus('unhealthy');
        }
      }
    };

    void poll();
    // Poll every 8s when healthy, every 4s when disconnected to pick up server restart quickly
    const interval = setInterval(poll, healthStatus === 'healthy' ? 8000 : 4000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [healthStatus]);

  const currentView = VIEW_TITLES[activeView];

  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '16px 32px',
        backgroundColor: 'rgba(11, 17, 32, 0.85)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid var(--border-subtle)',
        position: 'sticky',
        top: 0,
        zIndex: 20,
      }}
    >
      <div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '2px' }}>
          {currentView.breadcrumb}
        </div>
        <h1
          style={{
            fontSize: '18px',
            fontWeight: 700,
            color: 'var(--text-primary)',
            letterSpacing: '-0.3px',
          }}
        >
          {currentView.title}
        </h1>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
        {/* Model Spec Badge */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            border: '1px solid var(--border-subtle)',
            fontSize: '12px',
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-secondary)',
          }}
        >
          <span style={{ color: 'var(--accent-cyan-light)', fontWeight: 600 }}>BGE-base-en-v1.5</span>
          <span style={{ color: 'var(--text-muted)' }}>|</span>
          <span>768d Cosine</span>
        </div>

        {/* Live Backend Connection Indicator with Explicit Retry */}
        <button
          onClick={() => void checkConnection(true)}
          disabled={isRetrying}
          title={
            healthStatus === 'healthy'
              ? 'Backend API is connected on 127.0.0.1:8000. Click to recheck.'
              : 'Backend API is unreachable. Click to attempt reconnection.'
          }
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor:
              healthStatus === 'healthy'
                ? 'var(--success-bg)'
                : healthStatus === 'unhealthy'
                ? 'var(--danger-bg)'
                : 'rgba(148, 163, 184, 0.08)',
            border: `1px solid ${
              healthStatus === 'healthy'
                ? 'rgba(16, 185, 129, 0.35)'
                : healthStatus === 'unhealthy'
                ? 'rgba(239, 68, 68, 0.4)'
                : 'var(--border-subtle)'
            }`,
            color:
              healthStatus === 'healthy'
                ? 'var(--success)'
                : healthStatus === 'unhealthy'
                ? '#f87171'
                : 'var(--text-secondary)',
            fontSize: '12px',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'all 0.15s ease',
          }}
        >
          {isRetrying ? (
            <>
              <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} />
              <span>Checking...</span>
            </>
          ) : healthStatus === 'healthy' ? (
            <>
              <ShieldCheck size={14} />
              <span>API Ready</span>
              <RefreshCw size={11} style={{ opacity: 0.6, marginLeft: '2px' }} />
            </>
          ) : healthStatus === 'unhealthy' ? (
            <>
              <AlertCircle size={14} style={{ color: 'var(--danger)' }} />
              <span>API Offline &bull; Retry</span>
              <RefreshCw size={11} style={{ opacity: 0.8, marginLeft: '2px' }} />
            </>
          ) : (
            <>
              <Loader2 size={13} style={{ animation: 'spin 1s linear infinite' }} />
              <span>Connecting to API...</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
