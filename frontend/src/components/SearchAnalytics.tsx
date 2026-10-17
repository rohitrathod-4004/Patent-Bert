import React from 'react';
import type { PatentResult } from '../types/patent';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { BarChart3, TrendingUp, Layers } from 'lucide-react';

interface SearchAnalyticsProps {
  results: PatentResult[];
}

interface ScoreTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
  }>;
  label?: string;
}

const CustomScoreTooltip: React.FC<ScoreTooltipProps> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  return (
    <div
      style={{
        backgroundColor: '#0b1120',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        padding: '12px 14px',
        borderRadius: '8px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
        fontSize: '12px',
      }}
    >
      <div
        style={{
          color: '#f8fafc',
          fontWeight: 700,
          marginBottom: '8px',
          fontFamily: 'var(--font-mono)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          paddingBottom: '4px',
        }}
      >
        {label}
      </div>
      {payload.map((item, idx) => (
        <div
          key={idx}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '16px',
            color: '#cbd5e1',
            margin: '3px 0',
          }}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '2px',
                backgroundColor: item.color,
              }}
            />
            {item.name}:
          </span>
          <span style={{ fontWeight: 600, color: '#f1f5f9', fontFamily: 'var(--font-mono)' }}>
            {typeof item.value === 'number' ? item.value.toFixed(4) : item.value}
          </span>
        </div>
      ))}
      <div style={{ marginTop: '6px', fontSize: '10px', color: '#64748b', fontStyle: 'italic' }}>
        Hybrid = 0.5 &times; Semantic + 0.5 &times; Chunk
      </div>
    </div>
  );
};

interface IpcTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: {
      category: string;
      ipcClass: string;
      subclass: string;
      count: number;
      patents: string[];
    };
  }>;
}

const CustomIpcTooltip: React.FC<IpcTooltipProps> = ({ active, payload }) => {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0].payload;

  return (
    <div
      style={{
        backgroundColor: '#0b1120',
        border: '1px solid rgba(56, 189, 248, 0.3)',
        padding: '12px 14px',
        borderRadius: '8px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
        fontSize: '12px',
        maxWidth: '260px',
      }}
    >
      <div style={{ fontWeight: 700, color: '#f8fafc', marginBottom: '6px' }}>
        {data.category}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', color: '#94a3b8' }}>
        <div>
          IPC Class: <strong style={{ color: '#38bdf8' }}>{data.ipcClass}</strong>
        </div>
        <div>
          Subclass: <strong style={{ color: '#f1f5f9' }}>{data.subclass}</strong>
        </div>
        <div>
          Matching Candidates: <strong style={{ color: '#10b981' }}>{data.count}</strong>
        </div>
        <div style={{ marginTop: '4px', fontSize: '11px', color: '#64748b', wordBreak: 'break-all' }}>
          Patents: {data.patents.join(', ')}
        </div>
      </div>
    </div>
  );
};

export const SearchAnalytics: React.FC<SearchAnalyticsProps> = ({ results }) => {
  if (!results || results.length === 0) {
    return null;
  }

  // 1. Group retrieved results by IPC Class and Subclass
  const ipcMap: Record<
    string,
    { category: string; ipcClass: string; subclass: string; count: number; patents: string[] }
  > = {};

  results.forEach((r) => {
    const cls = r.ipc_class ? `Class ${r.ipc_class}` : 'No Class';
    const sub = r.subclass ? `Sub ${r.subclass}` : 'No Subclass';
    const groupKey = `${cls} / ${sub}`;

    if (!ipcMap[groupKey]) {
      ipcMap[groupKey] = {
        category: groupKey,
        ipcClass: r.ipc_class ?? 'N/A',
        subclass: r.subclass ?? 'N/A',
        count: 0,
        patents: [],
      };
    }
    ipcMap[groupKey].count += 1;
    ipcMap[groupKey].patents.push(r.patent_id);
  });

  const ipcChartData = Object.values(ipcMap);

  // 2. Score Comparison Data per Patent (strictly derived from API results)
  const scoreChartData = results.map((r, i) => ({
    name: `#${i + 1} ${r.patent_id.length > 12 ? r.patent_id.slice(-8) : r.patent_id}`,
    'Semantic Sim (Global)': parseFloat(r.similarity.toFixed(4)),
    'Chunk Sim (Max Local)': parseFloat(r.chunk_similarity.toFixed(4)),
    'Hybrid Score (Fusion)': parseFloat(r.hybrid_score.toFixed(4)),
  }));

  return (
    <section
      style={{
        marginTop: '36px',
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <div
          style={{
            padding: '6px',
            borderRadius: '8px',
            backgroundColor: 'var(--accent-cyan-muted)',
            color: 'var(--accent-cyan-light)',
          }}
        >
          <BarChart3 size={18} />
        </div>
        <div>
          <h2
            style={{
              fontSize: '16px',
              fontWeight: 700,
              color: 'var(--text-primary)',
              letterSpacing: '-0.2px',
            }}
          >
            Retrieval Analytics & Score Distribution
          </h2>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Real-time analytics computed strictly from the returned {results.length} patent candidates.
          </p>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
          gap: '20px',
        }}
      >
        {/* Chart 1: Multi-Metric Score Comparison */}
        <div
          style={{
            backgroundColor: 'var(--bg-card)',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-card)',
            padding: '20px',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <TrendingUp size={15} style={{ color: 'var(--accent-cyan)' }} />
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Score Comparison Across Top Candidates
              </span>
            </div>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Similarity (0.0 to 1.0)
            </span>
          </div>

          <div style={{ width: '100%', height: 270 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={scoreChartData} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.08)" vertical={false} />
                <XAxis
                  dataKey="name"
                  stroke="#64748b"
                  fontSize={11}
                  tickLine={false}
                  interval={0}
                  angle={-15}
                  textAnchor="end"
                />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 1]} tickCount={6} tickLine={false} />
                <Tooltip content={<CustomScoreTooltip />} />
                <Legend
                  wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
                  iconSize={8}
                />
                <Bar dataKey="Semantic Sim (Global)" fill="#38bdf8" radius={[4, 4, 0, 0]} maxBarSize={22} />
                <Bar dataKey="Chunk Sim (Max Local)" fill="#06b6d4" radius={[4, 4, 0, 0]} maxBarSize={22} />
                <Bar dataKey="Hybrid Score (Fusion)" fill="#10b981" radius={[4, 4, 0, 0]} maxBarSize={22} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: IPC Section & Subclass Distribution */}
        <div
          style={{
            backgroundColor: 'var(--bg-card)',
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border-card)',
            padding: '20px',
            boxShadow: 'var(--shadow-sm)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={15} style={{ color: 'var(--accent-sky)' }} />
              <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Candidate Breakdown by IPC Class & Subclass
              </span>
            </div>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Patent Count
            </span>
          </div>

          <div style={{ width: '100%', height: 270 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ipcChartData} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.08)" vertical={false} />
                <XAxis
                  dataKey="category"
                  stroke="#64748b"
                  fontSize={11}
                  tickLine={false}
                  interval={0}
                />
                <YAxis
                  stroke="#64748b"
                  fontSize={11}
                  allowDecimals={false}
                  tickLine={false}
                />
                <Tooltip content={<CustomIpcTooltip />} />
                <Bar dataKey="count" name="Patents" fill="#0284c7" radius={[4, 4, 0, 0]} maxBarSize={36} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
};
