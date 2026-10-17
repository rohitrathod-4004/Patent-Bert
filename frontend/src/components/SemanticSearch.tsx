import React, { useState, useEffect, useRef } from 'react';
import { searchPatents } from '../api/client';
import type { PatentResult, SearchResponse } from '../types/patent';
import { ResultCard } from './ResultCard';
import { SearchAnalytics } from './SearchAnalytics';
import {
  Search,
  Sparkles,
  AlertTriangle,
  RotateCcw,
  CheckCircle2,
  Clock,
  Database,
  Hash,
  Loader2,
} from 'lucide-react';

const EXAMPLE_QUERIES = [
  'flexible fluid sealing gasket for high pressure compressor conduit',
  'lithium-ion battery electrolyte with thermal runaway protection',
  'semiconductor wafer chemical mechanical planarization slurry',
  'neural network architecture for real-time video super-resolution',
  'biodegradable polymer composite for orthopedic surgical implants',
];

interface SemanticSearchProps {
  searchResponse?: SearchResponse | null;
  onSearchResponse?: (response: SearchResponse | null) => void;
}

export const SemanticSearch: React.FC<SemanticSearchProps> = ({
  searchResponse: propSearchResponse,
  onSearchResponse,
}) => {
  const [query, setQuery] = useState(propSearchResponse?.query || '');
  const [candidateK, setCandidateK] = useState(propSearchResponse?.candidate_k || 10);
  const [topK, setTopK] = useState(propSearchResponse?.top_k || 5);
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedRunningSec, setElapsedRunningSec] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [localSearchResponse, setLocalSearchResponse] = useState<SearchResponse | null>(null);
  const [searchDurationMs, setSearchDurationMs] = useState<number | null>(null);

  const searchResponse = propSearchResponse !== undefined ? propSearchResponse : localSearchResponse;

  const updateSearchResponse = (resp: SearchResponse | null) => {
    setLocalSearchResponse(resp);
    if (onSearchResponse) {
      onSearchResponse(resp);
    }
  };

  const timerRef = useRef<number | null>(null);

  // Live timer tick during search to keep UI responsive
  useEffect(() => {
    if (!isLoading) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    const start = performance.now();
    timerRef.current = window.setInterval(() => {
      setElapsedRunningSec(parseFloat(((performance.now() - start) / 1000).toFixed(1)));
    }, 100);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isLoading]);

  const handleCandidateKChange = (val: number) => {
    const clamped = Math.max(1, Math.min(20, val));
    setCandidateK(clamped);
    if (topK > clamped) {
      setTopK(clamped);
    }
  };

  const handleTopKChange = (val: number) => {
    const clamped = Math.max(1, Math.min(candidateK, val));
    setTopK(clamped);
  };

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();

    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setError('Please enter a descriptive patent search query before searching.');
      return;
    }

    setElapsedRunningSec(0);
    setIsLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const response = await searchPatents({
        query: trimmedQuery,
        candidate_k: candidateK,
        top_k: topK,
      });

      const elapsed = Math.round(performance.now() - startTime);
      setSearchDurationMs(elapsed);
      updateSearchResponse(response);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'An error occurred during search.';
      setError(msg);
      updateSearchResponse(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSearch();
    }
  };

  const clearSearch = () => {
    setQuery('');
    setError(null);
    updateSearchResponse(null);
    setSearchDurationMs(null);
  };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Intro Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
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
            Two-Stage Hybrid Search
          </span>
          <span style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
            200,380 Patent Corpus
          </span>
        </div>
        <h2
          style={{
            fontSize: '26px',
            fontWeight: 700,
            color: 'var(--text-primary)',
            letterSpacing: '-0.5px',
            marginBottom: '8px',
          }}
        >
          Semantic Patent Intelligence Search
        </h2>
        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.6, maxWidth: '780px' }}>
          Retrieve conceptually relevant patents using dense BGE vector representations, then rerank candidates
          via localized chunk tokenization with sliding-window cosine similarity.
        </p>
      </div>

      {/* Main Search Input Form Card */}
      <div
        style={{
          backgroundColor: 'var(--bg-card)',
          borderRadius: 'var(--radius-xl)',
          border: '1px solid var(--border-card)',
          padding: '24px',
          boxShadow: 'var(--shadow-md)',
          display: 'flex',
          flexDirection: 'column',
          gap: '20px',
        }}
      >
        <form onSubmit={handleSearch} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Query Textarea */}
          <div style={{ position: 'relative' }}>
            <textarea
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                if (error) setError(null);
              }}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              placeholder="Describe patent invention, technical claims, industrial mechanisms, or specific apparatus in natural language..."
              rows={3}
              style={{
                width: '100%',
                backgroundColor: 'var(--bg-input)',
                border: '1px solid var(--border-card)',
                borderRadius: 'var(--radius-md)',
                padding: '16px 18px',
                fontSize: '15px',
                lineHeight: 1.6,
                color: 'var(--text-primary)',
                resize: 'vertical',
                transition: 'border-color 0.15s ease',
                opacity: isLoading ? 0.7 : 1,
              }}
              onFocus={(e) => (e.target.style.borderColor = 'var(--accent-cyan)')}
              onBlur={(e) => (e.target.style.borderColor = 'var(--border-card)')}
            />

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '8px',
                fontSize: '12px',
                color: 'var(--text-muted)',
              }}
            >
              <span>Press <kbd style={{ padding: '2px 5px', borderRadius: '4px', backgroundColor: 'rgba(255, 255, 255, 0.08)' }}>Ctrl</kbd> + <kbd style={{ padding: '2px 5px', borderRadius: '4px', backgroundColor: 'rgba(255, 255, 255, 0.08)' }}>Enter</kbd> to execute</span>
              {query.length > 0 && <span>{query.length} characters</span>}
            </div>
          </div>

          {/* Search Controls Toolbar: Sliders & Actions */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '16px',
              paddingTop: '8px',
              borderTop: '1px solid var(--border-subtle)',
            }}
          >
            {/* Parameters: candidate_k and top_k */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px', flexWrap: 'wrap' }}>
              {/* Candidate K */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Database size={13} style={{ color: 'var(--accent-cyan)' }} />
                  Candidates (k):
                </span>
                <input
                  type="range"
                  min={1}
                  max={20}
                  disabled={isLoading}
                  value={candidateK}
                  onChange={(e) => handleCandidateKChange(parseInt(e.target.value, 10))}
                  style={{ width: '90px', accentColor: 'var(--accent-cyan)' }}
                />
                <span
                  style={{
                    fontSize: '13px',
                    fontWeight: 600,
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--accent-cyan-light)',
                    minWidth: '22px',
                  }}
                >
                  {candidateK}
                </span>
              </div>

              {/* Top K */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ fontSize: '13px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Hash size={13} style={{ color: 'var(--accent-sky)' }} />
                  Results (top):
                </span>
                <input
                  type="range"
                  min={1}
                  max={candidateK}
                  disabled={isLoading}
                  value={topK}
                  onChange={(e) => handleTopKChange(parseInt(e.target.value, 10))}
                  style={{ width: '90px', accentColor: 'var(--accent-sky)' }}
                />
                <span
                  style={{
                    fontSize: '13px',
                    fontWeight: 600,
                    fontFamily: 'var(--font-mono)',
                    color: 'var(--accent-sky)',
                    minWidth: '22px',
                  }}
                >
                  {topK}
                </span>
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              {query && (
                <button
                  type="button"
                  onClick={clearSearch}
                  disabled={isLoading}
                  style={{
                    padding: '8px 14px',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-muted)',
                    fontSize: '13px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.15s ease',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-primary)')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
                >
                  <RotateCcw size={14} />
                  <span>Reset</span>
                </button>
              )}

              <button
                type="submit"
                disabled={isLoading}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '10px 22px',
                  borderRadius: 'var(--radius-md)',
                  background: isLoading
                    ? 'rgba(6, 182, 212, 0.4)'
                    : 'linear-gradient(135deg, #0284c7 0%, #06b6d4 100%)',
                  color: '#ffffff',
                  fontWeight: 600,
                  fontSize: '14px',
                  boxShadow: '0 4px 14px rgba(6, 182, 212, 0.3)',
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  if (!isLoading) {
                    e.currentTarget.style.transform = 'translateY(-1px)';
                    e.currentTarget.style.boxShadow = '0 6px 18px rgba(6, 182, 212, 0.45)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading) {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 4px 14px rgba(6, 182, 212, 0.3)';
                  }
                }}
              >
                {isLoading ? (
                  <>
                    <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
                    <span>Searching ({elapsedRunningSec}s)...</span>
                  </>
                ) : (
                  <>
                    <Search size={16} />
                    <span>Search Patents</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </form>

        {/* Example Query Chips */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', paddingTop: '4px' }}>
          <div
            style={{
              fontSize: '11px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.6px',
              color: 'var(--text-muted)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Sparkles size={12} style={{ color: 'var(--accent-cyan)' }} />
            <span>Example Invention Queries</span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {EXAMPLE_QUERIES.map((example, idx) => (
              <button
                key={idx}
                type="button"
                disabled={isLoading}
                onClick={() => {
                  setQuery(example);
                  if (error) setError(null);
                }}
                style={{
                  fontSize: '12px',
                  padding: '6px 12px',
                  borderRadius: '16px',
                  backgroundColor: 'rgba(15, 23, 42, 0.8)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                  textAlign: 'left',
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.15s ease',
                  opacity: isLoading ? 0.6 : 1,
                }}
                onMouseEnter={(e) => {
                  if (!isLoading) {
                    e.currentTarget.style.backgroundColor = 'var(--accent-cyan-muted)';
                    e.currentTarget.style.color = 'var(--accent-cyan-light)';
                    e.currentTarget.style.borderColor = 'rgba(6, 182, 212, 0.3)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading) {
                    e.currentTarget.style.backgroundColor = 'rgba(15, 23, 42, 0.8)';
                    e.currentTarget.style.color = 'var(--text-secondary)';
                    e.currentTarget.style.borderColor = 'var(--border-subtle)';
                  }
                }}
              >
                &ldquo;{example}&rdquo;
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Validation / Network Error Banner */}
      {error && (
        <div
          style={{
            padding: '14px 18px',
            borderRadius: 'var(--radius-md)',
            backgroundColor: 'var(--danger-bg)',
            border: '1px solid rgba(239, 68, 68, 0.35)',
            color: '#fca5a5',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            fontSize: '14px',
          }}
        >
          <AlertTriangle size={18} style={{ color: 'var(--danger)', flexShrink: 0 }} />
          <div style={{ flex: 1 }}>{error}</div>
        </div>
      )}

      {/* Loading State Indicator */}
      {isLoading && (
        <div
          style={{
            padding: '48px 24px',
            borderRadius: 'var(--radius-lg)',
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border-card)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '16px',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '50%',
              border: '3px solid rgba(6, 182, 212, 0.2)',
              borderTopColor: 'var(--accent-cyan)',
              animation: 'spin 1s linear infinite',
            }}
          />
          <div>
            <div style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '4px' }}>
              Scanning 200,380 Patent Embeddings ({elapsedRunningSec}s)
            </div>
            <div style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              FAISS vector retrieval in progress &bull; Tokenizing &amp; reranking chunks for top-{candidateK} candidates...
            </div>
          </div>
        </div>
      )}

      {/* Initial Empty State (Before First Search) */}
      {!isLoading && !searchResponse && !error && (
        <div
          style={{
            padding: '40px 32px',
            borderRadius: 'var(--radius-xl)',
            backgroundColor: 'rgba(15, 23, 42, 0.4)',
            border: '1px dashed var(--border-card)',
            display: 'flex',
            flexDirection: 'column',
            gap: '24px',
          }}
        >
          <div style={{ textAlign: 'center', maxWidth: '640px', margin: '0 auto' }}>
            <h3 style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '8px' }}>
              Ready for Natural-Language Patent Queries
            </h3>
            <p style={{ fontSize: '14px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              PatentBERT bypasses keyword brittle limitations by matching semantic conceptual intent across
              patent claims, specifications, and technical abstracts.
            </p>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
              gap: '16px',
            }}
          >
            <div
              style={{
                padding: '16px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-cyan-light)', marginBottom: '4px' }}>
                Stage 1: FAISS Vector Retrieval
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Query encoded into 768-d vector space and matched using fast inner-product cosine search.
              </div>
            </div>

            <div
              style={{
                padding: '16px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-sky)', marginBottom: '4px' }}>
                Stage 2: Chunk-Based Reranking
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Candidate patent texts are split into sliding-window token chunks (400 tokens, 80 overlap) for fine-grained max-pooled scoring.
              </div>
            </div>

            <div
              style={{
                padding: '16px',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div style={{ fontSize: '12px', fontWeight: 600, color: '#10b981', marginBottom: '4px' }}>
                Stage 3: Hybrid Fusion
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                Balanced 50/50 linear fusion of global patent similarity and specific localized chunk similarity.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results Section */}
      {!isLoading && searchResponse && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Summary / Metadata Toolbar */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '12px',
              padding: '12px 18px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--bg-card)',
              border: '1px solid var(--border-card)',
              fontSize: '13px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle2 size={16} style={{ color: 'var(--success)' }} />
              <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                Found {searchResponse.total_results} top matches
              </span>
              <span style={{ color: 'var(--text-muted)' }}>
                (reranked from {searchResponse.candidate_k} candidates)
              </span>
            </div>

            {searchDurationMs !== null && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  color: 'var(--text-secondary)',
                  fontSize: '12px',
                  fontFamily: 'var(--font-mono)',
                  backgroundColor: 'rgba(15, 23, 42, 0.6)',
                  padding: '4px 10px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <Clock size={13} style={{ color: 'var(--accent-cyan)' }} />
                <span>
                  Measured latency:{' '}
                  {searchDurationMs >= 1000
                    ? `${(searchDurationMs / 1000).toFixed(2)}s (${searchDurationMs.toLocaleString()} ms)`
                    : `${searchDurationMs} ms`}
                </span>
              </div>
            )}
          </div>

          {/* Result Cards List */}
          {searchResponse.results.length === 0 ? (
            <div
              style={{
                padding: '36px',
                textAlign: 'center',
                backgroundColor: 'var(--bg-card)',
                borderRadius: 'var(--radius-lg)',
                color: 'var(--text-muted)',
              }}
            >
              No matching patent candidates returned for this query. Try adjusting your search terms or increasing the candidate pool.
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {searchResponse.results.map((result: PatentResult, idx: number) => (
                <ResultCard key={result.patent_id || idx} result={result} rank={idx + 1} />
              ))}
            </div>
          )}

          {/* Search Analytics Section (Recharts) */}
          <SearchAnalytics results={searchResponse.results} />
        </div>
      )}
    </div>
  );
};
