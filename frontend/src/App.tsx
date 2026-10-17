import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { SemanticSearch } from './components/SemanticSearch';
import { RelationshipExplorer } from './components/RelationshipExplorer';
import { OverviewView } from './components/OverviewView';
import { ComingSoonView } from './components/ComingSoonView';
import type { ActiveView, SearchResponse } from './types/patent';

export function App() {
  const [activeView, setActiveView] = useState<ActiveView>('search');
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);

  return (
    <div
      style={{
        display: 'flex',
        width: '100%',
        minHeight: '100vh',
        backgroundColor: 'var(--bg-main)',
        color: 'var(--text-primary)',
      }}
    >
      {/* Persistent Left Navigation Sidebar */}
      <Sidebar activeView={activeView} onSelectView={setActiveView} />

      {/* Main Workspace */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          height: '100vh',
          overflowY: 'auto',
        }}
      >
        <Header activeView={activeView} />

        <main style={{ flex: 1, padding: '32px 40px' }}>
          {activeView === 'search' && (
            <SemanticSearch
              searchResponse={searchResponse}
              onSearchResponse={setSearchResponse}
            />
          )}

          {activeView === 'explorer' && (
            <RelationshipExplorer
              searchResponse={searchResponse}
              onNavigateToSearch={() => setActiveView('search')}
            />
          )}

          {activeView === 'overview' && (
            <OverviewView onNavigateToSearch={() => setActiveView('search')} />
          )}

          {activeView !== 'search' &&
            activeView !== 'explorer' &&
            activeView !== 'overview' && (
              <ComingSoonView
                view={activeView}
                onNavigateToSearch={() => setActiveView('search')}
              />
            )}
        </main>
      </div>
    </div>
  );
}

export default App;
