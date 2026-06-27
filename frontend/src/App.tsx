import { useState } from 'react';
import './App.css';
import CategoryGrid from './components/CategoryGrid';
import CategoryPage from './components/CategoryPage';
import BrainDetail from './components/BrainDetail';

type View =
  | { v: 'home' }
  | { v: 'category'; slug: string }
  | { v: 'creator'; brainId: string; slug: string };

function App() {
  const [view, setView] = useState<View>({ v: 'home' });
  return (
    <div className="app">
      {view.v === 'home' && (
        <CategoryGrid onSelect={(slug) => setView({ v: 'category', slug })} />
      )}
      {view.v === 'category' && (
        <CategoryPage
          slug={view.slug}
          onBack={() => setView({ v: 'home' })}
          onSelectCreator={(brainId) => setView({ v: 'creator', brainId, slug: view.slug })}
        />
      )}
      {view.v === 'creator' && (
        <BrainDetail brainId={view.brainId} onBack={() => setView({ v: 'category', slug: view.slug })} />
      )}
    </div>
  );
}

export default App;
