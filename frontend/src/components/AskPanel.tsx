import { useState } from 'react';
import { askBrain } from '../api';
import type { AnswerResponse } from '../api';
import CitationList from './CitationList';

const MODES = ['qa', 'article', 'playbook', 'summary', 'faq'] as const;

interface AskPanelProps {
  brainId: string;
}

export default function AskPanel({ brainId }: AskPanelProps) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<string>('qa');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await askBrain(brainId, query.trim(), mode);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ask-panel">
      <form className="ask-panel__form" onSubmit={handleSubmit}>
        <div className="ask-panel__input-row">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about this brain's videos..."
            disabled={loading}
            className="ask-panel__input"
          />
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            disabled={loading}
            className="ask-panel__mode"
          >
            {MODES.map((m) => (
              <option key={m} value={m}>
                {m.toUpperCase()}
              </option>
            ))}
          </select>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="btn btn--primary"
          >
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </div>
      </form>

      {loading && (
        <div className="ask-panel__loading">
          <div className="spinner" />
          <span>Searching and generating answer...</span>
        </div>
      )}

      {error && <p className="form-error">{error}</p>}

      {result && !loading && (
        <div className="ask-panel__result">
          <div className="ask-panel__answer">
            <div className="ask-panel__answer-meta">
              <span>Mode: {result.mode}</span>
              <span>
                {result.chunks_used} / {result.chunks_searched} chunks used
              </span>
            </div>
            <div className="ask-panel__answer-text">
              {result.answer.split('\n').map((line, i) => (
                <p key={i}>{line || ' '}</p>
              ))}
            </div>
          </div>
          <CitationList
            citations={result.citations}
            confidence={result.confidence}
          />
        </div>
      )}
    </div>
  );
}
