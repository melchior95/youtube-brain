import { useEffect, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { getBrain, getEditorial, ytThumb } from '../api';
import type { BrainDetail as BrainDetailType, Editorial } from '../api';
import AskPanel from './AskPanel';

const STATUS_COLORS: Record<string, string> = {
  pending: '#6b7280',
  ingesting: '#d97706',
  partially_ready: '#3b82f6',
  ready: '#22c55e',
  error: '#ef4444',
};

interface BrainDetailProps {
  brainId: string;
  onBack: () => void;
}

function fmtDuration(seconds: number | null): string {
  if (!seconds) return '';
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

export default function BrainDetail({ brainId, onBack }: BrainDetailProps) {
  const [brain, setBrain] = useState<BrainDetailType | null>(null);
  const [editorial, setEditorial] = useState<Editorial | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchBrain = useCallback(async () => {
    try {
      const data = await getBrain(brainId);
      setBrain(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load brain');
    } finally {
      setLoading(false);
    }
  }, [brainId]);

  useEffect(() => {
    fetchBrain();
  }, [fetchBrain]);

  useEffect(() => {
    getEditorial(brainId).then(setEditorial).catch(() => setEditorial(null));
  }, [brainId]);

  // Auto-refresh while brain is ingesting
  useEffect(() => {
    if (!brain) return;
    const isIngesting = brain.status === 'ingesting' || brain.status === 'pending';
    if (!isIngesting) return;
    const interval = setInterval(fetchBrain, 5000);
    return () => clearInterval(interval);
  }, [brain, fetchBrain]);

  if (loading) {
    return (
      <div className="center-message">
        <div className="spinner" />
        <span>Loading…</span>
      </div>
    );
  }

  if (error || !brain) {
    return (
      <div className="center-message">
        <p className="form-error">{error ?? 'Brain not found'}</p>
        <button className="btn btn--secondary" onClick={onBack} type="button">
          Back
        </button>
      </div>
    );
  }

  const badgeColor = STATUS_COLORS[brain.status] ?? '#6b7280';

  return (
    <div className="brain-detail">
      <div className="brain-detail__header">
        <button className="btn btn--secondary btn--small" onClick={onBack} type="button">
          &larr; Back
        </button>
        <h1>{brain.name}</h1>
        <span className="status-badge" style={{ backgroundColor: badgeColor }}>
          {brain.status}
        </span>
        <span className="brain-detail__count">
          {brain.video_count} video{brain.video_count !== 1 ? 's' : ''}
        </span>
      </div>

      {/* 1 — recent videos with rich summaries */}
      <section className="creator-section">
        <h2 className="creator-section__title">Recent videos</h2>
        <div className="summary-list">
          {brain.videos.length === 0 ? (
            <p className="center-message">No videos ingested yet.</p>
          ) : (
            brain.videos.map((v) => (
              <article key={v.id} className="summary-card">
                <div className="summary-card__head">
                  <a
                    href={v.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="summary-card__thumb"
                  >
                    <img src={ytThumb(v.video_id)} alt="" loading="lazy" />
                  </a>
                  <div>
                    <a
                      href={v.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="summary-card__title"
                    >
                      {v.title || v.video_id}
                    </a>
                    <div className="summary-card__meta">{fmtDuration(v.duration_seconds)}</div>
                  </div>
                </div>
                {v.video_summary ? (
                  <div className="summary-card__text markdown">
                    <ReactMarkdown>{v.video_summary}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="summary-card__text muted">Summary not generated yet.</p>
                )}
              </article>
            ))
          )}
        </div>
      </section>

      {/* 2 — ask this creator */}
      <section className="creator-section">
        <h2 className="creator-section__title">Ask {brain.name}</h2>
        <AskPanel brainId={brainId} />
      </section>

      {/* 3 — weekly insight / editorial */}
      <section className="creator-section">
        <h2 className="creator-section__title">Weekly insight</h2>
        {editorial?.body ? (
          <div className="markdown">
            {editorial.title && <h3>{editorial.title}</h3>}
            <ReactMarkdown>{editorial.body}</ReactMarkdown>
          </div>
        ) : (
          <p className="muted">No weekly insight generated yet.</p>
        )}
      </section>
    </div>
  );
}
