import { useState } from 'react';
import { ingestUrl } from '../api';

interface IngestDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmitted: () => void;
}

export default function IngestDialog({ open, onClose, onSubmitted }: IngestDialogProps) {
  const [url, setUrl] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    try {
      await ingestUrl(url.trim(), name.trim() || undefined);
      setUrl('');
      setName('');
      onSubmitted();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ingestion failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal__header">
          <h2>Ingest YouTube Video</h2>
          <button className="modal__close" onClick={onClose} type="button">
            &times;
          </button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="ingest-url">YouTube URL</label>
            <input
              id="ingest-url"
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=... or playlist URL"
              required
              disabled={loading}
            />
          </div>
          <div className="form-group">
            <label htmlFor="ingest-name">Brain Name (optional)</label>
            <input
              id="ingest-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Brain"
              disabled={loading}
            />
          </div>
          {error && <p className="form-error">{error}</p>}
          <div className="modal__actions">
            <button type="button" onClick={onClose} disabled={loading} className="btn btn--secondary">
              Cancel
            </button>
            <button type="submit" disabled={loading || !url.trim()} className="btn btn--primary">
              {loading ? 'Ingesting...' : 'Start Ingest'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
