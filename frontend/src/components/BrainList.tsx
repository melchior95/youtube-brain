import { useEffect, useState, useCallback } from 'react';
import { listBrains } from '../api';
import type { Brain } from '../api';
import BrainCard from './BrainCard';
import IngestDialog from './IngestDialog';

interface BrainListProps {
  onSelectBrain: (id: string) => void;
}

export default function BrainList({ onSelectBrain }: BrainListProps) {
  const [brains, setBrains] = useState<Brain[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);

  const fetchBrains = useCallback(async () => {
    try {
      const data = await listBrains();
      setBrains(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load brains');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchBrains();
  }, [fetchBrains]);

  // Auto-refresh while any brain is ingesting
  useEffect(() => {
    const hasIngesting = brains.some(
      (b) => b.status === 'ingesting' || b.status === 'pending'
    );
    if (!hasIngesting) return;

    const interval = setInterval(fetchBrains, 5000);
    return () => clearInterval(interval);
  }, [brains, fetchBrains]);

  if (loading) {
    return (
      <div className="center-message">
        <div className="spinner" />
        <span>Loading brains...</span>
      </div>
    );
  }

  return (
    <div className="brain-list">
      <div className="brain-list__header">
        <h1>YouTube Brain</h1>
        <button
          className="btn btn--primary"
          onClick={() => setDialogOpen(true)}
          type="button"
        >
          + New Brain
        </button>
      </div>

      {error && <p className="form-error">{error}</p>}

      {brains.length === 0 ? (
        <div className="center-message">
          <p>No brains yet. Ingest a YouTube video or playlist to get started.</p>
        </div>
      ) : (
        <div className="brain-grid">
          {brains.map((brain) => (
            <BrainCard
              key={brain.id}
              brain={brain}
              onClick={() => onSelectBrain(brain.id)}
            />
          ))}
        </div>
      )}

      <IngestDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmitted={fetchBrains}
      />
    </div>
  );
}
