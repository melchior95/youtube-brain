import type { Brain } from '../api';

const STATUS_COLORS: Record<string, string> = {
  pending: '#6b7280',
  ingesting: '#d97706',
  partially_ready: '#3b82f6',
  ready: '#22c55e',
  error: '#ef4444',
};

interface BrainCardProps {
  brain: Brain;
  onClick: () => void;
}

export default function BrainCard({ brain, onClick }: BrainCardProps) {
  const badgeColor = STATUS_COLORS[brain.status] ?? '#6b7280';

  return (
    <button className="brain-card" onClick={onClick} type="button">
      <div className="brain-card__header">
        <h3 className="brain-card__name">{brain.name}</h3>
        <span
          className="status-badge"
          style={{ backgroundColor: badgeColor }}
        >
          {brain.status}
        </span>
      </div>
      <div className="brain-card__meta">
        <span>{brain.video_count} video{brain.video_count !== 1 ? 's' : ''}</span>
        <span className="brain-card__date">
          {new Date(brain.created_at).toLocaleDateString()}
        </span>
      </div>
    </button>
  );
}
