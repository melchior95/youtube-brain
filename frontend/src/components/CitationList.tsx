import type { Citation, Confidence } from '../api';

interface CitationListProps {
  citations: Citation[];
  confidence: Confidence;
}

export default function CitationList({ citations, confidence }: CitationListProps) {
  if (citations.length === 0) return null;

  const confidenceColor =
    confidence.level === 'high'
      ? '#22c55e'
      : confidence.level === 'medium'
        ? '#d97706'
        : '#ef4444';

  return (
    <div className="citations">
      <div className="citations__confidence">
        <span
          className="status-badge"
          style={{ backgroundColor: confidenceColor }}
        >
          {confidence.level} confidence
        </span>
        <span className="citations__stats">
          {confidence.supporting_chunks} chunks from{' '}
          {confidence.supporting_videos} video
          {confidence.supporting_videos !== 1 ? 's' : ''}
          {' | '}
          {confidence.caption_quality.replace('_', ' ')}
        </span>
      </div>

      <h4 className="citations__heading">Sources</h4>
      <ul className="citations__list">
        {citations.map((c, i) => (
          <li key={`${c.chunk_id}-${i}`} className="citation-item">
            <div className="citation-item__header">
              <span className="citation-item__title">{c.video_title}</span>
              <a
                href={c.video_url}
                target="_blank"
                rel="noopener noreferrer"
                className="citation-item__timestamp"
              >
                {c.timestamp_display}
              </a>
            </div>
            <p className="citation-item__text">{c.transcript_text}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
