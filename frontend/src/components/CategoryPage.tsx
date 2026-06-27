import { useEffect, useState } from 'react';
import {
  getCategory,
  getCategoryConsensus,
  ytLink,
  ytThumb,
  fmtTime,
  type CategoryDetail,
  type Intelligence,
} from '../api';
import PullCard from './PullCard';

export default function CategoryPage(
  { slug, onBack, onSelectCreator }:
  { slug: string; onBack: () => void; onSelectCreator: (brainId: string) => void },
) {
  const [cat, setCat] = useState<CategoryDetail | null>(null);
  const [tab, setTab] = useState<'creators' | 'consensus'>('creators');
  const [intel, setIntel] = useState<Intelligence | null>(null);

  useEffect(() => { getCategory(slug).then(setCat); }, [slug]);
  useEffect(() => {
    if (tab === 'consensus' && !intel) getCategoryConsensus(slug).then(setIntel);
  }, [tab, slug, intel]);

  if (!cat) return <div className="center-message">Loading…</div>;
  return (
    <div className="category-page">
      <button className="btn btn--secondary btn--small" onClick={onBack} type="button">
        &larr; categories
      </button>
      <h1>{cat.name}</h1>
      <div className="tabs">
        <button
          className={`tab ${tab === 'creators' ? 'tab--active' : ''}`}
          onClick={() => setTab('creators')}
          type="button"
        >
          Creators
        </button>
        <button
          className={`tab ${tab === 'consensus' ? 'tab--active' : ''}`}
          onClick={() => setTab('consensus')}
          type="button"
        >
          Consensus
        </button>
      </div>

      {tab === 'creators' && (
        <div className="creator-grid">
          {cat.creators.map((cr) =>
            cr.pulled ? (
              <button
                key={cr.handle}
                className="creator-card"
                onClick={() => onSelectCreator(cr.brain_id!)}
                type="button"
              >
                {cr.latest_video && (
                  <img
                    className="creator-card__thumb"
                    src={ytThumb(cr.latest_video)}
                    alt=""
                    loading="lazy"
                  />
                )}
                <h3>{cr.name}</h3>
                <span className="muted">
                  {cr.video_count} videos · {cr.status}
                </span>
              </button>
            ) : (
              <PullCard key={cr.handle} handle={cr.handle} url={cr.url!} />
            ),
          )}
        </div>
      )}

      {tab === 'consensus' &&
        (!intel ? (
          <div className="center-message">Computing consensus…</div>
        ) : intel.total_observations === 0 || intel.consensus.length === 0 ? (
          <p className="center-message">
            No shared observations yet — pull more creators in this category.
          </p>
        ) : (
          <div className="intel">
            {intel.consensus.map((theme, i) => (
              <div className="intel__theme" key={i}>
                <div className="intel__theme-head">
                  <span className="intel__label">{theme.label}</span>
                  <span className="intel__count">
                    {theme.founders}/{theme.total_founders} creators
                  </span>
                </div>
                <ul className="intel__evidence">
                  {theme.evidence.map((e, j) => (
                    <li key={j}>
                      <span className="intel__creator">{e.creator}</span>
                      <span className="intel__quote">"{e.quote}"</span>
                      {e.youtube_id && (
                        <a
                          className="intel__ts"
                          href={ytLink(e.youtube_id, e.start_time)}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {fmtTime(e.start_time)}
                        </a>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        ))}
    </div>
  );
}
