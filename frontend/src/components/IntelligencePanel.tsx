import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  getIntelligence,
  getTimeline,
  getEditorial,
  ytLink,
  fmtTime,
} from '../api';
import type { Intelligence, Timeline, Editorial, ConsensusTheme } from '../api';

interface Props {
  brainId: string;
}

type SubTab = 'overview' | 'consensus' | 'timeline' | 'new' | 'editorial' | 'audit';

const SUBTABS: { key: SubTab; label: string }[] = [
  { key: 'overview', label: 'Overview' },
  { key: 'consensus', label: 'Consensus' },
  { key: 'timeline', label: 'Timeline' },
  { key: 'new', label: 'New This Period' },
  { key: 'editorial', label: 'Editorial Report' },
  { key: 'audit', label: 'Raw Audit' },
];

const TIER_ICON: Record<string, string> = {
  'Strong signal': '🔥',
  'Emerging pattern': '⚠️',
  Outlier: '💡',
};

function tierOf(founders: number, total: number): string {
  if (total <= 0) return 'Outlier';
  if (founders / total >= 0.45) return 'Strong signal';
  if (founders >= Math.max(2, Math.ceil(0.25 * total))) return 'Emerging pattern';
  return 'Outlier';
}

function EvidenceList({ theme }: { theme: ConsensusTheme }) {
  return (
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
  );
}

export default function IntelligencePanel({ brainId }: Props) {
  const [intel, setIntel] = useState<Intelligence | null>(null);
  const [timeline, setTimeline] = useState<Timeline | null>(null);
  const [editorial, setEditorial] = useState<Editorial | null>(null);
  const [sub, setSub] = useState<SubTab>('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    Promise.all([
      getIntelligence(brainId),
      getTimeline(brainId, 'month').catch(() => null),
      getEditorial(brainId).catch(() => null),
    ])
      .then(([i, t, e]) => {
        if (!active) return;
        setIntel(i);
        setTimeline(t);
        setEditorial(e);
      })
      .catch((err) => active && setError(err instanceof Error ? err.message : 'Failed'))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [brainId]);

  if (loading) {
    return (
      <div className="center-message">
        <div className="spinner" />
        <span>Loading intelligence...</span>
      </div>
    );
  }
  if (error) return <p className="form-error">{error}</p>;
  if (!intel || intel.total_observations === 0) {
    return (
      <p className="center-message">
        No observations yet. Extract observations for this brain to generate the report.
      </p>
    );
  }

  const nf = intel.founders;
  const consensus = intel.consensus.filter((t) => !t.label.startsWith('Revenue & traction'));
  const absent = intel.absences.filter((a) => a.founders === 0);

  return (
    <div className="intel">
      <div className="subtabs">
        {SUBTABS.map((t) => (
          <button
            key={t.key}
            className={`subtab ${sub === t.key ? 'subtab--active' : ''}`}
            onClick={() => setSub(t.key)}
            type="button"
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ---- Overview ---- */}
      {sub === 'overview' && (
        <div>
          <div className="intel__stats">
            <div className="stat"><span className="stat__n">{nf}</span><span className="stat__l">founders</span></div>
            <div className="stat"><span className="stat__n">{intel.total_observations}</span><span className="stat__l">observations</span></div>
            <div className="stat"><span className="stat__n">{consensus.length}</span><span className="stat__l">shared themes</span></div>
            {timeline && timeline.periods.length > 0 && (
              <div className="stat">
                <span className="stat__n">{timeline.periods.length}</span>
                <span className="stat__l">{timeline.granularity}s tracked</span>
              </div>
            )}
          </div>

          {consensus[0] && (
            <div className="intel__headline">
              <div className="intel__headline-tag">Strongest pattern</div>
              <div className="intel__headline-text">{consensus[0].label}</div>
              <div className="intel__headline-sub">
                {consensus[0].founders} of {nf} founders, independently
              </div>
            </div>
          )}

          <h2 className="intel__h">Top shared themes</h2>
          {consensus.slice(0, 4).map((t, i) => (
            <div className="intel__theme-head" key={i}>
              <span className="intel__label">
                {TIER_ICON[tierOf(t.founders, nf)]} {t.label}
              </span>
              <span className="intel__count">{t.founders}/{nf}</span>
            </div>
          ))}

          {absent.length > 0 && (
            <>
              <h2 className="intel__h">What nobody mentioned</h2>
              <div className="intel__types">
                {absent.map((a) => (
                  <span className="intel__chip intel__chip--absent" key={a.topic}>{a.topic}</span>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ---- Consensus ---- */}
      {sub === 'consensus' && (
        <div>
          {consensus.map((theme, i) => (
            <div className="intel__theme" key={i}>
              <div className="intel__theme-head">
                <span className="intel__label">
                  {TIER_ICON[tierOf(theme.founders, nf)]} {theme.label}
                </span>
                <span className="intel__count">{theme.founders}/{nf} founders</span>
              </div>
              <EvidenceList theme={theme} />
            </div>
          ))}
        </div>
      )}

      {/* ---- Timeline ---- */}
      {sub === 'timeline' &&
        (!timeline || timeline.periods.length === 0 ? (
          <p className="center-message">
            No publish dates yet — run the back-catalog backfill to build the timeline.
          </p>
        ) : (
          <div>
            <p className="intel__summary">
              Cumulative founders who had expressed each theme, by{' '}
              {timeline.granularity}: <strong>{timeline.periods.join(' → ')}</strong>.
              Founders: {timeline.founders_cumulative.join(' → ')}.
            </p>
            {Object.entries(timeline.series).map(([cat, ents]) => (
              <div key={cat}>
                <h2 className="intel__h">{cat}</h2>
                {Object.entries(ents)
                  .sort((a, b) => b[1][b[1].length - 1] - a[1][a[1].length - 1])
                  .map(([ent, vals]) => (
                    <div className="tl__row" key={ent}>
                      <span className="tl__label">{ent}</span>
                      <span className="tl__bars">
                        {vals.map((v, i) => (
                          <span className="tl__cell" key={i} title={`${timeline.periods[i]}: ${v}`}>
                            <span
                              className="tl__bar"
                              style={{ width: `${(v / nf) * 100}%` }}
                            />
                            <span className="tl__v">{v}</span>
                          </span>
                        ))}
                      </span>
                    </div>
                  ))}
              </div>
            ))}
          </div>
        ))}

      {/* ---- New This Period ---- */}
      {sub === 'new' &&
        (!timeline || timeline.trends.length === 0 ? (
          <p className="center-message">No accelerating themes yet (needs ≥2 periods).</p>
        ) : (
          <div>
            <p className="intel__summary">
              What gained founders in <strong>{timeline.periods[timeline.periods.length - 1]}</strong>:
            </p>
            {timeline.trends.map((t, i) => (
              <div className="intel__theme-head" key={i}>
                <span className="intel__label">
                  📈 {t.entity} <span className="tl__cat">({t.category})</span>
                </span>
                <span className="intel__count">
                  {t.from} → {t.to} (+{t.gained})
                </span>
              </div>
            ))}
          </div>
        ))}

      {/* ---- Editorial ---- */}
      {sub === 'editorial' &&
        (editorial && editorial.body ? (
          <div className="markdown">
            <ReactMarkdown>{editorial.body}</ReactMarkdown>
          </div>
        ) : (
          <p className="center-message">
            No editorial report generated yet for this brain.
          </p>
        ))}

      {/* ---- Raw Audit ---- */}
      {sub === 'audit' && (
        <div>
          <p className="intel__summary">
            Every count below is computed from {intel.total_observations} attributable
            observations. This is the evidence trail behind the narrative.
          </p>
          {Object.entries(intel.rollups).map(([cat, rows]) =>
            rows.length === 0 ? null : (
              <div key={cat}>
                <h2 className="intel__h">{cat} by adoption</h2>
                <div className="intel__bars">
                  {rows.map((r, i) => (
                    <div className="intel__bar-row" key={i}>
                      <span className="intel__bar-label">{r.value}</span>
                      <span
                        className="intel__bar"
                        style={{ width: `${(r.founders / nf) * 100}%` }}
                        title={r.evidence.map((e) => e.creator).join(', ')}
                      />
                      <span className="intel__bar-n">{r.founders}</span>
                    </div>
                  ))}
                </div>
              </div>
            )
          )}
          <h2 className="intel__h">Observation coverage</h2>
          <div className="intel__types">
            {Object.entries(intel.by_type).map(([t, c]) => (
              <span className="intel__chip" key={t}>
                {t} <strong>{c}</strong>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
