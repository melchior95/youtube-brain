import { useState } from 'react';
import { ingestUrl } from '../api';

export default function PullCard({ handle, url }: { handle: string; url: string }) {
  const [state, setState] = useState<'idle' | 'pulling' | 'done' | 'error'>('idle');
  async function pull() {
    setState('pulling');
    try {
      await ingestUrl(url);
      setState('done');
    } catch {
      setState('error');
    }
  }
  return (
    <div className="creator-card pending">
      <h3>{handle}</h3>
      <p className="muted">not pulled yet</p>
      {state === 'idle' && (
        <button className="btn btn--primary btn--small" onClick={pull} type="button">
          Pull
        </button>
      )}
      {state === 'pulling' && <span className="muted">ingesting…</span>}
      {state === 'done' && <span className="muted">queued — refresh shortly</span>}
      {state === 'error' && (
        <button className="btn btn--secondary btn--small" onClick={pull} type="button">
          retry
        </button>
      )}
    </div>
  );
}
