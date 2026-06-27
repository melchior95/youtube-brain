import { useEffect, useState } from 'react';
import { listCategories, type CategorySummary } from '../api';

export default function CategoryGrid({ onSelect }: { onSelect: (slug: string) => void }) {
  const [cats, setCats] = useState<CategorySummary[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    listCategories().then(setCats).catch((e) => setErr(String(e)));
  }, []);
  if (err) return <p className="form-error">{err}</p>;
  return (
    <div className="category-grid">
      <h1>YouTube Brain</h1>
      {cats.map((c) => (
        <button
          key={c.slug}
          className="category-card"
          onClick={() => onSelect(c.slug)}
          type="button"
        >
          <h2>{c.name}</h2>
          <p>{c.description}</p>
          <span className="muted">
            {c.pulled_count}/{c.creator_count} creators pulled
          </span>
        </button>
      ))}
    </div>
  );
}
