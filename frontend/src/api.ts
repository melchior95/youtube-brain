const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Brain {
  id: string;
  name: string;
  status: string;
  video_count: number;
  visibility: string;
  created_at: string;
}

export interface Video {
  id: string;
  video_id: string;
  title: string;
  channel_name: string;
  url: string;
  status: string;
  duration_seconds: number | null;
  video_summary: string | null;
  created_at: string;
}

export interface BrainDetail extends Brain {
  videos: Video[];
  multi_creator?: boolean;
}

export interface Citation {
  video_title: string;
  video_url: string;
  timestamp: number;
  timestamp_display: string;
  transcript_text: string;
  caption_kind: string;
  chunk_id: string;
}

export interface Confidence {
  level: string;
  supporting_chunks: number;
  supporting_videos: number;
  caption_quality: string;
}

export interface AnswerResponse {
  answer: string;
  citations: Citation[];
  confidence: Confidence;
  chunks_searched: number;
  chunks_used: number;
  mode: string;
}

export interface Evidence {
  creator: string;
  quote: string;
  youtube_id: string | null;
  start_time: number | null;
  obs_type?: string;
}

export interface ConsensusTheme {
  label: string;
  founders: number;
  total_founders: number;
  evidence: Evidence[];
}

export interface RollupRow {
  value: string;
  founders: number;
  evidence: Evidence[];
}

export interface Absence {
  topic: string;
  founders: number;
}

export interface Intelligence {
  brain_name: string;
  total_observations: number;
  founders: number;
  consensus: ConsensusTheme[];
  rollups: Record<string, RollupRow[]>;
  absences: Absence[];
  by_type: Record<string, number>;
}

export interface TimelineTrend {
  category: string;
  entity: string;
  from: number;
  to: number;
  gained: number;
  period: string;
}

export interface Timeline {
  granularity: string;
  periods: string[];
  series: Record<string, Record<string, number[]>>;
  volume: number[];
  founders_cumulative: number[];
  trends: TimelineTrend[];
}

export interface Editorial {
  title: string | null;
  body: string | null;
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function listBrains(): Promise<Brain[]> {
  const res = await fetch(`${BASE}/api/brains`);
  if (!res.ok) throw new Error(`Failed to list brains: ${res.statusText}`);
  return res.json();
}

export async function getBrain(id: string): Promise<BrainDetail> {
  const res = await fetch(`${BASE}/api/brains/${id}`);
  if (!res.ok) throw new Error(`Failed to get brain: ${res.statusText}`);
  return res.json();
}

export async function ingestUrl(
  url: string,
  name?: string
): Promise<{ status: string; url: string }> {
  const res = await fetch(`${BASE}/api/brains/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url, name: name || undefined }),
  });
  if (!res.ok) throw new Error(`Failed to ingest: ${res.statusText}`);
  return res.json();
}

export async function getIntelligence(brainId: string): Promise<Intelligence> {
  const res = await fetch(`${BASE}/api/brains/${brainId}/intelligence`);
  if (!res.ok) throw new Error(`Failed to load intelligence: ${res.statusText}`);
  return res.json();
}

export async function getTimeline(
  brainId: string,
  granularity: string = 'month'
): Promise<Timeline> {
  const res = await fetch(`${BASE}/api/brains/${brainId}/timeline?granularity=${granularity}`);
  if (!res.ok) throw new Error(`Failed to load timeline: ${res.statusText}`);
  return res.json();
}

export async function getEditorial(brainId: string): Promise<Editorial> {
  const res = await fetch(`${BASE}/api/brains/${brainId}/editorial`);
  if (!res.ok) throw new Error(`Failed to load editorial: ${res.statusText}`);
  return res.json();
}

export function ytLink(youtubeId: string | null, start: number | null): string {
  if (!youtubeId) return '';
  const t = start != null ? `?t=${Math.floor(start)}` : '';
  return `https://youtu.be/${youtubeId}${t}`;
}

// YouTube's canonical thumbnail — always available, no storage/yt-dlp needed.
export function ytThumb(youtubeId: string): string {
  return `https://i.ytimg.com/vi/${youtubeId}/hqdefault.jpg`;
}

export function fmtTime(seconds: number | null): string {
  if (seconds == null) return '';
  const s = Math.floor(seconds);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  const mm = m % 60;
  const ss = s % 60;
  return h > 0
    ? `${h}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
    : `${m}:${String(ss).padStart(2, '0')}`;
}

export async function askBrain(
  brainId: string,
  query: string,
  mode: string = 'qa'
): Promise<AnswerResponse> {
  const res = await fetch(`${BASE}/api/brains/${brainId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, mode }),
  });
  if (!res.ok) throw new Error(`Failed to ask: ${res.statusText}`);
  return res.json();
}

export interface CategorySummary {
  slug: string; name: string; description: string;
  creator_count: number; pulled_count: number;
}

export interface CategoryCreator {
  handle: string; pulled: boolean;
  brain_id?: string; name?: string; status?: string; video_count?: number;
  latest_video?: string | null;
  url?: string; channel_id?: string | null;
}

export interface CategoryDetail {
  slug: string; name: string; description: string; creators: CategoryCreator[];
}

export async function listCategories(): Promise<CategorySummary[]> {
  const res = await fetch(`${BASE}/api/categories`);
  if (!res.ok) throw new Error(`Failed to list categories: ${res.statusText}`);
  return res.json();
}

export async function getCategory(slug: string): Promise<CategoryDetail> {
  const res = await fetch(`${BASE}/api/categories/${slug}`);
  if (!res.ok) throw new Error(`Failed to get category: ${res.statusText}`);
  return res.json();
}

export async function getCategoryConsensus(slug: string): Promise<Intelligence> {
  const res = await fetch(`${BASE}/api/categories/${slug}/consensus`);
  if (!res.ok) throw new Error(`Failed to load consensus: ${res.statusText}`);
  return res.json();
}
