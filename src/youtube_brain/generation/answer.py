"""Answer generation with five output modes, citation formatting, and confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

from youtube_brain.generation.prompts import PROMPTS
from youtube_brain.llm.gemini import GeminiClient
from youtube_brain.retrieval.search import RetrievalResult, SearchResult, retrieve


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A citation linking a claim back to a specific video moment."""

    video_title: str
    video_url: str
    timestamp: float
    timestamp_display: str
    transcript_text: str
    caption_kind: str
    chunk_id: str


@dataclass
class AnswerResult:
    """Complete answer with citations and confidence metadata."""

    answer: str
    citations: list[Citation]
    confidence: dict
    chunks_searched: int
    chunks_used: int
    mode: str = "qa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``M:SS`` or ``H:MM:SS``.

    Examples:
        >>> _format_timestamp(0)
        '0:00'
        >>> _format_timestamp(65)
        '1:05'
        >>> _format_timestamp(3661)
        '1:01:01'
    """
    total = int(seconds)
    hrs, remainder = divmod(total, 3600)
    mins, secs = divmod(remainder, 60)
    if hrs > 0:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


def _build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a prompt-ready context block."""
    blocks: list[str] = []
    for r in results:
        start_disp = _format_timestamp(r.start_time)
        end_disp = _format_timestamp(r.end_time)
        topics_str = ", ".join(r.topics) if r.topics else "n/a"
        header = (
            f"[Video: {r.video_title} | Channel: {r.channel_name}]\n"
            f"[Timestamp: {start_disp} – {end_disp} | Caption: {r.caption_kind}]\n"
            f"[Topics: {topics_str}]"
        )
        blocks.append(f"{header}\n{r.text}\n---")
    return "\n\n".join(blocks)


def _build_citations(results: list[SearchResult]) -> list[Citation]:
    """Map each :class:`SearchResult` to a :class:`Citation`."""
    citations: list[Citation] = []
    for r in results:
        t_param = int(r.start_time)
        url = f"https://youtu.be/{r.youtube_id}?t={t_param}"
        transcript_text = r.text[:200] if len(r.text) > 200 else r.text
        citations.append(
            Citation(
                video_title=r.video_title,
                video_url=url,
                timestamp=r.start_time,
                timestamp_display=_format_timestamp(r.start_time),
                transcript_text=transcript_text,
                caption_kind=r.caption_kind,
                chunk_id=r.chunk_id,
            )
        )
    return citations


def _compute_confidence(results: list[SearchResult]) -> dict:
    """Score confidence based on supporting evidence breadth and caption quality.

    Returns a dict with keys ``level``, ``supporting_chunks``,
    ``supporting_videos``, and ``caption_quality``.
    """
    chunk_count = len(results)
    video_ids = {r.video_id for r in results}
    video_count = len(video_ids)

    # Determine level
    if chunk_count >= 10 and video_count >= 5:
        level = "high"
    elif chunk_count >= 5 and video_count >= 2:
        level = "medium"
    else:
        level = "low"

    # Caption quality
    manual_count = sum(
        1 for r in results if r.caption_kind and r.caption_kind.lower() == "manual"
    )
    if chunk_count > 0 and manual_count / chunk_count > 0.5:
        caption_quality = "mostly_manual"
    else:
        caption_quality = "mostly_auto"

    return {
        "level": level,
        "supporting_chunks": chunk_count,
        "supporting_videos": video_count,
        "caption_quality": caption_quality,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def generate_answer(
    query: str,
    brain_id: str,
    brain_name: str,
    client: GeminiClient,
    mode: str = "qa",
    recency_weight: float = 0.1,
) -> AnswerResult:
    """Generate an answer by retrieving evidence and prompting Gemini.

    Parameters
    ----------
    query:
        The user's natural-language question.
    brain_id:
        UUID of the brain to search.
    brain_name:
        Human-readable brain name (inserted into the system prompt).
    client:
        A :class:`GeminiClient` instance.
    mode:
        One of ``"qa"``, ``"article"``, ``"playbook"``, ``"summary"``,
        ``"faq"``.  Controls which system prompt template is used.
    recency_weight:
        Weight given to recency in the retrieval ranking (0..1).

    Returns
    -------
    AnswerResult
        The generated answer with citations, confidence, and chunk counts.
    """
    # 1. Retrieve relevant chunks
    retrieval: RetrievalResult = await retrieve(
        query, brain_id, client, recency_weight=recency_weight
    )
    results = retrieval.results

    # 2. Handle no results
    if not results:
        return AnswerResult(
            answer=(
                "I couldn't find enough evidence in the knowledge base to "
                "answer this question. Try rephrasing or adding more videos."
            ),
            citations=[],
            confidence={"level": "low", "supporting_chunks": 0, "supporting_videos": 0, "caption_quality": "n/a"},
            chunks_searched=retrieval.chunks_searched,
            chunks_used=0,
            mode=mode,
        )

    # 3. Build context from results
    context = _build_context(results)

    # 4. Select system prompt
    system_template = PROMPTS.get(mode, PROMPTS["qa"])
    system_prompt = system_template.format(brain_name=brain_name)

    # 5. Generate answer via Gemini
    user_prompt = f"Question: {query}\n\nEvidence:\n{context}"
    answer_text = await client.generate(
        user_prompt, system=system_prompt, temperature=0.4
    )

    # 6. Build response
    citations = _build_citations(results)
    confidence = _compute_confidence(results)

    return AnswerResult(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        chunks_searched=retrieval.chunks_searched,
        chunks_used=len(results),
        mode=mode,
    )
