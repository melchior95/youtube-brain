"""Chunk labeling via Gemini using a controlled taxonomy."""

from __future__ import annotations

import logging

from youtube_brain.llm.gemini import GeminiClient

logger = logging.getLogger(__name__)

LABEL_SYSTEM = """\
You are a transcript labeling assistant. Given transcript chunks from a YouTube video, \
label each chunk using ONLY the controlled taxonomy values below.

business_type (pick one): saas, ecommerce, agency, marketplace, content, physical_product, \
service, mobile_app, other

advice_category (pick one): marketing, distribution, pricing, hiring, fundraising, product, \
operations, customer_acquisition, retention, monetization, launch, growth, technical, legal, other

stage (pick one): idea, pre_launch, early_stage, growth, scaling, mature, exit, other

asset_type (pick one): interview, tutorial, review, commentary, case_study, earnings_call, \
lecture, panel, other

topics: 2-5 free-form keywords describing the chunk content.

Return a JSON array with one object per chunk, each containing: \
business_type, advice_category, stage, asset_type, topics. \
Use the exact chunk indices provided. Do not add commentary outside the JSON.\
"""


async def label_chunks(
    client: GeminiClient,
    chunks: list[dict],
    video_title: str,
    channel_name: str,
    batch_size: int | None = None,
) -> list[dict]:
    """Label transcript chunks with taxonomy metadata via Gemini.

    Sends ``batch_size`` chunks per Gemini call as a single request (one
    longer prompt, not multiple requests), keeping request count low to
    respect tight free-tier rate limits. On failure for a batch, returns
    empty dicts for those chunks.

    Args:
        client: An initialized GeminiClient.
        chunks: List of chunk dicts with 'text', 'start_time', 'end_time'.
        video_title: Title of the video for context.
        channel_name: Channel name for context.
        batch_size: Chunks per request; defaults to settings.label_batch_size.

    Returns:
        List of label dicts, one per chunk. Empty dict on failure.
    """
    if batch_size is None:
        from youtube_brain.config.settings import get_settings

        batch_size = get_settings().label_batch_size
    all_labels: list[dict] = []

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        # Format chunks for the prompt
        chunk_texts = []
        for i, chunk in enumerate(batch):
            idx = batch_start + i
            start = chunk.get("start_time", 0)
            end = chunk.get("end_time", 0)
            text = chunk.get("text", "")
            chunk_texts.append(f"[Chunk {idx}] ({start:.1f}s - {end:.1f}s):\n{text}")

        prompt = (
            f"Video: {video_title}\n"
            f"Channel: {channel_name}\n\n"
            f"Label the following {len(batch)} chunks:\n\n"
            + "\n\n".join(chunk_texts)
        )

        try:
            result = await client.generate_json(prompt, system=LABEL_SYSTEM)
            if isinstance(result, list) and len(result) == len(batch):
                all_labels.extend(result)
            else:
                logger.warning(
                    "Label batch at index %d returned unexpected structure, using empty labels",
                    batch_start,
                )
                all_labels.extend({} for _ in batch)
        except Exception:
            logger.warning(
                "Label batch at index %d failed, using empty labels",
                batch_start,
                exc_info=True,
            )
            all_labels.extend({} for _ in batch)

    return all_labels
