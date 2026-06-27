"""Click CLI entry point for YouTube Brain."""

import asyncio
import logging

import click

from youtube_brain.config.settings import get_settings


def run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose (DEBUG) logging.")
def main(verbose: bool) -> None:
    """YouTube Brain -- turn YouTube channels into searchable advisors."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@main.command()
@click.argument("url")
@click.option("--name", "-n", default=None, help="Custom brain name.")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=None,
    help="Max videos to ingest, most recent first (bounds cost on large channels).",
)
def ingest(url: str, name: str | None, limit: int | None) -> None:
    """Ingest a YouTube URL (video, playlist, or channel) into a new brain."""
    from youtube_brain.ingest.pipeline import ingest_url
    from youtube_brain.storage.database import init_database

    async def _run():
        await init_database()
        return await ingest_url(url, brain_name=name, limit=limit)

    result = run_async(_run())

    click.echo(f"Brain ID:          {result.brain_id}")
    click.echo(f"Videos found:      {result.videos_found}")
    click.echo(f"Videos processed:  {result.videos_processed}")
    click.echo(f"Chunks created:    {result.chunks_created}")
    if result.errors:
        click.echo(f"Errors ({len(result.errors)}):")
        for err in result.errors:
            click.echo(f"  - {err}")
    else:
        click.echo("Errors:            0")


@main.command()
@click.argument("brain_id")
@click.option("--max-videos", "-n", type=int, default=None,
              help="Max new videos to ingest this refresh (bounds cost).")
@click.option("--no-ingest", is_flag=True,
              help="Skip new-video ingest; only (re)extract observations + recluster.")
def refresh(brain_id: str, max_videos: int | None, no_ingest: bool) -> None:
    """Watchlist refresh: ingest new videos, extract observations, show what changed."""
    from youtube_brain.observations.refresh import refresh_brain
    from youtube_brain.storage.database import init_database

    async def _run():
        await init_database()
        return await refresh_brain(brain_id, max_videos=max_videos, ingest_new=not no_ingest)

    out = run_async(_run())
    cl = out["changelog"]
    click.echo(f"New videos ingested:   {out['new_videos']}")
    click.echo(f"New observations:      {out['new_observations']}")
    click.echo(f"Founders added:        {cl['new_founders']}")
    if cl["rollup_changes"]:
        click.echo("\nWhat changed:")
        for c in cl["rollup_changes"]:
            tag = " (NEW)" if c["is_new"] else ""
            who = ", ".join(c["new_creators"])
            click.echo(f"  {c['category']} / {c['value']}: {c['before']}->{c['after']}{tag}  +{who}")
    else:
        click.echo("\nNo changes since last refresh.")


@main.command()
@click.argument("brain_id")
@click.argument("query")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["qa", "article", "playbook", "summary", "faq"]),
    default="qa",
    help="Output mode.",
)
def ask(brain_id: str, query: str, mode: str) -> None:
    """Ask a question against a brain."""
    from youtube_brain.generation.answer import generate_answer
    from youtube_brain.llm.gemini import GeminiClient
    from youtube_brain.storage.brains import get_brain
    from youtube_brain.storage.database import init_database

    async def _run():
        await init_database()
        brain = await get_brain(brain_id)
        if brain is None:
            click.echo(f"Error: Brain '{brain_id}' not found.", err=True)
            raise SystemExit(1)

        client = GeminiClient()
        try:
            result = await generate_answer(
                query=query,
                brain_id=str(brain.id),
                brain_name=brain.name,
                client=client,
                mode=mode,
                recency_weight=brain.recency_weight,
            )
        finally:
            await client.close()
        return result

    result = run_async(_run())

    click.echo(result.answer)
    click.echo()
    click.echo(f"Confidence: {result.confidence.get('level', 'unknown')}")
    click.echo(f"Chunks searched: {result.chunks_searched}  |  Chunks used: {result.chunks_used}")

    if result.citations:
        click.echo()
        click.echo("Sources:")
        for i, cite in enumerate(result.citations[:5], 1):
            click.echo(f"  [{i}] {cite.video_title} @ {cite.timestamp_display}")
            click.echo(f"      {cite.video_url}")


@main.command()
@click.option("--host", "-h", default=None, help="Host to bind to.")
@click.option("--port", "-p", default=None, type=int, help="Port to listen on.")
def serve(host: str | None, port: int | None) -> None:
    """Start the API server."""
    import uvicorn

    from youtube_brain.storage.database import init_database

    settings = get_settings()
    host = host or settings.api_host
    port = port or settings.api_port

    run_async(init_database())

    uvicorn.run(
        "youtube_brain.api.app:create_app",
        host=host,
        port=port,
        factory=True,
        reload=True,
    )


@main.command(name="list")
def list_brains_cmd() -> None:
    """List all brains."""
    from youtube_brain.storage.brains import list_brains
    from youtube_brain.storage.database import init_database

    async def _run():
        await init_database()
        return await list_brains()

    brains = run_async(_run())

    if not brains:
        click.echo("No brains found.")
        return

    for brain in brains:
        status_tag = f"[{brain.status.value}]"
        click.echo(f"  {status_tag:18s} {brain.name} ({brain.video_count} videos) -- {brain.id}")


@main.group()
def watch() -> None:
    """Manage automatic watchlist refresh schedules."""


@watch.command("enable")
@click.argument("brain_id")
@click.option("--interval-hours", default=24, help="Hours between refreshes.")
@click.option("--max-videos", default=2, help="Max new videos per refresh (bounds cost).")
def watch_enable(brain_id: str, interval_hours: int, max_videos: int) -> None:
    """Enable scheduled refresh for a brain."""
    from youtube_brain.storage.database import init_database
    from youtube_brain.storage.schedules import upsert_schedule

    async def _run():
        await init_database()
        await upsert_schedule(brain_id, enabled=True,
                              interval_hours=interval_hours, max_videos=max_videos)

    run_async(_run())
    click.echo(f"Watching {brain_id}: every {interval_hours}h, up to {max_videos} new videos.")


@watch.command("disable")
@click.argument("brain_id")
def watch_disable(brain_id: str) -> None:
    """Disable scheduled refresh for a brain."""
    from youtube_brain.storage.database import init_database
    from youtube_brain.storage.schedules import get_schedule, upsert_schedule

    async def _run():
        await init_database()
        sched = await get_schedule(brain_id)
        hrs = sched["interval_hours"] if sched else 24
        mv = sched["max_videos"] if sched else 2
        await upsert_schedule(brain_id, enabled=False, interval_hours=hrs, max_videos=mv)

    run_async(_run())
    click.echo(f"Stopped watching {brain_id}.")


@watch.command("list")
def watch_list() -> None:
    """List watchlist schedules and whether each is due."""
    from datetime import datetime, timezone

    from youtube_brain.observations.scheduler import is_due
    from youtube_brain.storage.database import init_database
    from youtube_brain.storage.schedules import list_schedules

    async def _run():
        await init_database()
        return await list_schedules()

    scheds = run_async(_run())
    if not scheds:
        click.echo("No watchlist schedules. Use 'ytbrain watch enable <brain_id>'.")
        return
    now = datetime.now(timezone.utc)
    for s in scheds:
        state = "off" if not s["enabled"] else ("DUE" if is_due(s, now) else "waiting")
        last = s["last_refreshed_at"].strftime("%Y-%m-%d %H:%M") if s["last_refreshed_at"] else "never"
        click.echo(f"  [{state:7s}] {s['brain_id']}  every {s['interval_hours']}h  "
                   f"max {s['max_videos']}  last: {last}")


@watch.command("run")
def watch_run() -> None:
    """Refresh all brains whose schedule is due (wire this into cron / Task Scheduler)."""
    from youtube_brain.observations.scheduler import run_due
    from youtube_brain.storage.database import init_database

    async def _run():
        await init_database()
        return await run_due()

    results = run_async(_run())
    if not results:
        click.echo("Nothing due.")
        return
    for r in results:
        if "error" in r:
            click.echo(f"  {r['brain_id']}: ERROR {r['error']}")
        else:
            cl = r["changelog"]
            click.echo(f"  {r['brain_id']}: +{r['new_videos']} videos, "
                       f"+{r['new_observations']} observations, "
                       f"+{cl['new_founders']} founders, {len(cl['rollup_changes'])} changes")


@watch.command("loop")
@click.option("--poll-minutes", default=60, help="Minutes between due-checks.")
def watch_loop(poll_minutes: int) -> None:
    """Run a foreground daemon that refreshes due brains every poll interval."""
    import time

    from youtube_brain.observations.scheduler import run_due
    from youtube_brain.storage.database import init_database

    run_async(init_database())
    click.echo(f"Watchlist daemon started (poll every {poll_minutes}m). Ctrl-C to stop.")
    while True:
        results = run_async(run_due())
        if results:
            click.echo(f"Refreshed {len(results)} brain(s).")
        time.sleep(poll_minutes * 60)


if __name__ == "__main__":
    main()
