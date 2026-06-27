"""Click CLI entry point for YouTube Brain.

The CLI is intentionally thin: it lists what's been ingested. All ingestion,
retrieval, and intelligence run through the zero-API skill bridge
(`scripts/skill_bridge.py`), driven by Claude in-loop.
"""

import asyncio
import logging

import click


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


if __name__ == "__main__":
    main()
