"""Entry point for Rz-Flow.

Usage:
    uv run rz-flow               # normal production run
    uv run rz-flow --dry-run     # evaluate but don't publish
    uv run rz-flow --init-db     # create DB tables and exit
"""

import argparse
import asyncio
import os
import sys

import structlog

from rz_flow.config import get_settings
from rz_flow.flow_config import load_flow_config
from rz_flow.logging_config import configure_logging
from rz_flow.pipeline import Pipeline
from rz_flow.storage import create_storage
from rz_flow.telegram import TelegramPublisher


async def _async_main(dry_run: bool = False, init_db_only: bool = False) -> int:
    settings = get_settings()
    flow_config = load_flow_config()
    log = structlog.get_logger("main")

    storage = create_storage(
        database_url=settings.turso_database_url,
        auth_token=settings.turso_auth_token,
    )

    try:
        await storage.init()
        if init_db_only:
            log.info("db_initialized")
            return 0

        pipeline = Pipeline(settings=settings, storage=storage, flow_config=flow_config)

        try:
            stats = await pipeline.run(dry_run=dry_run)
        except Exception as exc:
            log.exception("pipeline_fatal_error", error=str(exc))
            # Send alert to Telegram if this is a real run (not dry run)
            if not dry_run and settings.is_production:
                alert = TelegramPublisher(
                    bot_token=settings.telegram_bot_token,
                    channel_id=settings.telegram_channel_id,
                    admin_chat_id=settings.telegram_admin_chat_id,
                )
                await alert.send_alert(f"Pipeline crashed: {type(exc).__name__}: {exc}")
            _write_github_summary(
                f"❌ Pipeline crashed: {type(exc).__name__}: {exc}",
                dry_run=dry_run,
            )
            return 1

        # Send Telegram alert if quota was exhausted (informational, not a crash)
        if stats.quota_exhausted and not dry_run and settings.is_production:
            alert = TelegramPublisher(
                bot_token=settings.telegram_bot_token,
                channel_id=settings.telegram_channel_id,
                admin_chat_id=settings.telegram_admin_chat_id,
            )
            await alert.send_alert(
                "Gemini daily quota exhausted. "
                f"Processed {stats.posted + stats.skipped} articles before stopping. "
                "Quota resets at midnight UTC."
            )

        # Write summary to GitHub Actions step summary (no-op locally)
        quota_flag = " [QUOTA EXHAUSTED]" if stats.quota_exhausted else ""
        _write_github_summary(
            f"{'⚠️' if stats.quota_exhausted else '✅'}{quota_flag} "
            f"posted={stats.posted} skipped={stats.skipped} errors={stats.errors}",
            dry_run=dry_run,
        )

        # Non-zero exit if all new articles errored (triggers GitHub Alert)
        if stats.new_articles > 0 and stats.errors == stats.new_articles:
            return 1
        return 0

    finally:
        await storage.close()


def _write_github_summary(message: str, dry_run: bool = False) -> None:
    """Append a line to GITHUB_STEP_SUMMARY if running in GitHub Actions."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    mode = " (dry run)" if dry_run else ""
    try:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(f"**Rz-Flow{mode}**: {message}\n")
    except OSError:
        pass


def main() -> None:
    """CLI entry point registered in pyproject.toml [project.scripts]."""
    parser = argparse.ArgumentParser(description="Rz-Flow Telegram channel bot")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate articles but do NOT publish to Telegram",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create DB tables and exit",
    )
    args = parser.parse_args()

    # Set up logging before anything else
    settings = get_settings()
    configure_logging(settings.app_env)

    exit_code = asyncio.run(_async_main(dry_run=args.dry_run, init_db_only=args.init_db))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
