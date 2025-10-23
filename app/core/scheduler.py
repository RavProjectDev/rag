from apscheduler.schedulers.asyncio import AsyncIOScheduler

from rag.app.db.connections import EmbeddingConnection
from rag.app.schemas.data import EmbeddingConfiguration
from rag.app.services.sync_service.sync_db import run

scheduler = AsyncIOScheduler()


def start_scheduler(
    connection: EmbeddingConnection,
    embedding_configuration: EmbeddingConfiguration,
):
    scheduler.add_job(
        run,
        "interval",
        hours=24 * 7,
        args=[connection, embedding_configuration],
    )
    scheduler.start()
