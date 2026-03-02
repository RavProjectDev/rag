from functools import lru_cache

from rag.app.core.config import SharedSettings


class WebhookSettings(SharedSettings):
    """
    Settings for the Sync Webhook API.

    Only requires DB + embedding config from SharedSettings.
    Does NOT require OpenAI, Supabase, Redis, or rate-limit vars.

    Start with:
        uvicorn rag.app.webhook:app
    """

    # HMAC secret used to verify incoming Sanity webhook signatures.
    # Set this to the secret configured in the Sanity webhook dashboard.
    webhook_secret: str | None = None


@lru_cache()
def get_webhook_settings() -> WebhookSettings:
    return WebhookSettings()
