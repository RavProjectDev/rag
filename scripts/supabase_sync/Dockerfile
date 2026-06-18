FROM python:3.13-slim

WORKDIR /workspace

COPY requirements-rate-limit-sync.txt .
RUN pip install --no-cache-dir -r requirements-rate-limit-sync.txt

# Copy only what the sync job needs.
COPY scripts/supabase_sync /workspace/rag/scripts/supabase_sync

CMD ["python", "/workspace/rag/scripts/supabase_sync/rate_limit_batch_sync.py"]
