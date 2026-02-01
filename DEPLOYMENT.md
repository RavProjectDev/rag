# Deployment Guide

## Google Cloud Platform (GCP) Deployment

The application is configured to deploy to GCP Cloud Run using GitHub Actions. There are two environments:

- **Production**: Deploys from `main` branch
- **Staging**: Deploys from `stg` branch

### Prerequisites

Before deploying, you need to configure the following in GCP Secret Manager:

#### Required Secrets

All secrets must be created in GCP Secret Manager before deployment:

1. **`OPENAI_API_KEY`** - OpenAI API key for LLM completions
2. **`MONGODB_URI`** - MongoDB connection string
3. **`MONGODB_DB_NAME`** - MongoDB database name
4. **`PINECONE_API_KEY`** - Pinecone API key for vector storage
5. **`GEMINI_API_KEY`** - Google Gemini API key for embeddings
6. **`GOOGLE_CLOUD_PROJECT_ID`** - GCP project ID
7. **`VERTEX_REGION`** - Vertex AI region (e.g., `northamerica-northeast1`)
8. **`UPSTASH_REDIS_REST_URL`** - Upstash Redis REST URL for rate limiting
9. **`UPSTASH_REDIS_REST_TOKEN`** - Upstash Redis REST token
10. **`SUPABASE_URL`** - Supabase project URL for JWT authentication (e.g., `https://your-project.supabase.co`)

#### Creating Secrets in GCP

Use the `gcloud` CLI to create secrets:

```bash
# Example: Create SUPABASE_URL secret
echo -n "https://your-project.supabase.co" | gcloud secrets create SUPABASE_URL \
  --data-file=- \
  --project=YOUR_PROJECT_ID

# Grant the Cloud Run service account access to the secret
gcloud secrets add-iam-policy-binding SUPABASE_URL \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=YOUR_PROJECT_ID
```

Repeat for all required secrets listed above.

### GitHub Secrets

Configure the following secrets in your GitHub repository settings (`Settings` > `Secrets and variables` > `Actions`):

1. **`GCP_PROJECT_ID`** - Your GCP project ID
2. **`GCP_REGION`** - GCP region for Cloud Run (e.g., `us-central1`)
3. **`GCP_SERVICE_ACCOUNT`** - Service account email for Cloud Run
4. **`GCP_SA_KEY`** - GCP service account key (JSON)
5. **`DOCKERHUB_USERNAME`** - Docker Hub username (for production)
6. **`DOCKERHUB_TOKEN`** - Docker Hub access token (for production)

### Deployment Configuration

#### Environment Variables

The following environment variables are set directly in the deployment (not as secrets):

- **`ENVIRONMENT`** - `PRD` or `STG`
- **`DATABASE_CONFIGURATION`** - `pinecone` (vector database)
- **`PINECONE_INDEX_NAME`** - `gemini` (Pinecone index name)
- **`CHUNKING_STRATEGY`** - `fixed_size` (chunking strategy)
- **`AUTH_MODE`** - `prd` (requires JWT authentication)

### Authentication

The application uses **Supabase** for JWT authentication in production mode (`AUTH_MODE=prd`).

#### How It Works:

1. Client obtains a JWT token from Supabase authentication
2. Client includes the token in the `Authorization: Bearer <token>` header
3. API validates the token using Supabase's JWKS (JSON Web Key Set)
4. User ID is extracted from the token's `sub` claim

#### Development Mode:

For local development or testing without authentication, set `AUTH_MODE=dev` in your `.env` file. This will skip JWT validation and use a dummy user ID.

### Deployment Workflow

#### Production Deployment

1. Push to `main` branch or trigger workflow manually
2. GitHub Actions will:
   - Build Docker image
   - Push to Docker Hub
   - Deploy to Cloud Run (service: `rag-api-prd`)

#### Staging Deployment

1. Push to `stg` branch or trigger workflow manually
2. GitHub Actions will:
   - Build Docker image
   - Push to GCR (Google Container Registry)
   - Deploy to Cloud Run (service: `rag-api-stg`)

### Post-Deployment Verification

After deployment, verify the service is running:

```bash
# Get the service URL
gcloud run services describe rag-api-prd \
  --region=YOUR_REGION \
  --format='value(status.url)'

# Test health endpoint
curl https://YOUR_SERVICE_URL/api/v1/health

# Expected response:
# {"status":"ok","version":"1.0.0","environment":"PRD"}
```

### Rate Limiting

The application includes user-based rate limiting using Redis (Upstash):

- **Global endpoint limit**: Configurable via deployment
- **Per-user monthly limit**: Resets on the 1st of each month at midnight ET
- **Rate limit headers**: Included in all responses
  - `X-RateLimit-Limit`: Maximum requests per month
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp (ISO 8601)

### Troubleshooting

#### Common Issues:

1. **"Authentication not configured" error**
   - Ensure `SUPABASE_URL` is added to GCP Secret Manager
   - Verify the secret is accessible by the Cloud Run service account
   - Check that `AUTH_MODE=prd` is set in deployment

2. **Rate limiting not working**
   - Verify `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` are configured
   - Check logs for Redis connection errors

3. **Deployment fails**
   - Verify all required secrets exist in GCP Secret Manager
   - Check service account has necessary permissions
   - Review GitHub Actions logs for specific errors

### Service Account Permissions

The GCP service account needs the following roles:

- `roles/run.admin` - Deploy Cloud Run services
- `roles/iam.serviceAccountUser` - Act as service account
- `roles/secretmanager.secretAccessor` - Access Secret Manager secrets

### Monitoring

View logs and metrics:

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api-prd" \
  --limit=50 \
  --format=json

# Monitor with Cloud Console
# https://console.cloud.google.com/run
```

### Rollback

If you need to rollback to a previous version:

```bash
# List revisions
gcloud run revisions list --service=rag-api-prd --region=YOUR_REGION

# Rollback to specific revision
gcloud run services update-traffic rag-api-prd \
  --to-revisions=REVISION_NAME=100 \
  --region=YOUR_REGION
```
