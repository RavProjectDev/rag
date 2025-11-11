# Google Cloud Run Deployment Guide

This guide covers deploying the RAG API to Google Cloud Run using GitHub Actions CI/CD.

## 🏗️ Architecture Overview

- **Container Registry**: Google Artifact Registry
- **Compute**: Cloud Run (serverless containers)
- **Secrets Management**: Google Secret Manager
- **CI/CD**: GitHub Actions
- **Authentication**: Workload Identity Federation (recommended) or Service Account Keys

## 📋 Prerequisites

### 1. Google Cloud Project Setup

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"

# Enable required APIs
gcloud services enable run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  --project=$PROJECT_ID
```

### 2. Create Artifact Registry Repository

```bash
# Create repository for Docker images
gcloud artifacts repositories create rag-api \
  --repository-format=docker \
  --location=$REGION \
  --description="RAG API Docker images" \
  --project=$PROJECT_ID
```

### 3. Set Up Google Secret Manager

Store your environment variables as secrets:

```bash
# OpenAI
echo -n "your-openai-api-key" | \
  gcloud secrets create OPENAI_API_KEY --data-file=- --project=$PROJECT_ID

# MongoDB
echo -n "your-mongodb-uri" | \
  gcloud secrets create MONGODB_URI --data-file=- --project=$PROJECT_ID

echo -n "rag_database" | \
  gcloud secrets create MONGODB_DB_NAME --data-file=- --project=$PROJECT_ID

echo -n "embeddings" | \
  gcloud secrets create MONGODB_VECTOR_COLLECTION --data-file=- --project=$PROJECT_ID

# Google Cloud
echo -n "your-gemini-api-key" | \
  gcloud secrets create GEMINI_API_KEY --data-file=- --project=$PROJECT_ID

echo -n "$PROJECT_ID" | \
  gcloud secrets create GOOGLE_CLOUD_PROJECT_ID --data-file=- --project=$PROJECT_ID

echo -n "us-central1" | \
  gcloud secrets create VERTEX_REGION --data-file=- --project=$PROJECT_ID
```

### 4. Configure GitHub Actions Authentication

#### Option A: Workload Identity Federation (Recommended - More Secure)

```bash
# Create service account for GitHub Actions
gcloud iam service-accounts create github-actions-deploy \
  --display-name="GitHub Actions Deploy" \
  --project=$PROJECT_ID

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Set up Workload Identity Federation
gcloud iam workload-identity-pools create github-pool \
  --location="global" \
  --display-name="GitHub Actions Pool" \
  --project=$PROJECT_ID

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --project=$PROJECT_ID

# Get the Workload Identity Provider name
gcloud iam workload-identity-pools providers describe github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --format="value(name)" \
  --project=$PROJECT_ID

# Bind GitHub repo to service account (replace YOUR_ORG/YOUR_REPO)
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_ORG/YOUR_REPO" \
  --project=$PROJECT_ID
```

#### Option B: Service Account Key (Simpler, Less Secure)

```bash
# Create service account
gcloud iam service-accounts create github-actions-deploy \
  --display-name="GitHub Actions Deploy" \
  --project=$PROJECT_ID

# Grant permissions (same as above)
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Create and download key
gcloud iam service-accounts keys create github-sa-key.json \
  --iam-account=github-actions-deploy@${PROJECT_ID}.iam.gserviceaccount.com \
  --project=$PROJECT_ID

# Copy the contents of github-sa-key.json to GitHub Secret: GCP_SA_KEY
cat github-sa-key.json
```

### 5. Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

#### For Workload Identity Federation:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_REGION`: Your deployment region (e.g., `us-central1`)
- `GCP_WORKLOAD_IDENTITY_PROVIDER`: The full provider name from step 4
- `GCP_SERVICE_ACCOUNT`: `github-actions-deploy@PROJECT_ID.iam.gserviceaccount.com`

#### For Service Account Key:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_REGION`: Your deployment region (e.g., `us-central1`)
- `GCP_SA_KEY`: Contents of the service account JSON key file

## 🚀 Deployment Workflows

### Staging Deployment
- **Trigger**: Push to `stg` branch
- **Service**: `rag-api-stg`
- **Resources**: 2GB RAM, 1 CPU
- **Scaling**: 0-10 instances
- **Environment**: `STG`

### Production Deployment
- **Trigger**: Push to `main` branch
- **Service**: `rag-api-prd`
- **Resources**: 4GB RAM, 2 CPU
- **Scaling**: 1-100 instances (min 1 for warm starts)
- **Environment**: `PRD`

## 📝 How to Deploy

### First-Time Deployment

1. Complete all prerequisites above
2. Push to the appropriate branch:

```bash
# Deploy to staging
git push origin stg

# Deploy to production
git push origin main
```

3. Monitor deployment in GitHub Actions tab
4. Once deployed, get your service URL:

```bash
# Staging
gcloud run services describe rag-api-stg \
  --region=$REGION \
  --format='value(status.url)'

# Production
gcloud run services describe rag-api-prd \
  --region=$REGION \
  --format='value(status.url)'
```

### Manual Deployment

You can also trigger deployments manually from GitHub Actions:
1. Go to Actions → Select workflow
2. Click "Run workflow"
3. Select branch and run

## 🔍 Monitoring & Debugging

### View Logs

```bash
# Staging logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api-stg" \
  --limit=50 \
  --format=json

# Production logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-api-prd" \
  --limit=50 \
  --format=json
```

### Service Status

```bash
# Check service details
gcloud run services describe rag-api-stg --region=$REGION

# List all revisions
gcloud run revisions list --service=rag-api-stg --region=$REGION
```

### Test Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe rag-api-stg --region=$REGION --format='value(status.url)')

# Test health endpoint
curl $SERVICE_URL/api/v1/health

# Test API docs
open $SERVICE_URL/docs
```

## 🔧 Configuration

### Update Secrets

```bash
# Update a secret
echo -n "new-value" | gcloud secrets versions add SECRET_NAME --data-file=-

# Cloud Run will automatically use the latest version on next deployment
```

### Modify Service Configuration

Edit the workflow files:
- `.github/workflows/deploy-stg-gcp.yml` - Staging configuration
- `.github/workflows/deploy-prd-gcp.yml` - Production configuration

Common settings to adjust:
- `--memory`: RAM allocation (e.g., `2Gi`, `4Gi`)
- `--cpu`: CPU allocation (e.g., `1`, `2`, `4`)
- `--min-instances`: Minimum running instances
- `--max-instances`: Maximum scaling limit
- `--timeout`: Request timeout in seconds
- `--concurrency`: Max concurrent requests per instance

## 💰 Cost Optimization

### Staging (Lower Cost)
- Min instances: 0 (scales to zero when idle)
- Lower resources: 2GB RAM, 1 CPU
- Suitable for testing and development

### Production (Balanced)
- Min instances: 1 (always warm, better response times)
- Higher resources: 4GB RAM, 2 CPU
- Auto-scaling based on demand

### Estimated Costs
- **Idle staging**: ~$0/month (scales to zero)
- **Active staging**: ~$5-20/month
- **Production**: ~$30-100/month (depends on traffic)

See [Cloud Run Pricing](https://cloud.google.com/run/pricing) for details.

## 🔐 Security Best Practices

1. ✅ Use Workload Identity Federation (no long-lived keys)
2. ✅ Store secrets in Secret Manager (never in code)
3. ✅ Use least-privilege IAM roles
4. ✅ Enable Cloud Armor for DDoS protection
5. ✅ Set up VPC connector for private MongoDB access
6. ✅ Enable authentication for production endpoints
7. ✅ Monitor with Cloud Logging and Cloud Monitoring

## 🆘 Troubleshooting

### Deployment Fails

```bash
# Check recent deployments
gcloud run services describe rag-api-stg --region=$REGION

# View deployment logs
gcloud logging read "resource.type=cloud_run_revision" --limit=20
```

### Secret Access Issues

```bash
# Verify service account has access to secrets
gcloud secrets get-iam-policy SECRET_NAME

# Grant access if needed
gcloud secrets add-iam-policy-binding SECRET_NAME \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Container Crashes

```bash
# Get crash logs
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=50
```

## 📚 Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [Secret Manager](https://cloud.google.com/secret-manager/docs)
- [GitHub Actions for GCP](https://github.com/google-github-actions)

