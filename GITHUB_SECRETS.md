# GitHub Secrets Configuration

This document lists all the GitHub Secrets needed for the CI/CD workflows.

## How to Add Secrets

1. Go to your GitHub repository
2. Navigate to **Settings → Secrets and variables → Actions**
3. Click **New repository secret**
4. Add each secret listed below

---

## GCP Cloud Run Workflows

Required for both `deploy-stg-gcp.yml` and `deploy-prd-gcp.yml`

### Deployment Secrets (3 required)

```
GCP_PROJECT_ID
Description: Your Google Cloud project ID
Example: my-rag-project-123456

GCP_REGION
Description: Google Cloud region for deployment
Example: us-central1

GCP_SA_KEY
Description: Service account JSON key (entire file contents)
Example: {
  "type": "service_account",
  "project_id": "my-rag-project-123456",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "github-actions-deploy@project.iam.gserviceaccount.com",
  ...
}
```

### Application Secrets (stored in Google Secret Manager, not GitHub)

These must be created in Google Secret Manager:
- `OPENAI_API_KEY`
- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `MONGODB_VECTOR_COLLECTION`
- `GEMINI_API_KEY`
- `GOOGLE_CLOUD_PROJECT_ID`
- `VERTEX_REGION`

See `GCP_DEPLOYMENT.md` for instructions on creating these secrets.

---

## EC2 Workflow

Required for `deploy_ec2_stg.yml`

### Deployment Secrets

```
DOCKERHUB_USERNAME
Description: Your DockerHub username
Example: ravprojectdev

DOCKERHUB_TOKEN
Description: DockerHub access token (not password)
Get from: https://hub.docker.com/settings/security

EC2_HOST
Description: EC2 instance IP address or hostname
Example: 54.123.45.67

EC2_USER
Description: SSH username for EC2
Example: ubuntu

EC2_SSH_KEY
Description: Private SSH key for EC2 access
Example: -----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----

GCLOUD_KEY_JSON_B64
Description: Base64-encoded Google Cloud service account key
Get from: cat gcloud-key.json | base64
```

### Application Secrets (passed directly to container)

```
OPENAI_API_KEY
Description: OpenAI API key
Example: sk-...

SBERT_API_URL
Description: SBERT API endpoint URL
Example: https://api.example.com/sbert

MONGODB_URI
Description: MongoDB connection string
Example: mongodb+srv://user:pass@cluster.mongodb.net/

MONGODB_DB_NAME
Description: MongoDB database name
Example: rag_database

MONGODB_VECTOR_COLLECTION
Description: MongoDB collection for embeddings
Example: embeddings

GEMINI_API_KEY
Description: Google Gemini API key
Example: AIza...

GOOGLE_CLOUD_PROJECT_ID
Description: Google Cloud project ID
Example: my-project-123456

VERTEX_REGION
Description: Google Cloud Vertex AI region
Example: us-central1
```

---

## Quick Setup Checklist

### For GCP Cloud Run:
- [ ] `GCP_PROJECT_ID`
- [ ] `GCP_REGION`
- [ ] `GCP_SA_KEY`
- [ ] Create secrets in Google Secret Manager (see GCP_DEPLOYMENT.md)

### For EC2:
- [ ] `DOCKERHUB_USERNAME`
- [ ] `DOCKERHUB_TOKEN`
- [ ] `EC2_HOST`
- [ ] `EC2_USER`
- [ ] `EC2_SSH_KEY`
- [ ] `GCLOUD_KEY_JSON_B64`
- [ ] `OPENAI_API_KEY`
- [ ] `SBERT_API_URL`
- [ ] `MONGODB_URI`
- [ ] `MONGODB_DB_NAME`
- [ ] `MONGODB_VECTOR_COLLECTION`
- [ ] `GEMINI_API_KEY`
- [ ] `GOOGLE_CLOUD_PROJECT_ID`
- [ ] `VERTEX_REGION`

---

## Security Best Practices

1. ✅ Never commit secrets to git
2. ✅ Use environment-specific secrets (different values for STG/PRD)
3. ✅ Rotate service account keys regularly
4. ✅ Use least-privilege IAM roles
5. ✅ Enable secret scanning on GitHub
6. ✅ Audit secret access regularly

