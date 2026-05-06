#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/supabase_sync/push_to_dockerhub.sh <tag>
#
# Optional env:
#   DOCKERHUB_USERNAME=...
#   DOCKERHUB_TOKEN=...
#   IMAGE_NAME=ravprojectdev/rate-limit-sync

if [[ $# -lt 1 ]]; then
  echo "Usage: ./scripts/supabase_sync/push_to_dockerhub.sh <tag>"
  echo "Example: ./scripts/supabase_sync/push_to_dockerhub.sh monthly-sync"
  exit 1
fi

CUSTOM_TAG="$1"
IMAGE_NAME="${IMAGE_NAME:-ravprojectdev/rate-limit-sync}"
SHA_TAG="$(git rev-parse --short HEAD)"
PLATFORM="${PLATFORM:-linux/amd64}"

if [[ -n "${DOCKERHUB_USERNAME:-}" && -n "${DOCKERHUB_TOKEN:-}" ]]; then
  echo "${DOCKERHUB_TOKEN}" | docker login -u "${DOCKERHUB_USERNAME}" --password-stdin
else
  echo "Skipping docker login (set DOCKERHUB_USERNAME/DOCKERHUB_TOKEN to auto-login)."
fi

echo "Building and pushing ${IMAGE_NAME}:${SHA_TAG} and ${IMAGE_NAME}:${CUSTOM_TAG} for ${PLATFORM}"
docker buildx build \
  --platform "${PLATFORM}" \
  --provenance=false \
  -f scripts/supabase_sync/Dockerfile \
  -t "${IMAGE_NAME}:${SHA_TAG}" \
  -t "${IMAGE_NAME}:${CUSTOM_TAG}" \
  --push \
  .

echo "Done:"
echo "  ${IMAGE_NAME}:${SHA_TAG}"
echo "  ${IMAGE_NAME}:${CUSTOM_TAG}"
