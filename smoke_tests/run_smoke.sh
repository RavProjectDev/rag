#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_smoke.sh — Chat endpoint smoke test
#
# Usage:
#   cd smoke_tests
#   chmod +x run_smoke.sh
#   ./run_smoke.sh
#
# What it does:
#   1. Builds and starts the API via docker compose (AUTH_MODE=dev, no rate limit)
#   2. Waits for /api/v1/health to return 200
#   3. Sends two requests to POST /api/v1/chat/:
#        Test 1 — random gibberish (expect a "no context" response)
#        Test 2 — a real on-topic question (expect a substantive answer)
#   4. Prints the full responses so an agent or human can evaluate them
#      against smoke_tests/SPEC.md
#   5. Tears down the container
#
# Requires: docker, curl, jq  (jq is optional — output is still readable without it)
# ---------------------------------------------------------------------------

set -euo pipefail

if ! docker info > /dev/null 2>&1; then
  echo -e "\033[0;31m[ERROR]\033[0m  Docker is not running. Start Docker Desktop and re-run."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.smoke.yml"
BASE_URL="http://localhost:8080"
CHAT_URL="$BASE_URL/api/v1/chat/"
HEALTH_URL="$BASE_URL/api/v1/health"

VALID_QUESTION="What does the Rav say about the purpose of marriage in a Jewish life?"
GIBBERISH="$(LC_ALL=C tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 50 || true)"

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${BOLD}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}   $*"; }
error()   { echo -e "${RED}[ERROR]${RESET}  $*"; }

# ── pretty-print JSON if jq is available ─────────────────────────────────────
pretty_json() {
  if command -v jq &>/dev/null; then
    echo "$1" | jq '.'
  else
    echo "$1"
  fi
}

# ── cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
  info "Tearing down containers..."
  docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── 1. build & start ──────────────────────────────────────────────────────────
info "Building and starting API (this may take a minute on first run)..."
docker compose -f "$COMPOSE_FILE" up -d --build

# ── 2. wait for health ────────────────────────────────────────────────────────
info "Waiting for service to become healthy..."
MAX_WAIT=120
ELAPSED=0
until curl -sf "$HEALTH_URL" > /dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    error "Service did not become healthy after ${MAX_WAIT}s. Dumping logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail=50
    exit 1
  fi
  sleep 3
  ELAPSED=$((ELAPSED + 3))
done
success "Service is healthy (${ELAPSED}s)"

echo ""

# ── 3a. test 1 — gibberish ────────────────────────────────────────────────────
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD} TEST 1 — Nonsense input${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
info "Input: $GIBBERISH"
echo ""

HTTP_STATUS_1=$(curl -s -o /tmp/smoke_response_1.json -w "%{http_code}" \
  -X POST "$CHAT_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"$GIBBERISH\",
    \"submit_query\": false
  }")

echo "HTTP Status: $HTTP_STATUS_1"
echo ""
echo "Response:"
pretty_json "$(cat /tmp/smoke_response_1.json)"

echo ""
MAIN_TEXT_1=$(cat /tmp/smoke_response_1.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('main_text',''))" 2>/dev/null || echo "(could not parse)")
SOURCE_COUNT_1=$(cat /tmp/smoke_response_1.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('sources',[])))" 2>/dev/null || echo "?")

echo -e "${BOLD}── Summary ──────────────────────────────────────────────${RESET}"
echo "main_text : $MAIN_TEXT_1"
echo "sources   : $SOURCE_COUNT_1 item(s)"
echo ""

# ── 3b. test 2 — valid question ───────────────────────────────────────────────
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD} TEST 2 — Valid on-topic question${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
info "Input: $VALID_QUESTION"
echo ""

HTTP_STATUS_2=$(curl -s -o /tmp/smoke_response_2.json -w "%{http_code}" \
  -X POST "$CHAT_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"$VALID_QUESTION\",
    \"submit_query\": false
  }")

echo "HTTP Status: $HTTP_STATUS_2"
echo ""
echo "Response:"
pretty_json "$(cat /tmp/smoke_response_2.json)"

echo ""
MAIN_TEXT_2=$(cat /tmp/smoke_response_2.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('main_text',''))" 2>/dev/null || echo "(could not parse)")
SOURCE_COUNT_2=$(cat /tmp/smoke_response_2.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('sources',[])))" 2>/dev/null || echo "?")

echo -e "${BOLD}── Summary ──────────────────────────────────────────────${RESET}"
echo "main_text : $MAIN_TEXT_2"
echo "sources   : $SOURCE_COUNT_2 item(s)"
echo ""

# ── 4. basic exit-code checks (HTTP level only) ───────────────────────────────
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD} Result${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════${RESET}"

FAIL=0
if [ "$HTTP_STATUS_1" != "200" ]; then
  error "Test 1 returned HTTP $HTTP_STATUS_1 (expected 200)"
  FAIL=1
else
  success "Test 1 HTTP 200"
fi

if [ "$HTTP_STATUS_2" != "200" ]; then
  error "Test 2 returned HTTP $HTTP_STATUS_2 (expected 200)"
  FAIL=1
else
  success "Test 2 HTTP 200"
fi

echo ""
warn "Semantic assertions (does Test 1 refuse? does Test 2 make sense?) are"
warn "evaluated by reading the output above against smoke_tests/SPEC.md"
echo ""

if [ "$FAIL" -eq 1 ]; then
  error "One or more HTTP checks failed."
  exit 1
fi

success "All HTTP checks passed. Review responses above against SPEC.md."
