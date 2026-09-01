#!/usr/bin/env bash
# Live GraphRAG pipeline smoke: status -> upload -> stage samples -> artifacts -> graph chat.
set -euo pipefail

DEFAULT_BASE_URL="${GRAPHRAG_SMOKE_BASE:-http://localhost:3002}"
DEFAULT_DOMAIN="${GRAPHRAG_SMOKE_DOMAIN:-manual-smoke}"
DEFAULT_QUERY="${GRAPHRAG_SMOKE_QUERY:-Which graph path connects two important entities in this document, and what provenance should I cite?}"
DEFAULT_REQUIRE_PATHS="${GRAPHRAG_REQUIRE_PATHS:-true}"
DEFAULT_REQUIRE_SOURCE_ARTIFACT="${GRAPHRAG_REQUIRE_SOURCE_ARTIFACT:-true}"
DEFAULT_REQUIRE_EVIDENCE_FAMILIES="${GRAPHRAG_REQUIRE_EVIDENCE_FAMILIES:-}"
DEFAULT_MAX_CHUNKS="${GRAPHRAG_SMOKE_MAX_CHUNKS:-4}"
DEFAULT_UPLOAD_TO_REMOTE="${GRAPHRAG_UPLOAD_TO_REMOTE:-true}"

BASE_URL="$DEFAULT_BASE_URL"
DOMAIN="$DEFAULT_DOMAIN"
QUERY="$DEFAULT_QUERY"
REQUIRE_PATHS="$DEFAULT_REQUIRE_PATHS"
REQUIRE_SOURCE_ARTIFACT="$DEFAULT_REQUIRE_SOURCE_ARTIFACT"
REQUIRE_EVIDENCE_FAMILIES=()
MAX_CHUNKS="$DEFAULT_MAX_CHUNKS"
UPLOAD_TO_REMOTE="$DEFAULT_UPLOAD_TO_REMOTE"

is_false() {
  case "${1,,}" in
    false|0|no|off) return 0 ;;
    *) return 1 ;;
  esac
}

add_required_evidence_families() {
  local raw="${1:-}"
  local family
  raw="${raw//,/ }"
  for family in $raw; do
    [[ -n "$family" ]] && REQUIRE_EVIDENCE_FAMILIES+=("$family")
  done
}

add_required_evidence_families "$DEFAULT_REQUIRE_EVIDENCE_FAMILIES"

usage() {
  echo "Usage: $0 [options] /path/to/document [domain]" >&2
  echo "Options:" >&2
  echo "  --base-url URL        Frontend/API base URL (default: $DEFAULT_BASE_URL)" >&2
  echo "  --domain NAME         GraphRAG domain for the smoke run" >&2
  echo "  --query TEXT          Chat-context query" >&2
  echo "  --max-chunks N        Chat-context max_chunks value (default: $DEFAULT_MAX_CHUNKS)" >&2
  echo "  --local-only          Use /ingest/pipeline/preview instead of remote upload" >&2
  echo "  --remote              Use /ingest/pipeline/upload (default)" >&2
  echo "  --allow-no-paths      Do not fail when graph_paths=0" >&2
  echo "  --require-paths       Fail when graph_paths=0 (default)" >&2
  echo "  --allow-no-source     Do not fail when no source_document artifact is available" >&2
  echo "  --require-source      Fail when no source_document artifact is available (default)" >&2
  echo "  --require-evidence-family NAME" >&2
  echo "                         Require an extracted evidence family such as tables, charts, forms, styled_text, speaker_notes, connectors, images, links, or attachments. May be repeated." >&2
  echo "Env: GRAPHRAG_SMOKE_BASE, GRAPHRAG_SMOKE_DOMAIN, GRAPHRAG_SMOKE_QUERY, GRAPHRAG_SMOKE_MAX_CHUNKS, GRAPHRAG_REQUIRE_PATHS=false, GRAPHRAG_REQUIRE_SOURCE_ARTIFACT=false, GRAPHRAG_REQUIRE_EVIDENCE_FAMILIES=tables,charts,forms, GRAPHRAG_UPLOAD_TO_REMOTE=false" >&2
  echo "Set GRAPHRAG_REQUIRE_PATHS=false/0/no/off for small/simple documents where upload, artifacts, and chat context are enough." >&2
  echo "Set GRAPHRAG_REQUIRE_SOURCE_ARTIFACT=false/0/no/off when intentionally testing documents too large to keep as source artifacts." >&2
  echo "Set GRAPHRAG_UPLOAD_TO_REMOTE=false/0/no/off for local-only pipeline smoke in staged deployments." >&2
}

die_usage() {
  local message="${1:-}"
  if [[ -n "$message" ]]; then
    echo "$message" >&2
  fi
  usage
  exit 2
}

positionals=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --base-url)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        die_usage "--base-url requires a URL"
      fi
      BASE_URL="$2"
      shift 2
      ;;
    --domain)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        die_usage "--domain requires a non-empty name"
      fi
      DOMAIN="$2"
      shift 2
      ;;
    --query)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        die_usage "--query requires text"
      fi
      QUERY="$2"
      shift 2
      ;;
    --max-chunks)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        die_usage "--max-chunks requires a positive integer"
      fi
      MAX_CHUNKS="$2"
      shift 2
      ;;
    --local-only|--preview)
      UPLOAD_TO_REMOTE="false"
      shift
      ;;
    --remote|--upload)
      UPLOAD_TO_REMOTE="true"
      shift
      ;;
    --allow-no-paths)
      REQUIRE_PATHS="false"
      shift
      ;;
    --require-paths)
      REQUIRE_PATHS="true"
      shift
      ;;
    --allow-no-source|--allow-no-source-artifact)
      REQUIRE_SOURCE_ARTIFACT="false"
      shift
      ;;
    --require-source|--require-source-artifact)
      REQUIRE_SOURCE_ARTIFACT="true"
      shift
      ;;
    --require-evidence-family|--require-family)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        die_usage "$1 requires a family name"
      fi
      add_required_evidence_families "$2"
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        positionals+=("$1")
        shift
      done
      ;;
    -*)
      die_usage "Unknown option: $1"
      ;;
    *)
      positionals+=("$1")
      shift
      ;;
  esac
done

FILE_PATH="${positionals[0]:-${GRAPHRAG_SMOKE_FILE:-}}"
if [[ -z "$FILE_PATH" ]]; then
  die_usage "Document path is required"
fi
if [[ "${positionals[1]:-}" != "" ]]; then
  DOMAIN="${positionals[1]}"
fi
BASE_URL="${BASE_URL%/}"
if [[ -z "$BASE_URL" ]]; then
  die_usage "Base URL cannot be empty"
fi
if [[ -z "$DOMAIN" ]]; then
  die_usage "Domain cannot be empty"
fi
if ! [[ "$MAX_CHUNKS" =~ ^[1-9][0-9]*$ ]]; then
  die_usage "--max-chunks must be a positive integer"
fi
if [[ ! -f "$FILE_PATH" ]]; then
  echo "GraphRAG smoke file not found: $FILE_PATH" >&2
  exit 2
fi

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 2
  fi
}

need curl
need jq

tmp_upload="$(mktemp)"
tmp_detail="$(mktemp)"
tmp_chat="$(mktemp)"
tmp_samples="$(mktemp)"
tmp_source="$(mktemp)"
status_json="$(mktemp)"
trap 'rm -f "$tmp_upload" "$tmp_detail" "$tmp_chat" "$tmp_samples" "$tmp_source" "$status_json"' EXIT

echo "GraphRAG smoke base: $BASE_URL"
echo "Document: $FILE_PATH"
echo "Domain: $DOMAIN"
echo "Upload to remote GraphRAG: $UPLOAD_TO_REMOTE"

curl -fsS "$BASE_URL/api/v1/graphrag/services/status" >"$status_json"
status_line="$(
  jq -r '._meta | "ready=\(.ready) operational_ready=\(.operational_ready) mode=\(.operating_mode.id // .status // "unknown") workflows=\((.available_workflows // []) | join(",")) warnings=\((.warnings // []) | length)"' "$status_json"
)"
echo "Status: $status_line"
status_ready="$(jq -r '._meta.ready // ._meta.operational_ready // false' "$status_json")"
if [[ "$status_ready" != "true" ]]; then
  echo "GraphRAG services are not ready" >&2
  jq -r '
    ._meta
    | [
        ("summary=" + (.summary // "n/a")),
        ("available_workflows=" + ((.available_workflows // []) | join(","))),
        ("blocked_workflows=" + ((.blocked_workflows // []) | join(","))),
        ("next_steps=" + ((.next_steps // []) | map(if type == "object" then (.message // .label // .id // tostring) else tostring end) | join(" | ")))
      ]
    | .[]
  ' "$status_json" >&2
  exit 1
fi

if is_false "$UPLOAD_TO_REMOTE"; then
  pipeline_endpoint="$BASE_URL/api/v1/graphrag/ingest/pipeline/preview"
else
  pipeline_endpoint="$BASE_URL/api/v1/graphrag/ingest/pipeline/upload"
fi
echo "Pipeline endpoint: $pipeline_endpoint"

curl -fsS "$pipeline_endpoint" \
  -F "file=@${FILE_PATH}" \
  -F "domain=${DOMAIN}" \
  -F "emit_samples=true" \
  -F "submit_structured_text=true" \
  >"$tmp_upload"

run_id="$(jq -r '.run_id // empty' "$tmp_upload")"
if [[ -z "$run_id" ]]; then
  echo "Upload did not return a run_id" >&2
  jq '.' "$tmp_upload" >&2
  exit 1
fi

upload_summary="$(
  jq -r '"run_id=\(.run_id) status=\(.status) chunks=\(.summary.chunk_count // 0) relationships=\(.summary.relationship_candidate_count // 0) upstream=\(.upstream_ingest.status // "unknown") code=\(.upstream_ingest.status_code // "n/a")"' "$tmp_upload"
)"
echo "Upload: $upload_summary"

curl -fsS "$BASE_URL/api/v1/graphrag/ingest/pipeline/${run_id}" >"$tmp_detail"
detail_summary="$(
  jq -r '"artifacts=\(.artifacts.artifact_count // 0) resume_ready=\(.resume_handoff.ready) chunks=\(.summary.chunk_count // 0) relationships=\(.summary.relationship_candidate_count // 0)"' "$tmp_detail"
)"
echo "Detail: $detail_summary"

artifact_count="$(jq -r '.artifacts.artifact_count // 0' "$tmp_detail")"
resume_ready="$(jq -r '.resume_handoff.ready // false' "$tmp_detail")"
if (( artifact_count < 1 )); then
  echo "No pipeline artifacts were exposed for run ${run_id}" >&2
  exit 1
fi
if [[ "$resume_ready" != "true" ]]; then
  echo "Resume handoff was not ready for run ${run_id}" >&2
  exit 1
fi

sample_value_count() {
  local stage="$1"
  local key="$2"
  jq -r \
    --arg stage "$stage" \
    --arg key "$key" \
    '
      [
        .stages[]?
        | select(.name == $stage)
        | .sample_outputs[$key]
      ]
      | if length == 0 or .[0] == null then 0
        else (
          .[0]
          | if type == "array" or type == "object" or type == "string" then length
            elif type == "boolean" then (if . then 1 else 0 end)
            else 1
            end
        )
        end
    ' \
    "$tmp_detail"
}

require_stage_sample() {
  local stage="$1"
  local key="$2"
  local count
  count="$(sample_value_count "$stage" "$key")"
  if (( count < 1 )); then
    echo "Missing pipeline sample ${stage}.${key} for run ${run_id}" >&2
    exit 1
  fi
}

require_stage_sample "structure_extraction" "elements"
require_stage_sample "chunking" "chunks"
require_stage_sample "graph_candidates" "relationships"
require_stage_sample "embedding_plan" "embedding_inputs"
require_stage_sample "local_graph_artifact" "edges"
require_stage_sample "local_graph_reasoning_index" "path_candidates"
require_stage_sample "local_context_cache" "chunks"
require_stage_sample "graph_interaction_plan" "starter_prompts"

echo "Stage samples: extraction chunking graph embedding local_graph reasoning context interaction"

samples_artifact_endpoint="$(jq -r '.artifacts.artifacts[]? | select(.kind == "samples" and .available == true) | .endpoint // empty' "$tmp_detail" | head -n 1)"
if [[ -z "$samples_artifact_endpoint" ]]; then
  echo "No samples artifact was available for run ${run_id}" >&2
  exit 1
fi
samples_artifact_url="$samples_artifact_endpoint"
if [[ "$samples_artifact_url" == /* ]]; then
  samples_artifact_url="${BASE_URL}${samples_artifact_url}"
fi
curl -fsS "$samples_artifact_url" >"$tmp_samples"
samples_summary="$(
  jq -r '"elements=\((.elements // []) | length) chunks=\((.chunks // []) | length) entities=\((.entities // []) | length) relationships=\((.relationships // []) | length) highlights=\((.relationship_highlights // []) | length)"' "$tmp_samples"
)"
echo "Samples artifact: $samples_summary"

for samples_key in elements chunks entities relationships; do
  samples_count="$(jq -r --arg key "$samples_key" '(.[$key] // []) | length' "$tmp_samples")"
  if (( samples_count < 1 )); then
    echo "Samples artifact missing ${samples_key} for run ${run_id}" >&2
    exit 1
  fi
done

evidence_family_count() {
  local family="$1"
  jq -r \
    --arg family "$family" \
    '
      def norm: ascii_downcase | gsub("[ -]"; "_");
      def aliases($family):
        ($family | norm) as $f
        | if ["table", "tables", "table_fact", "table_facts"] | index($f) then
            ["table", "tables", "table_elements", "table_fact"]
          elif ["chart", "charts", "chart_fact", "chart_facts"] | index($f) then
            ["chart", "charts", "chart_fact"]
          elif ["form", "forms", "form_field", "form_fields"] | index($f) then
            ["form", "forms", "form_field"]
          elif ["styled_text", "style", "styles", "emphasis"] | index($f) then
            ["styled_text", "styled_text_blocks"]
          elif ["speaker_note", "speaker_notes", "notes"] | index($f) then
            ["speaker_notes", "speaker_note"]
          elif ["connector", "connectors", "arrow", "arrows"] | index($f) then
            ["connector", "connectors", "semantic_connector"]
          elif ["diagram", "diagrams"] | index($f) then
            ["diagram", "diagrams"]
          elif ["shape", "shapes"] | index($f) then
            ["shape", "shapes"]
          elif ["image", "images"] | index($f) then
            ["image", "images"]
          elif ["attachment", "attachments", "embedded_object", "embedded_objects"] | index($f) then
            ["attachment", "attachments", "embedded_objects", "embedded_object"]
          elif ["link", "links"] | index($f) then
            ["link", "links"]
          elif ["media"] | index($f) then
            ["media"]
          elif ["comment", "comments"] | index($f) then
            ["comment", "comments"]
          elif ["formula", "formulas"] | index($f) then
            ["formula", "formulas"]
          else
            [$f]
          end;
      def matches_alias($key; $aliases):
        ($key | norm) as $normalized
        | any($aliases[]; $normalized == . or $normalized == (. + "_count") or $normalized == (. + "_relationship_count"));
      aliases($family) as $aliases
      | [
          (.summary.detected_capabilities[]? | norm | select(. as $cap | any($aliases[]; $cap == .)) | 1),
          (.summary.evidence_profile.quick_counts // {} | to_entries[]? | select(matches_alias(.key; $aliases)) | .value),
          (.summary.evidence_profile.graph_evidence // {} | to_entries[]? | select(matches_alias(.key; $aliases)) | .value),
          (.stages[]? | select(.name == "graph_candidates") | .sample_outputs.relationship_sample_categories // {} | to_entries[]? | select(matches_alias(.key; $aliases)) | .value),
          (.samples.relationships // [] | .[]? | (.metadata.element_type // .kind // "" | norm) as $kind | select(any($aliases[]; $kind == .)) | 1)
        ]
      | map(tonumber? // 0)
      | add // 0
    ' \
    "$tmp_detail"
}

if ((${#REQUIRE_EVIDENCE_FAMILIES[@]} > 0)); then
  for family in "${REQUIRE_EVIDENCE_FAMILIES[@]}"; do
    family_count="$(evidence_family_count "$family")"
    if (( family_count < 1 )); then
      echo "Required evidence family '${family}' was not detected for run ${run_id}" >&2
      exit 1
    fi
    echo "Evidence family: ${family} count=${family_count}"
  done
fi

source_artifact_endpoint="$(jq -r '.artifacts.artifacts[]? | select(.kind == "source_document" and .available == true) | .endpoint // empty' "$tmp_detail" | head -n 1)"
source_artifact_content_type="$(jq -r '.artifacts.artifacts[]? | select(.kind == "source_document" and .available == true) | .content_type // empty' "$tmp_detail" | head -n 1)"
if [[ -z "$source_artifact_endpoint" ]]; then
  if ! is_false "$REQUIRE_SOURCE_ARTIFACT"; then
    echo "No source_document artifact was available for run ${run_id}" >&2
    exit 1
  fi
  echo "Source artifact: unavailable"
else
  source_artifact_url="$source_artifact_endpoint"
  if [[ "$source_artifact_url" == /* ]]; then
    source_artifact_url="${BASE_URL}${source_artifact_url}"
  fi
  if [[ "$source_artifact_url" == *\?* ]]; then
    source_artifact_url="${source_artifact_url}&download=false"
  else
    source_artifact_url="${source_artifact_url}?download=false"
  fi
  curl -fsS "$source_artifact_url" >"$tmp_source"
  source_artifact_bytes="$(wc -c <"$tmp_source" | tr -d '[:space:]')"
  if (( source_artifact_bytes < 1 )); then
    echo "Source artifact endpoint returned an empty body for run ${run_id}" >&2
    exit 1
  fi
  if [[ "${source_artifact_content_type,,}" == application/pdf* ]] && [[ "$(head -c 4 "$tmp_source")" != "%PDF" ]]; then
    echo "Source artifact for PDF run did not return PDF bytes" >&2
    exit 1
  fi
  echo "Source artifact: bytes=${source_artifact_bytes} content_type=${source_artifact_content_type:-unknown}"
fi

jq -n \
  --arg query "$QUERY" \
  --arg run_id "$run_id" \
  --arg domain "$DOMAIN" \
  --argjson max_chunks "$MAX_CHUNKS" \
  '{
    query: $query,
    pipeline_run_ids: [$run_id],
    domain: $domain,
    max_chunks: $max_chunks,
    include_graph: true,
    use_local_pipeline_context: true
  }' \
  | curl -fsS "$BASE_URL/api/v1/graphrag/chat-context-v2" \
      -H "Content-Type: application/json" \
      --data @- \
      >"$tmp_chat"

chat_summary="$(
  jq -r '"chunks=\((.chunks // []) | length) graph_status=\(.graph_reasoning.status // "unknown") graph_paths=\((.graph_reasoning.paths // []) | length) direct=\((.graph_reasoning.direct_relationships // []) | length) local_graph=\(.services_used.local_pipeline_graph) local_context=\(.services_used.local_pipeline_context) artifacts=\(.services_used.local_pipeline_artifacts) errors=\(.errors)"' "$tmp_chat"
)"
echo "Chat: $chat_summary"

chunks="$(jq -r '(.chunks // []) | length' "$tmp_chat")"
graph_status="$(jq -r '.graph_reasoning.status // "unknown"' "$tmp_chat")"
graph_paths="$(jq -r '(.graph_reasoning.paths // []) | length' "$tmp_chat")"
local_graph="$(jq -r '.services_used.local_pipeline_graph // false' "$tmp_chat")"
errors="$(jq -r '.errors // empty' "$tmp_chat")"

if [[ -n "$errors" && "$errors" != "null" ]]; then
  echo "Chat context returned errors: $errors" >&2
  exit 1
fi
if (( chunks < 1 )); then
  echo "Chat context returned no chunks" >&2
  exit 1
fi
if [[ "$graph_status" != "complete" ]]; then
  echo "Graph reasoning did not complete: $graph_status" >&2
  exit 1
fi
if [[ "$local_graph" != "true" ]]; then
  echo "Local pipeline graph was not used" >&2
  exit 1
fi
if ! is_false "$REQUIRE_PATHS" && [[ "$graph_paths" == "0" ]]; then
  echo "Graph path requirement enabled but no graph paths were returned" >&2
  exit 1
fi

echo "GraphRAG smoke passed for run ${run_id}"
