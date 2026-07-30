import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_BASED_DIR = BASE_DIR.parents[0]
GRAPH_DIR = BASE_DIR / "GraphRAG"
DEFAULT_GRAPH = GRAPH_DIR / "or_benchmark_graphrag_adaptive.json"
DEFAULT_DATASET = BASE_DIR / "Dataset" / "NL4OPT_old.csv"
DEFAULT_RESULT = BASE_DIR / "RESULT" / "NL4OPT_old_model_gpt-5.1__graphrag__with_validation__judge_gemini-3.1-pro-preview_20260623_153619" / "result.csv"
DEFAULT_LOG_DIR = BASE_DIR / "LOG" / "NL4OPT_old_model_gpt-5.1__graphrag__with_validation__judge_gemini-3.1-pro-preview_20260623_153619"
DEFAULT_OUTPUT_GRAPH = GRAPH_DIR / "or_benchmark_graphrag_adaptive.json"
DEFAULT_OUTPUT_RAG = GRAPH_DIR / "OR_Benchmark_ModelingPattern_RAG_adaptive.csv"
DEFAULT_PROMPT_OUT = GRAPH_DIR / "adaptive_update_prompt_preview.json"


SYSTEM_PROMPT = """You are an operations research modeling expert updating a GraphRAG knowledge base.
Your job is to read benchmark execution traces and extract reusable modeling knowledge.

Return strict JSON only. Do not use markdown.
The JSON schema is:
{
  "updates": [
    {
      "variant_id": "short_snake_case_id",
      "problem_type": "broad OR category",
      "variant": "specific modeling pattern name",
      "description": "when this pattern applies",
      "math_model": "compact mathematical modeling guidance",
      "code_snippet": "compact Python/Gurobi or Python/PuLP skeleton",
      "keywords": ["retrieval", "keywords"],
      "dataset_scope": ["NL4OPT"],
      "evidence_problem_ids": ["prob_..."],
      "errors": [
        {
          "error_type": "short error type",
          "typical_error": "common modeling or coding mistake",
          "fix_hint": "actionable correction"
        }
      ]
    }
  ]
}

Rules:
- Prefer reusable patterns over copying one specific problem.
- Add a new variant only when the existing retrieved graph context was too generic, mismatched, or missed a recurring substructure.
- Typical errors should be concrete and searchable.
- The code snippet must print exactly one objective value using `print("result:", value)`.
- Do not include API keys, file paths, or benchmark labels except evidence problem ids.
- If traces do not justify new knowledge, return {"updates": []}.
"""


def parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip().replace(",", "")
        if text.lower() in {"", "none", "no best solution", "infeasible", "code error", "nan"}:
            return None
        return float(text)
    except Exception:
        return None


def within_rel(pred: Any, label: Any, rel: float = 0.05) -> bool:
    p = parse_float(pred)
    y = parse_float(label)
    if p is None or y is None:
        return str(pred).strip().lower() == str(label).strip().lower()
    return abs(p - y) / max(abs(y), 1e-12) <= rel


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_result_values(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        field = reader.fieldnames[0] if reader.fieldnames else "result"
        return [(row.get(field) or "").strip() for row in reader]


def safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def compact_model_spec(model_spec: Any) -> Any:
    if not isinstance(model_spec, dict):
        return model_spec
    keep = {}
    for key in ["problem_summary", "problem_type", "objective", "decision_variables", "constraints", "assumptions"]:
        if key in model_spec:
            keep[key] = model_spec[key]
    return keep


def truncate(text: Any, limit: int = 1800) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def log_file_for(log_dir: Path, row_index: int, stem: str) -> Path | None:
    folder = log_dir / str(row_index)
    if not folder.exists():
        return None
    matches = list(folder.glob(stem))
    return matches[0] if matches else None


def build_trace(dataset_row: dict[str, str], pred: str, row_index: int, log_dir: Path) -> dict[str, Any]:
    final_state_path = log_file_for(log_dir, row_index, "final_state_*.json")
    final_state = safe_load_json(final_state_path) if final_state_path else None
    model_spec = None
    solver_code = ""
    final_status = ""
    review = ""
    graph_context = []
    retry_counts = {}

    if final_state:
        model_spec = compact_model_spec(final_state.get("model_spec"))
        solver_code = final_state.get("solver_code") or ""
        final_status = final_state.get("final_status") or ""
        review = final_state.get("final_review_feedback") or ""
        retry_counts = {
            "model_retry_count": final_state.get("model_retry_count", 0),
            "code_retry_count": final_state.get("code_retry_count", 0),
            "repair_retry_count": final_state.get("repair_retry_count", 0),
        }
        for ctx in (final_state.get("graph_rag_context") or {}).get("ranked_contexts", [])[:2]:
            graph_context.append({
                "problem_type": ctx.get("problem_type"),
                "variant": ctx.get("variant"),
                "score": ctx.get("score"),
                "typical_errors": ctx.get("typical_errors", [])[:2],
                "fix_hints": ctx.get("fix_hints", [])[:2],
            })

    label = dataset_row.get("Label", "")
    exact = parse_float(pred) is not None and parse_float(label) is not None and abs(parse_float(pred) - parse_float(label)) <= 1e-6
    return {
        "problem_id": dataset_row.get("count") or dataset_row.get("Problem") or f"row_{row_index}",
        "row_index": row_index,
        "query": truncate(dataset_row.get("Query", ""), 1400),
        "label": label,
        "prediction": pred,
        "exact": bool(exact),
        "within_5_percent": within_rel(pred, label),
        "final_status": final_status,
        "retry_counts": retry_counts,
        "retrieved_graph_context": graph_context,
        "model_spec": model_spec,
        "code_excerpt": truncate(solver_code, 1800),
        "validation_feedback": truncate(review, 900),
        "has_log": final_state is not None,
    }


def select_traces(dataset: list[dict[str, str]], preds: list[str], log_dir: Path, max_traces: int) -> list[dict[str, Any]]:
    traces = []
    for i, row in enumerate(dataset, start=1):
        if i > len(preds):
            break
        trace = build_trace(row, preds[i - 1], i, log_dir)
        traces.append(trace)

    failures = [t for t in traces if not t["within_5_percent"] or t["final_status"] not in {"", "SUCCESS"}]
    repaired = [
        t for t in traces
        if t["within_5_percent"] and (
            int(t.get("retry_counts", {}).get("model_retry_count", 0) or 0)
            + int(t.get("retry_counts", {}).get("code_retry_count", 0) or 0)
            + int(t.get("retry_counts", {}).get("repair_retry_count", 0) or 0)
        ) > 0
    ]
    successes = [t for t in traces if t["within_5_percent"] and t["final_status"] in {"", "SUCCESS"}]

    selected = failures[: max_traces]
    if len(selected) < max_traces:
        selected.extend(repaired[: max_traces - len(selected)])
    if len(selected) < max_traces:
        selected.extend(successes[: max_traces - len(selected)])
    return selected[:max_traces]


def build_update_prompt(traces: list[dict[str, Any]], graph_meta: dict[str, Any]) -> list[dict[str, str]]:
    user_payload = {
        "task": "Update an OR GraphRAG knowledge base from completed NL4OPT benchmark traces.",
        "current_graph_meta": graph_meta,
        "trace_selection_policy": "Prefer failed, repaired, or hard cases; generalize them into reusable graph patterns.",
        "traces": traces,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def load_openai_env() -> dict[str, str]:
    env = os.environ.copy()
    base = env.get("OPENAI_API_BASE") or env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    env["OPENAI_API_BASE"] = base
    env["OPENAI_BASE_URL"] = base
    return env


def call_llm(messages: list[dict[str, str]], model: str) -> dict[str, Any]:
    env = load_openai_env()
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available.")
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env["OPENAI_API_BASE"])
    params = {"model": model, "messages": messages}
    if model.lower().startswith("gpt-5"):
        params["max_completion_tokens"] = 6000
    else:
        params["temperature"] = 0
        params["max_tokens"] = 6000
    response = client.chat.completions.create(**params)
    text = response.choices[0].message.content or ""
    return parse_llm_json(text)


def parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def slug(text: str, limit: int = 90) -> str:
    value = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    value = "_".join(part for part in value.split("_") if part)
    return value[:limit] or "unknown"


def node_id(prefix: str, text: str) -> str:
    return f"{prefix}_{slug(text)}"


def add_node(nodes: list[dict[str, Any]], seen: set[str], node: dict[str, Any]) -> None:
    if node["id"] in seen:
        return
    seen.add(node["id"])
    nodes.append(node)


def add_edge(edges: list[dict[str, Any]], seen: set[tuple[str, str, str]], source: str, target: str, relation: str, weight: float = 1.0) -> None:
    key = (source, target, relation)
    if key in seen:
        return
    seen.add(key)
    edges.append({
        "id": f"e_{len(edges) + 1:04d}",
        "source": source,
        "target": target,
        "type": relation,
        "weight": weight,
    })


def apply_updates(graph: dict[str, Any], updates: list[dict[str, Any]], source_name: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    new_graph = json.loads(json.dumps(graph, ensure_ascii=False))
    nodes = new_graph.setdefault("nodes", [])
    edges = new_graph.setdefault("edges", [])
    seen_nodes = {node["id"] for node in nodes}
    seen_edges = {(edge["source"], edge["target"], edge["type"]) for edge in edges}
    rag_rows = []

    for update in updates:
        variant_id_raw = update.get("variant_id") or update.get("variant") or "adaptive_variant"
        variant_slug = "adaptive_" + slug(variant_id_raw)
        problem_type = str(update.get("problem_type") or "Adaptive OR Pattern")
        variant = str(update.get("variant") or variant_id_raw)
        errors = update.get("errors") or []
        if not isinstance(errors, list) or not errors:
            errors = [{"error_type": "AdaptiveModelingRisk", "typical_error": "The pattern is applied without checking all stated constraints.", "fix_hint": "Verify objective, constraints, domains, and output value against the problem statement."}]

        pt_id = node_id("pt", problem_type)
        var_id = node_id("var", variant_slug)
        desc_id = node_id("desc", variant_slug)
        model_id = node_id("model", variant_slug)
        code_id = node_id("code", variant_slug)

        add_node(nodes, seen_nodes, {"id": pt_id, "type": "ProblemType", "name": problem_type})
        add_node(nodes, seen_nodes, {
            "id": var_id,
            "type": "Variant",
            "name": variant,
            "problem_type": problem_type,
            "keywords": update.get("keywords") or [],
            "dataset_scope": update.get("dataset_scope") or ["NL4OPT"],
            "source": source_name,
            "evidence_problem_ids": update.get("evidence_problem_ids") or [],
        })
        add_node(nodes, seen_nodes, {"id": desc_id, "type": "ProblemDescription", "name": f"{variant} description", "text": update.get("description") or ""})
        add_node(nodes, seen_nodes, {"id": model_id, "type": "MathModel", "name": f"{variant} model", "text": update.get("math_model") or ""})
        add_node(nodes, seen_nodes, {"id": code_id, "type": "CodeSnippet", "name": f"{variant} code skeleton", "language": "python-optimization", "snippet": update.get("code_snippet") or ""})

        add_edge(edges, seen_edges, pt_id, var_id, "has_variant")
        add_edge(edges, seen_edges, var_id, desc_id, "has_description")
        add_edge(edges, seen_edges, desc_id, model_id, "formulated_as")
        add_edge(edges, seen_edges, model_id, code_id, "implemented_by")

        for error in errors[:4]:
            error_type = str(error.get("error_type") or "AdaptiveModelingRisk")
            err_id = node_id("err", f"{variant_slug}_{error_type}")
            fix_id = node_id("fix", f"{variant_slug}_{error_type}")
            typical_error = str(error.get("typical_error") or "")
            fix_hint = str(error.get("fix_hint") or "")
            add_node(nodes, seen_nodes, {"id": err_id, "type": "TypicalError", "name": error_type, "text": typical_error})
            add_node(nodes, seen_nodes, {"id": fix_id, "type": "FixHint", "name": f"{error_type} fix", "text": fix_hint})
            add_edge(edges, seen_edges, model_id, err_id, "prone_to", 0.9)
            add_edge(edges, seen_edges, err_id, fix_id, "fix_with", 0.95)
            add_edge(edges, seen_edges, fix_id, code_id, "implemented_by", 0.7)
            rag_rows.append({
                "id": f"adaptive_{len(rag_rows) + 1:03d}",
                "variant_id": variant_slug,
                "ProblemType": problem_type,
                "Variant": variant,
                "ProblemDescription": str(update.get("description") or ""),
                "MathModel": str(update.get("math_model") or ""),
                "CodeSnippet": str(update.get("code_snippet") or ""),
                "ErrorType": error_type,
                "TypicalError": typical_error,
                "FixHint": fix_hint,
                "DatasetScope": ";".join(update.get("dataset_scope") or ["NL4OPT"]),
                "RetrievalText": " ".join([
                    problem_type,
                    variant,
                    str(update.get("description") or ""),
                    str(update.get("math_model") or ""),
                    " ".join(update.get("keywords") or []),
                    typical_error,
                    fix_hint,
                ]),
                "SourceSeed": source_name,
            })

    meta = new_graph.setdefault("meta", {})
    meta["adaptive_update"] = {
        "source": source_name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "added_update_count": len(updates),
        "added_rag_rows": len(rag_rows),
    }
    meta["node_count"] = len(nodes)
    meta["edge_count"] = len(edges)
    return new_graph, rag_rows


def write_rag_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptively update GraphRAG from completed benchmark traces.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-graph", type=Path, default=DEFAULT_OUTPUT_GRAPH)
    parser.add_argument("--output-rag", type=Path, default=DEFAULT_OUTPUT_RAG)
    parser.add_argument("--prompt-out", type=Path, default=DEFAULT_PROMPT_OUT)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--max-traces", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset = read_csv(args.dataset)
    preds = read_result_values(args.result)
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    traces = select_traces(dataset, preds, args.log_dir, args.max_traces)
    messages = build_update_prompt(traces, graph.get("meta", {}))

    preview = {
        "dataset": str(args.dataset),
        "result": str(args.result),
        "log_dir": str(args.log_dir),
        "selected_trace_count": len(traces),
        "selected_problem_ids": [t["problem_id"] for t in traces],
        "messages": messages,
    }
    args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_out.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote prompt preview: {args.prompt_out}")
    print(f"Selected traces: {', '.join(preview['selected_problem_ids'])}")

    if args.dry_run:
        print("Dry run only. No LLM call and no graph update written.")
        return

    update_payload = call_llm(messages, args.model)
    updates = update_payload.get("updates") or []
    if not isinstance(updates, list):
        raise ValueError("LLM output field 'updates' must be a list.")

    new_graph, rag_rows = apply_updates(graph, updates, source_name=f"adaptive_update_from_{args.result.parent.name}")
    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    args.output_graph.write_text(json.dumps(new_graph, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rag_csv(args.output_rag, rag_rows)

    raw_out = args.output_graph.with_suffix(".llm_output.json")
    raw_out.write_text(json.dumps(update_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"LLM updates: {len(updates)}")
    print(f"Wrote updated graph: {args.output_graph}")
    print(f"Wrote adaptive RAG rows: {args.output_rag}")
    print(f"Wrote raw LLM output: {raw_out}")


if __name__ == "__main__":
    main()
