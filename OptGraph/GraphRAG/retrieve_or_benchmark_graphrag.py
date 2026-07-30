import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH_PATH = BASE_DIR / "or_benchmark_graphrag_adaptive.json"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def overlap_score(query: str, texts: Iterable[str]) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    doc_tokens = set()
    for text in texts:
        doc_tokens.update(tokenize(text))
    return len(q_tokens & doc_tokens) / max(1, len(q_tokens))


def phrase_keyword_bonus(query: str, keywords: list[str]) -> float:
    q = (query or "").lower()
    q_tokens = set(tokenize(q))
    bonus = 0.0
    for keyword in keywords:
        kw = keyword.lower()
        if " " in kw:
            if kw in q:
                bonus += 0.045
        elif kw in q_tokens:
            bonus += 0.014
    return min(bonus, 0.18)


def looks_like_direct_calculation(query: str) -> bool:
    q = (query or "").lower()
    direct_markers = [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bdetermine\b",
        r"\bavailable in total\b",
        r"\bby the start of\b",
        r"\bassuming\b",
        r"\bin total\b",
        r"\bwhat is the total\b",
    ]
    optimization_markers = [
        r"\bmaximize\b",
        r"\bminimize\b",
        r"\boptimal\b",
        r"\boptimize\b",
        r"\bdecision\b",
        r"\bobjective\b",
        r"\blp\b",
        r"\bmilp\b",
    ]
    has_direct = any(re.search(pattern, q) for pattern in direct_markers)
    has_optimization = any(re.search(pattern, q) for pattern in optimization_markers)
    return has_direct and not has_optimization


def pattern_bonus(query: str, node: dict) -> float:
    """Bias retrieval toward structural OR patterns and away from generic hygiene nodes."""
    q = (query or "").lower()
    variant = node.get("name", "")
    ptype = node.get("problem_type", "")
    keywords = node.get("keywords", [])
    bonus = phrase_keyword_bonus(q, keywords)
    direct_calc = looks_like_direct_calculation(q)

    pattern_rules = [
        ("Direct formula or derived KPI", r"\b(how many|how much|calculate|compute|determine|available in total|by the start of|assuming|in total|what is the total)\b", 0.35),
        ("Two-variable textbook LP", r"\b(two types|two kinds|both|printer|printers|make|makes|produce|produces|shared|combined|factory|team)\b", 0.15),
        ("Multi-activity resource allocation", r"\b(allocate|allocation|resource|capacity|limit|activity|unit|profit|cost)\b", 0.08),
        ("Multi-period production", r"\b(month|period|inventory|workforce|hiring|firing|outsourcing|holding)\b", 0.12),
        ("Jobshop", r"\b(job|operation|machine|processing time|precedence|makespan|completion)\b", 0.13),
        ("Time-window", r"\b(aircraft|landing|time window|separation|earliness|lateness|penalty)\b", 0.13),
        ("TSP", r"\b(tsp|vrp|vehicle|depot|city|visit|tour|route|routing|delivery)\b", 0.12),
        ("Network flow", r"\b(network|node|arc|edge|source|sink|flow|bandwidth|shortest path)\b", 0.12),
        ("Transportation", r"\b(transport|ship|shipment|warehouse|store|origin|destination|supply|demand)\b", 0.10),
        ("Diet", r"\b(diet|meal|food|nutrient|protein|calories|blend|mixture|ingredient)\b", 0.10),
        ("Facility", r"\b(facility|location|open|site|warehouse|center|serve|customer)\b", 0.10),
        ("Set cover", r"\b(cover|coverage|tower|area|population|served)\b", 0.11),
        ("Knapsack", r"\b(select|choose|investment|portfolio|asset|project|knapsack|risk|return)\b", 0.10),
        ("Price-demand", r"\b(price|demand function|revenue|profit|market research|discount|quadratic)\b", 0.12),
        ("Geometric", r"\b(rectangle|garden|area|perimeter|fencing|length|width)\b", 0.12),
        ("Ratio", r"\b(percent|percentage|ratio|proportion|share|fraction|twice|times as many|of total)\b", 0.10),
        ("Advertising", r"\b(marketing|advertising|ads|campaign|channel|budget|reach|audience)\b", 0.09),
        ("Staffing", r"\b(nurse|staff|shift|coverage|time period|worker|hospital|schedule)\b", 0.10),
        ("Production", r"\b(produce|production|manufacture|factory|machine|labor|raw material|component|product)\b", 0.09),
        ("Two-channel minimum-cost requirement", r"\b(marketing manager|advertising|channel x|channel y|campaign|minimum total cost|minimum cost|effectiveness score|cost per unit|budget allocated)\b", 0.28),
        ("Linearized ratio, multiple, and difference wording", r"\b(difference of at least|minimum excess|twice|three times|greater than|no more than|should not exceed|ratio|proportion|at least twice|at most twice)\b", 0.24),
        ("Positive-cost lower-bound optimum", r"\b(minimum number|minimum required|minimum hires|salary cost|employees|category x|category y|category z|total number cannot exceed)\b", 0.22),
        ("Cost, effectiveness, and objective-unit sanity check", r"\b(nearest dollar|minimum total cost in dollars|objective value|dollars|effectiveness|units|total cost)\b", 0.20),
        ("MAMO Easy monetary scale and reporting units", r"\b(thousand dollars|in thousands|nearest thousand|rounded to nearest thousand dollars|costs? are .*thousand|\\$[0-9,]{4,}|0\\.\\d+|percentage|rate|two decimal|three decimal)\b", 0.30),
        ("MAMO Easy objective value not allocation quantity", r"\b(minimum total cost|minimum cost|total expenditure|objective value|cost per unit|rounded to nearest dollar|cost associated|incurs costs?)\b", 0.28),
        ("MAMO Easy coefficient ledger audit", r"\b(cost per unit|each unit costs|costs? \\$|fatigue score|staffing cost|production cost|contributes .* score|minimum total cost|minimum total score|per hour|per unit)\b", 0.34),
        ("MAMO Easy one-sided difference constraints", r"\b(difference between|minus twice|twice that of|should not exceed|cannot exceed|no more than|at least .* more than|minimum excess)\b", 0.27),
        ("MAMO Easy small integer cross-check", r"\b(integer|integers|whole numbers|whole dollars|indivisible|x1|x2|x3|x4)\b", 0.18),
    ]
    target_text = f"{ptype} {variant}"
    for label, regex, value in pattern_rules:
        if label.lower() in target_text.lower() and re.search(regex, q):
            bonus += value

    if direct_calc:
        if variant == "Direct formula or derived KPI":
            bonus += 0.28
        elif ptype not in {"Deterministic OR Calculation", "Modeling Hygiene"}:
            bonus -= 0.08

    if variant == "Price-demand revenue or profit optimization" and not re.search(
        r"\b(price|discount|market research|demand function|for every|cents?|lower the price|raise the price|quadratic)\b",
        q,
    ):
        bonus -= 0.09

    if variant == "Jobshop or flowshop makespan" and not re.search(
        r"\b(job|operation|processing time|precedence|makespan|completion|flowshop|jobshop)\b",
        q,
    ):
        bonus -= 0.08

    mamo_easy_variants = {
        "Two-channel minimum-cost requirement",
        "Linearized ratio, multiple, and difference wording",
        "Positive-cost lower-bound optimum",
        "Cost, effectiveness, and objective-unit sanity check",
        "MAMO Easy monetary scale and reporting units",
        "MAMO Easy objective value not allocation quantity",
        "MAMO Easy coefficient ledger audit",
        "MAMO Easy one-sided difference constraints",
        "MAMO Easy small integer cross-check",
    }
    if variant in mamo_easy_variants:
        if re.search(r"\b(channel x|channel y|category x|category y|option x|option y|type x|type y|minimum|at least|no more than|cost|budget|effectiveness)\b", q):
            bonus += 0.08
        if re.search(r"\b(job|operation|machine|tsp|vehicle|route|warehouse|shipment|diet|nutrient)\b", q):
            bonus -= 0.10

    hygiene_types = {"Modeling Hygiene"}
    if ptype in hygiene_types:
        bonus -= 0.055
        if variant == "Integer, binary, and logical constraints" and re.search(r"\b(integer|binary|if|then|either|or|activate|number of|how many)\b", q):
            bonus += 0.045
        elif variant == "Table and matrix data parsing" and re.search(r"\b(table|matrix|data|row|column|distance|cost matrix|compatibility)\b", q):
            bonus += 0.07
        elif variant == "Objective extraction and solver status" and re.search(r"\b(infeasible|no best solution|objective value|status|optimal)\b", q):
            bonus += 0.045

    return bonus


def graph_path_or_fallback(path: Path | None = None) -> Path:
    if path is not None and path.exists():
        return path
    if DEFAULT_GRAPH_PATH.exists():
        return DEFAULT_GRAPH_PATH
    raise FileNotFoundError(f"GraphRAG file not found: {DEFAULT_GRAPH_PATH}")


def load_graph(path: Path | None = None) -> dict:
    graph_path = graph_path_or_fallback(path)
    return json.loads(graph_path.read_text(encoding="utf-8"))


def index_graph(graph: dict) -> tuple[dict, dict]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    outgoing = defaultdict(list)
    for edge in graph["edges"]:
        outgoing[edge["source"]].append(edge)
    return nodes, outgoing


def rank_variants(graph: dict, query: str, top_k: int = 3) -> list[dict]:
    scored = []
    nodes, outgoing = index_graph(graph)
    for node in graph["nodes"]:
        if node["type"] != "Variant":
            continue
        linked_texts = [
            node.get("name", ""),
            node.get("problem_type", ""),
            " ".join(node.get("keywords", [])),
            " ".join(node.get("dataset_scope", [])),
        ]
        for edge in outgoing[node["id"]]:
            target = nodes[edge["target"]]
            if target["type"] in {"ProblemDescription", "MathModel", "TypicalError", "FixHint"}:
                linked_texts.append(target.get("text", ""))
        base_score = overlap_score(query, linked_texts)
        scored.append({
            "score": base_score + pattern_bonus(query, node),
            "base_score": base_score,
            "node": node,
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def collect_variant_context(graph: dict, variant_id: str) -> dict:
    nodes, outgoing = index_graph(graph)
    variant = nodes[variant_id]
    context = {
        "problem_type": variant.get("problem_type", ""),
        "variant": variant.get("name", ""),
        "description": "",
        "math_model": "",
        "code_snippet": "",
        "typical_errors": [],
        "fix_hints": [],
        "similar_variants": [],
        "dataset_scope": variant.get("dataset_scope", []),
    }

    for edge in outgoing[variant_id]:
        target = nodes[edge["target"]]
        if edge["type"] == "has_description":
            context["description"] = target.get("text", "")
            for edge2 in outgoing[target["id"]]:
                model_node = nodes[edge2["target"]]
                if edge2["type"] != "formulated_as":
                    continue
                context["math_model"] = model_node.get("text", "")
                for edge3 in outgoing[model_node["id"]]:
                    target3 = nodes[edge3["target"]]
                    if edge3["type"] == "implemented_by":
                        context["code_snippet"] = target3.get("snippet", "")
                    elif edge3["type"] == "prone_to":
                        context["typical_errors"].append({
                            "error_type": target3.get("name", ""),
                            "text": target3.get("text", ""),
                        })
                        for edge4 in outgoing[target3["id"]]:
                            fix = nodes[edge4["target"]]
                            if edge4["type"] == "fix_with":
                                context["fix_hints"].append({
                                    "error_type": target3.get("name", ""),
                                    "text": fix.get("text", ""),
                                })
        elif edge["type"] == "similar_to" and target["type"] == "Variant":
            context["similar_variants"].append({
                "variant": target.get("name", ""),
                "problem_type": target.get("problem_type", ""),
                "weight": edge.get("weight", 1.0),
            })

    context["typical_errors"] = context["typical_errors"][:3]
    context["fix_hints"] = context["fix_hints"][:3]
    context["similar_variants"] = sorted(
        context["similar_variants"],
        key=lambda item: item.get("weight", 0),
        reverse=True,
    )[:3]
    return context


def retrieve_graph_context(query: str, top_k: int = 3, graph_path: Path | None = None) -> dict:
    graph = load_graph(graph_path)
    ranked = rank_variants(graph, query, top_k=top_k)
    contexts = []
    for item in ranked:
        context = collect_variant_context(graph, item["node"]["id"])
        context["score"] = item["score"]
        context["base_score"] = item["base_score"]
        contexts.append(context)
    return {
        "graph": graph.get("meta", {}),
        "ranked_contexts": contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve OR benchmark GraphRAG context.")
    parser.add_argument("query", help="Natural language optimization problem")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--graph-path", default="")
    args = parser.parse_args()

    path = Path(args.graph_path) if args.graph_path else None
    print(json.dumps(retrieve_graph_context(args.query, args.top_k, path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
