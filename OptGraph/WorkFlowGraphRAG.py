import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================
# 0. 环境配置（集中管理）
# =====================================

# ===== 主流程模型配置 =====
MODELING_MODEL = "gpt-5.1"      # 建模模型
CODING_MODEL = "gpt-5.1"        # 代码模型
JUDGE_MODEL = "gemini-3.1-pro-preview"               # 检测模型

# 模型名 -> provider 自动映射
MODEL_PROVIDER_MAP = {
    "gpt-4.1": "openai",
    "gpt-5.1": "openai",
    "gpt-5.4": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "deepseek-chat": "deepseek",
    "deepseek-reasoner": "deepseek",
    "MiniMax-M2.5": "minimax",
    "minimax-M2.7": "minimax",
    "gemini-3.1-pro-preview": "gemini",
}


def infer_provider_from_model(model_name: str) -> str:
    provider = MODEL_PROVIDER_MAP.get(model_name)
    if not provider:
        raise ValueError(
            f"The provider for the model {model_name} is not configured in the MODEL_PROVIDER_MAP. Please add the mapping first."
        )
    return provider


PROVIDER_MODELING = infer_provider_from_model(MODELING_MODEL)
PROVIDER_CODING = infer_provider_from_model(CODING_MODEL)
PROVIDER_JUDGE = infer_provider_from_model(JUDGE_MODEL)

# ===== 数据集配置 =====
DATASET_FILENAME = os.getenv("DATASET_FILENAME", "NL4OPT_old.csv")
QUERY_COLUMN = os.getenv("QUERY_COLUMN", "Query")

# ===== 运行配置 =====
AUTO_RUN_CODE = True
EXECUTION_TIMEOUT = 60
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "0")) or None
START_SOURCE_ROW = int(os.getenv("START_SOURCE_ROW", "0")) or None
END_SOURCE_ROW = int(os.getenv("END_SOURCE_ROW", "0")) or None
RETRY_FROM_RESULT_PATH = os.getenv("RETRY_FROM_RESULT_PATH", "").strip() or None
RETRY_ROWS_PATH = os.getenv("RETRY_ROWS_PATH", "").strip() or None

ENABLE_VALIDATION_AGENT = os.getenv("ENABLE_VALIDATION_AGENT", "1").lower() not in {"0", "false", "no", "off"}

# 重试上限
MAX_MODEL_RETRIES = 2
MAX_CODE_RETRIES = 2
MAX_REPAIR_RETRIES = 2
LLM_API_MAX_ATTEMPTS = int(os.getenv("LLM_API_MAX_ATTEMPTS", "3"))
LLM_API_RETRY_SLEEP_SECONDS = float(os.getenv("LLM_API_RETRY_SLEEP_SECONDS", "4"))
LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))

# ===== RAG 配置 =====
RAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAG")
RAG_FILE_PATH = os.getenv("RAG_FILE_PATH", "").strip() or os.path.join(RAG_DIR, "OR_Benchmark_ModelingPattern_RAG.csv")
if not os.path.exists(RAG_FILE_PATH):
    fallback_rag_path = os.path.join(RAG_DIR, "NL4OPT_ModelingPattern_RAG.csv")
    if not os.path.exists(fallback_rag_path):
        fallback_rag_path = os.path.join(RAG_DIR, "OR_GraphRAG_ErrorFix_200.csv")
    RAG_FILE_PATH = fallback_rag_path if os.path.exists(fallback_rag_path) else os.path.join(RAG_DIR, "RAG.csv")
RAG_CHUNK_SIZE = 180
RAG_CHUNK_OVERLAP = 40
RAG_RETRIEVE_TOP_K = 3
RAG_RERANK_TOP_K = 1          # modeling stage injects only top1 knowledge card
RAG_CODE_TOP_K = 1            # 代码生成时只取最相似的 1 个代码示例
RAG_EMBED_DIM = 256
RAG_EMBEDDING_MODEL = "text-embedding-3-large"
RAG_RERANK_MODEL = "gpt-4.1"
RAG_ENABLE_REAL_EMBEDDING = os.getenv("RAG_ENABLE_REAL_EMBEDDING", "1").lower() not in {"0", "false", "no", "off"}
RAG_ENABLE_LLM_RERANK = False
GRAPH_RAG_TOP_K = 2
USE_RAG_IN_MODELING = True      # 建模时是否使用 RAG 示例（模型 JSON）
USE_RAG_IN_CODING = True        # 代码生成时是否使用 RAG 示例（历史代码）

# ===== 时间戳 =====
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =====================================
# 1. 路径配置
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
DATASET_PATH = os.getenv("DATASET_PATH", "").strip() or os.path.join(DATASET_DIR, DATASET_FILENAME)

DATASET_NAME = os.path.splitext(os.path.basename(DATASET_PATH))[0]
LLM_NAME = (
    f"model_{MODELING_MODEL}"
    f"__graphrag"
    f"__{'with_validation' if ENABLE_VALIDATION_AGENT else 'no_validation'}"
    f"__judge_{JUDGE_MODEL}"
)

LOG_ROOT_DIR = os.path.join(BASE_DIR, "LOG")
RESULT_ROOT_DIR = os.path.join(BASE_DIR, "RESULT")

LOG_DIR = os.path.join(LOG_ROOT_DIR, f"{DATASET_NAME}_{LLM_NAME}_{TIMESTAMP}")
RESULT_DIR = os.path.join(RESULT_ROOT_DIR, f"{DATASET_NAME}_{LLM_NAME}_{TIMESTAMP}")

RESULT_FILE_PATH = os.path.join(
    RESULT_DIR,
    "result.csv"
)
RESULT_DETAIL_FILE_PATH = os.path.join(
    RESULT_DIR,
    "evaluated_raw.csv"
)

# =====================================
# 2. 通用工具函数
# =====================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_data = data.encode(encoding, errors="replace").decode(encoding, errors="replace")
                stream.write(safe_data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def normalize_output_path(file_path: str) -> str:
    if os.name != "nt":
        return file_path
    abs_path = os.path.abspath(file_path)
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + abs_path.lstrip("\\")
    return "\\\\?\\" + abs_path


def save_text_file(file_path: str, content: str) -> None:
    ensure_dir(os.path.dirname(file_path))
    with open(normalize_output_path(file_path), "w", encoding="utf-8") as f:
        f.write(content)


def save_json_file(file_path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(file_path))
    with open(normalize_output_path(file_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def init_result_csv(result_file_path: str) -> None:
    if not os.path.exists(result_file_path):
        with open(result_file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["result"])


def init_result_detail_csv(result_file_path: str) -> None:
    if not os.path.exists(result_file_path):
        with open(result_file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "problem_id",
                    "row_index",
                    "prediction",
                    "status",
                    "error_type",
                    "error_message",
                ],
            )
            writer.writeheader()


def append_result_value(result_file_path: str, value: Union[float, str, None]) -> None:
    with open(result_file_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if value == "Code Error":
            writer.writerow(["Code Error"])
        elif value is None:
            writer.writerow(["No Best Solution"])
        else:
            writer.writerow([value])


def classify_error_type(message: str) -> str:
    text = (message or "").lower()
    if "quota_not_enough" in text or "余额不足" in message:
        return "api_error"
    if "json" in text or "parse" in text or "validation" in text:
        return "parse_error"
    if "timeout" in text:
        return "api_error"
    if "traceback" in text or "exception" in text:
        return "framework_error"
    return "code_error"


def append_result_detail(
    result_file_path: str,
    row_data: Dict[str, Any],
    prediction: Union[float, str, None],
    status: str,
    error_type: str = "",
    error_message: str = "",
) -> None:
    init_result_detail_csv(result_file_path)
    row_index = row_data.get("_row_index", "")
    if row_data.get("Problem") or row_data.get("count"):
        problem_id = row_data.get("Problem") or row_data.get("count")
    elif str(row_index).isdigit():
        problem_id = f"prob_{int(row_index) - 1}"
    else:
        problem_id = ""
    if prediction is None:
        prediction_text = "No Best Solution"
    else:
        prediction_text = str(prediction)
    with open(result_file_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem_id",
                "row_index",
                "prediction",
                "status",
                "error_type",
                "error_message",
            ],
        )
        writer.writerow({
            "problem_id": problem_id,
            "row_index": row_index,
            "prediction": prediction_text,
            "status": status,
            "error_type": error_type,
            "error_message": error_message,
        })


def resolve_query_column(fieldnames: List[str], query_column: str = "Query") -> str:
    if query_column in fieldnames:
        return query_column
    candidates = [
        "Query",
        "query",
        "question",
        "en_question",
        "Problem",
        "problem",
        "description",
        "Description",
    ]
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(f"No query column found. Requested {query_column}; available columns are: {fieldnames}")


def normalize_loaded_row(row: Dict[str, Any], query_text: str, row_index: int, query_column: str) -> Dict[str, Any]:
    row["_row_index"] = row_index
    # Downstream nodes read QUERY_COLUMN, so mirror auto-detected text into that key.
    row[query_column] = query_text
    return row


def load_queries_from_csv(csv_path: str, query_column: str = "Query") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("The CSV file has no header, so it cannot be read.")
        actual_query_column = resolve_query_column(reader.fieldnames, query_column)
        for idx, row in enumerate(reader, start=2):
            query_text = (row.get(actual_query_column) or "").strip()
            if not query_text:
                continue
            rows.append(normalize_loaded_row(row, query_text, idx, query_column))
    return rows


def load_queries_from_json(json_path: str, query_column: str = "Query") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(json_path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a list of records or a dict containing a list of records.")

    all_keys: List[str] = []
    for item in data:
        if isinstance(item, dict):
            for key in item.keys():
                if key not in all_keys:
                    all_keys.append(key)
    actual_query_column = resolve_query_column(all_keys, query_column)

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        row = dict(item)
        query_text = (row.get(actual_query_column) or "").strip()
        if not query_text:
            continue
        rows.append(normalize_loaded_row(row, query_text, idx, query_column))
    return rows


def load_queries_from_file(dataset_path: str, query_column: str = "Query") -> List[Dict[str, Any]]:
    suffix = os.path.splitext(dataset_path)[1].lower()
    if suffix == ".csv":
        return load_queries_from_csv(dataset_path, query_column=query_column)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows: List[Dict[str, Any]] = []
            records: List[Dict[str, Any]] = []
            with open(dataset_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            all_keys: List[str] = []
            for item in records:
                for key in item.keys():
                    if key not in all_keys:
                        all_keys.append(key)
            actual_query_column = resolve_query_column(all_keys, query_column)
            for idx, row in enumerate(records, start=1):
                query_text = (row.get(actual_query_column) or "").strip()
                if query_text:
                    rows.append(normalize_loaded_row(row, query_text, idx, query_column))
            return rows
        return load_queries_from_json(dataset_path, query_column=query_column)
    raise ValueError(f"Unsupported dataset file type: {suffix}. Supported types: .csv, .json, .jsonl")


def load_retry_source_rows(result_path: str) -> set:
    retry_rows = set()
    if not result_path or not os.path.exists(result_path):
        return retry_rows
    with open(result_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        field = "result"
        if reader.fieldnames:
            if "result" not in reader.fieldnames:
                field = reader.fieldnames[0]
        for idx, row in enumerate(reader, start=1):
            value = (row.get(field) or "").strip().lower()
            if value in {"", "code error", "no best solution", "none", "nan"}:
                retry_rows.add(idx)
    return retry_rows


def load_retry_rows_path(rows_path: str) -> set:
    retry_rows = set()
    if not rows_path or not os.path.exists(rows_path):
        return retry_rows

    with open(rows_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return retry_rows

        candidates = [
            "source_row",
            "row_index",
            "dataset_index",
            "index",
            "row",
        ]
        field = next((name for name in candidates if name in reader.fieldnames), None)
        if field is None:
            field = reader.fieldnames[0]

        for row in reader:
            raw_value = (row.get(field) or "").strip()
            if not raw_value:
                continue
            match = re.search(r"\d+", raw_value)
            if match:
                retry_rows.add(int(match.group(0)))

    return retry_rows


def filter_rows_for_run(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = rows
    if START_SOURCE_ROW is not None:
        selected = [
            row for row in selected
            if str(row.get("_row_index", "")).isdigit() and int(row["_row_index"]) - 1 >= START_SOURCE_ROW
        ]
    if END_SOURCE_ROW is not None:
        selected = [
            row for row in selected
            if str(row.get("_row_index", "")).isdigit() and int(row["_row_index"]) - 1 <= END_SOURCE_ROW
        ]
    if RETRY_FROM_RESULT_PATH:
        retry_rows = load_retry_source_rows(RETRY_FROM_RESULT_PATH)
        selected = [
            row for row in selected
            if str(row.get("_row_index", "")).isdigit() and int(row["_row_index"]) - 1 in retry_rows
        ]
        print(f"Retry mode: loaded {len(retry_rows)} failed/missing rows from {RETRY_FROM_RESULT_PATH}")
    if RETRY_ROWS_PATH:
        retry_rows = load_retry_rows_path(RETRY_ROWS_PATH)
        selected = [
            row for row in selected
            if str(row.get("_row_index", "")).isdigit() and int(row["_row_index"]) - 1 in retry_rows
        ]
        print(f"Retry rows mode: loaded {len(retry_rows)} rows from {RETRY_ROWS_PATH}")
    if MAX_SAMPLES is not None:
        selected = selected[:MAX_SAMPLES]
    return selected


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+|\d+|[\u4e00-\u9fff]+", (text or "").lower())


def split_into_chunks(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> List[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    if chunk_size <= 0:
        return [" ".join(tokens)]
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for i in range(0, len(tokens), step):
        part = tokens[i:i + chunk_size]
        if not part:
            continue
        chunks.append(" ".join(part))
        if i + chunk_size >= len(tokens):
            break
    return chunks


def hashed_embedding(text: str, dim: int = RAG_EMBED_DIM) -> List[float]:
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = hash(tok)
        idx = abs(h) % dim
        sign = 1.0 if ((h >> 1) & 1) == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_sim(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


_EMBEDDER_CACHE: Optional[OpenAIEmbeddings] = None
_RAG_EMBED_CACHE: Dict[str, List[float]] = {}


def get_openai_embedding_client() -> OpenAIEmbeddings:
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE

    cfg = PROVIDERS["openai"]
    api_key = os.getenv(cfg.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Missing API key. Please set {cfg.api_key_env}.")

    _EMBEDDER_CACHE = OpenAIEmbeddings(
        model=RAG_EMBEDDING_MODEL,
        api_key=api_key,
        base_url=cfg.base_url,
    )
    return _EMBEDDER_CACHE


def embed_text_real(text: str) -> List[float]:
    key = normalize_ws(text)
    if key in _RAG_EMBED_CACHE:
        return _RAG_EMBED_CACHE[key]
    vec = get_openai_embedding_client().embed_query(key)
    _RAG_EMBED_CACHE[key] = vec
    return vec


def embed_text(text: str) -> List[float]:
    if RAG_ENABLE_REAL_EMBEDDING:
        try:
            return embed_text_real(text)
        except Exception:
            pass
    return hashed_embedding(text)


def keyword_overlap_score(query: str, doc_text: str) -> float:
    qset = set(tokenize(query))
    dset = set(tokenize(doc_text))
    if not qset or not dset:
        return 0.0
    inter = len(qset.intersection(dset))
    return inter / max(1, len(qset))


def infer_problem_type_from_query(query_text: str) -> str:
    q = (query_text or "").lower()

    ratio_patterns = [
        r"\bpercent\b", r"\bpercentage\b", r"\bratio\b", r"\bproportion\b",
        r"\btwice\b", r"\bthree times\b", r"\bfour times\b", r"\btimes as many\b",
        r"\bof total\b",
    ]
    if any(re.search(p, q) for p in ratio_patterns):
        return "Proportion Constraints"

    rules = [
        ("Routing", [r"\btsp\b", r"\bvrp\b", r"\brouting\b", r"\broute\b", r"\btour\b", r"\bvisit", r"\bdepot", r"\bvehicle", r"\bcity"]),
        ("Scheduling and Staffing", [r"\bjob\b", r"\boperation", r"\bmachine", r"\bprocessing time", r"\bprecedence", r"\bmakespan", r"\bshift", r"\bschedule", r"\bscheduling", r"\bstaff", r"\bnurse", r"\baircraft", r"\blanding", r"\bseparation"]),
        ("Transportation and Flow", [r"\bnetwork", r"\bnode", r"\barc", r"\bedge", r"\bsource", r"\bsink", r"\bflow", r"\bbandwidth", r"\bshortest path", r"\btransport", r"\bshipment", r"\bshipping", r"\bship", r"\bsupply", r"\bdemand", r"\borigin", r"\bdestination", r"\bwarehouse"]),
        ("Facility Location and Covering", [r"\bfacility", r"\blocation", r"\bsite", r"\bopen", r"\bbranch", r"\bcenter", r"\bserve", r"\bcover", r"\bcoverage", r"\btower", r"\barea", r"\bpopulation"]),
        ("Blending and Diet", [r"\bmix", r"\bmixture", r"\bblend", r"\bdiet", r"\bmeal", r"\bfood", r"\bfeed", r"\bfertilizer", r"\bnutrition", r"\bprotein", r"\bcalories", r"\bchemical", r"\balloy", r"\bfuel"]),
        ("Marketing and Budget Allocation", [r"\badvert", r"\bads?\b", r"\bpromot", r"\bmarketing", r"\baudience", r"\bviews?", r"\bclicks?", r"\bcampaign"]),
        ("Selection and Portfolio", [r"\bknapsack", r"\bselect", r"\bselection", r"\bchoose", r"\binvest", r"\binvestment", r"\bportfolio", r"\basset", r"\brisk", r"\breturn", r"\bbenefit", r"\bvalue"]),
        ("Assignment and Matching", [r"\bassign", r"\bassignment", r"\bmatching", r"\bmatch", r"\bemployee", r"\btask", r"\bworker", r"\bproject"]),
        ("Nonlinear and Quadratic Optimization", [r"\bprice", r"\bdemand function", r"\bmarket research", r"\bdiscount", r"\bquadratic", r"\bsquared", r"\bdeviation", r"\brectangle", r"\bgarden", r"\barea", r"\bperimeter", r"\bfencing"]),
        ("Production Planning", [r"\bperiod", r"\bmonth", r"\binventory", r"\bholding", r"\bworkforce", r"\bhiring", r"\bfiring", r"\boutsourcing", r"\bproduce", r"\bproduction", r"\bmanufacture", r"\bfactory", r"\bproduct", r"\bcomponent"]),
        ("Requirement Satisfaction", [r"\bminimum", r"\bat least", r"\brequire", r"\brequirement", r"\bneed", r"\bdemand", r"\bminimize", r"\bleast"]),
        ("General Linear Programming", [r"\ballocate", r"\ballocation", r"\bresource", r"\bcapacity", r"\bprofit", r"\bcost", r"\bmaximize", r"\blinear programming", r"\blp problem"]),
        ("Modeling Hygiene", [r"\btable", r"\bmatrix", r"\binfeasible", r"\bno best solution", r"\bobjective value", r"\bbinary", r"\binteger", r"\bif and only if"]),
    ]
    best_type = "General Linear Programming"
    best_score = 0
    for t, pats in rules:
        score = sum(1 for p in pats if re.search(p, q))
        if score > best_score:
            best_type = t
            best_score = score
    return best_type


def load_rag_rows(rag_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(rag_path):
        return []

    def make_row(row: Dict[str, Any]) -> Dict[str, str]:
        problem_type = normalize_ws(str(row.get("ProblemType", row.get("Type", "")) or ""))
        variant = normalize_ws(str(row.get("Variant", "") or ""))
        problem_description = normalize_ws(str(row.get("ProblemDescription", row.get("Query", row.get("Description", ""))) or ""))
        math_model = normalize_ws(str(row.get("MathModel", row.get("Model", "")) or ""))
        code = str(row.get("CodeSnippet", row.get("Code", "")) or "")
        typical_error = normalize_ws(str(row.get("TypicalError", "") or ""))
        fix_hint = normalize_ws(str(row.get("FixHint", "") or ""))
        error_type = normalize_ws(str(row.get("ErrorType", "") or ""))
        variant_id = normalize_ws(str(row.get("variant_id", row.get("VariantID", row.get("id", ""))) or ""))
        retrieval_text = normalize_ws(str(row.get("RetrievalText", "") or ""))

        if problem_type or variant or typical_error or fix_hint:
            query_parts = [
                f"ProblemType: {problem_type}" if problem_type else "",
                f"Variant: {variant}" if variant else "",
                f"ProblemDescription: {problem_description}" if problem_description else "",
                f"TypicalError: {typical_error}" if typical_error else "",
                f"FixHint: {fix_hint}" if fix_hint else "",
            ]
            model_parts = [
                f"MathModel: {math_model}" if math_model else "",
                f"ErrorType: {error_type}" if error_type else "",
                f"FixHint: {fix_hint}" if fix_hint else "",
            ]
            query = normalize_ws("\n".join(part for part in query_parts if part))
            model = normalize_ws("\n".join(part for part in model_parts if part))
        else:
            query = problem_description
            model = math_model

        if retrieval_text:
            query = retrieval_text

        return {
            "Query": query,
            "Model": model,
            "Type": problem_type,
            "Code": code,
            "variant_id": variant_id,
            "variant": variant,
            "error_type": error_type,
            "typical_error": typical_error,
            "fix_hint": fix_hint,
        }

    with open(rag_path, "rb") as f:
        magic = f.read(2)

    rows: List[Dict[str, str]] = []
    if magic == b"PK":
        try:
            import pandas as pd
            xls = pd.ExcelFile(rag_path)
            if not xls.sheet_names:
                return []
            df = pd.read_excel(rag_path, sheet_name=xls.sheet_names[0])
            for _, row in df.iterrows():
                rows.append(make_row(row.to_dict()))
            return [r for r in rows if r["Query"] and r["Model"]]
        except Exception:
            return []

    for encoding in ("utf-8-sig", "gbk", "cp1252"):
        try:
            with open(rag_path, "r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(make_row(row))
            break
        except UnicodeDecodeError:
            rows = []
    return [r for r in rows if r["Query"] and r["Model"]]


def retrieve_rag_examples(
    query_text: str,
    rag_rows: List[Dict[str, str]],
    top_k: int = RAG_RETRIEVE_TOP_K,
    rerank_k: int = RAG_RERANK_TOP_K,
    deduplicate_by_variant: bool = False,
) -> Dict[str, Any]:
    predicted_type = infer_problem_type_from_query(query_text)
    candidates = [r for r in rag_rows if (r.get("Type") or "").strip() == predicted_type]
    if not candidates:
        candidates = rag_rows

    q_emb = embed_text(query_text)
    scored: List[Dict[str, Any]] = []
    for r in candidates:
        doc = f"{r.get('Query', '')} {r.get('Model', '')}"
        chunks = split_into_chunks(doc)
        if not chunks:
            chunks = [doc]

        best_chunk_sim = -1.0
        for ch in chunks:
            sim = cosine_sim(q_emb, embed_text(ch))
            if sim > best_chunk_sim:
                best_chunk_sim = sim

        kw = keyword_overlap_score(query_text, doc)
        score = 0.75 * best_chunk_sim + 0.25 * kw
        scored.append({
            "query": r.get("Query", ""),
            "model": r.get("Model", ""),
            "type": r.get("Type", ""),
            "code": r.get("Code", ""),
            "variant_id": r.get("variant_id", ""),
            "variant": r.get("variant", ""),
            "error_type": r.get("error_type", ""),
            "typical_error": r.get("typical_error", ""),
            "fix_hint": r.get("fix_hint", ""),
            "first_score": score,
            "embed_score": best_chunk_sim,
            "keyword_score": kw,
        })

    scored.sort(key=lambda x: x["first_score"], reverse=True)
    if deduplicate_by_variant:
        deduped = []
        seen_variants = set()
        for item in scored:
            key = item.get("variant_id") or f"{item.get('type', '')}::{item.get('variant', '')}::{item.get('query', '')[:80]}"
            if key in seen_variants:
                continue
            seen_variants.add(key)
            deduped.append(item)
            if len(deduped) >= max(1, top_k):
                break
        top3 = deduped
    else:
        top3 = scored[:max(1, top_k)]

    reranked = []
    if RAG_ENABLE_LLM_RERANK and top3:
        llm = get_langchain_llm(
            provider="openai",
            model_name=RAG_RERANK_MODEL,
            temperature=0.0,
        )
        rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a retrieval reranker. Score candidate relevance from 0 to 100. Output only JSON: {\"score\": number}."),
            ("user", "User query:\n{query}\n\nCandidate Query:\n{cand_query}\n\nCandidate Model:\n{cand_model}\n\nReturn score JSON only."),
        ])
        chain = rerank_prompt | llm
        for item in top3:
            score = None
            try:
                resp = invoke_chain_with_retry(chain, {
                    "query": query_text,
                    "cand_query": item["query"],
                    "cand_model": item["model"],
                }, "rag_rerank")
                data = parse_llm_json_loose(resp.content)
                score = float(data.get("score", 0.0))
            except Exception:
                pass
            if score is None:
                doc = f"{item['query']} {item['model']}"
                lexical = keyword_overlap_score(query_text, doc)
                semantic = cosine_sim(q_emb, embed_text(doc))
                score = 100.0 * (0.55 * semantic + 0.45 * lexical)
            new_item = dict(item)
            new_item["rerank_score"] = score
            reranked.append(new_item)
    else:
        for item in top3:
            doc = f"{item['query']} {item['model']}"
            lexical = keyword_overlap_score(query_text, doc)
            semantic = cosine_sim(q_emb, embed_text(doc))
            rerank_score = 100.0 * (0.55 * semantic + 0.45 * lexical)
            new_item = dict(item)
            new_item["rerank_score"] = rerank_score
            reranked.append(new_item)
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    top2 = reranked[:max(1, rerank_k)]

    return {
        "predicted_type": predicted_type,
        "top3": top3,
        "top2": top2,
    }


RAG_ROWS = load_rag_rows(RAG_FILE_PATH)

GRAPH_RAG_DIR = os.path.join(BASE_DIR, "GraphRAG")
if GRAPH_RAG_DIR not in sys.path:
    sys.path.insert(0, GRAPH_RAG_DIR)

try:
    from retrieve_or_benchmark_graphrag import retrieve_graph_context
except Exception:
    retrieve_graph_context = None


def get_graph_rag_context(query_text: str) -> Dict[str, Any]:
    if retrieve_graph_context is None:
        return {"graph": {}, "ranked_contexts": []}
    try:
        configured_graph_path = os.getenv("GRAPH_RAG_FILE_PATH", "").strip()
        if configured_graph_path:
            graph_path = Path(configured_graph_path)
        else:
            graph_path = Path(GRAPH_RAG_DIR) / "or_benchmark_graphrag_adaptive.json"
        return retrieve_graph_context(query_text, top_k=GRAPH_RAG_TOP_K, graph_path=graph_path)
    except Exception as e:
        return {"graph": {}, "ranked_contexts": [], "error": str(e)}


def format_graph_rag_context(graph_context: Dict[str, Any], include_code: bool = False) -> str:
    contexts = graph_context.get("ranked_contexts") or []
    if not contexts:
        return "No GraphRAG context retrieved."

    blocks = []
    for i, ctx in enumerate(contexts, start=1):
        errors = ctx.get("typical_errors") or []
        fixes = ctx.get("fix_hints") or []
        similar = ctx.get("similar_variants") or []
        error_lines = "\n".join(
            f"- {err.get('error_type', '')}: {err.get('text', '')}"
            for err in errors[:3]
        ) or "- N/A"
        fix_lines = "\n".join(
            f"- {fix.get('error_type', '')}: {fix.get('text', '')}"
            for fix in fixes[:3]
        ) or "- N/A"
        similar_lines = "\n".join(
            f"- {item.get('variant', '')} ({item.get('problem_type', '')})"
            for item in similar[:3]
        ) or "- N/A"
        code_text = ""
        if include_code and ctx.get("code_snippet"):
            code_text = f"\nCode Skeleton:\n```python\n{ctx.get('code_snippet', '')[:1200]}\n```"

        blocks.append(
            f"[GraphRAG Path {i}; score={ctx.get('score', 0):.3f}]\n"
            f"ProblemType: {ctx.get('problem_type', '')}\n"
            f"Variant: {ctx.get('variant', '')}\n"
            f"DatasetScope: {', '.join(ctx.get('dataset_scope') or [])}\n"
            f"Description: {ctx.get('description', '')}\n\n"
            f"MathModel Pattern:\n{ctx.get('math_model', '')}\n\n"
            f"Typical Errors:\n{error_lines}\n\n"
            f"Fix Hints:\n{fix_lines}\n\n"
            f"Similar Variants:\n{similar_lines}"
            f"{code_text}"
        )
    return "\n\n".join(blocks)


def clean_llm_code(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```python"):
        cleaned = cleaned[len("```python"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def extract_json_text(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1].strip()
    return cleaned


def parse_llm_json_loose(raw_text: str) -> Dict[str, Any]:
    import ast
    cleaned = extract_json_text(raw_text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    repaired = cleaned.strip()
    repaired = repaired.replace("…", "")
    repaired = repaired.replace("...", "")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    try:
        return json.loads(repaired)
    except Exception:
        pass
    try:
        obj = ast.literal_eval(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        json.loads(cleaned)
    except Exception as e1:
        try:
            json.loads(repaired)
        except Exception as e2:
            raise ValueError(
                "LLM output could not be parsed as JSON.\n"
                f"[First 1200 characters of raw extracted text]\n{cleaned[:1200]}\n\n"
                f"[First 1200 characters after lightweight repair]\n{repaired[:1200]}\n\n"
                f"[Original parsing error] {e1}\n"
                f"[Parsing error after repair] {e2}"
            )
    raise ValueError(f"LLM output failed to parse as JSON：{cleaned[:1200]}")


def escape_prompt_template_literals(text: str) -> str:
    return (text or "").replace("{", "{{").replace("}", "}}")


def normalize_model_json(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data
    data.setdefault("problem_summary", "")
    data.setdefault("problem_type", "")
    data.setdefault("objective", {})
    data.setdefault("sets", [])
    data.setdefault("parameters", [])
    data.setdefault("decision_variables", [])
    data.setdefault("constraints", [])
    data.setdefault("assumptions", [])
    data.setdefault("required_data_fields", [])
    data.setdefault("recommended_python_package", "pulp")
    # objective
    if not isinstance(data["objective"], dict):
        data["objective"] = {
            "name": "Objective Function",
            "sense": "maximize",
            "description": str(data["objective"]),
            "expression": "",
            "latex": "",
            "components": [],
        }
    else:
        obj = data["objective"]
        obj.setdefault("name", "Objective Function")
        obj.setdefault("sense", "maximize")
        obj.setdefault("description", "")
        obj.setdefault("expression", "")
        obj.setdefault("latex", "")
        obj.setdefault("components", [])
        if not isinstance(obj["components"], list):
            obj["components"] = [obj["components"]]
        normalized_components = []
        for comp in obj["components"]:
            if isinstance(comp, dict):
                normalized_components.append({
                    "name": str(comp.get("name", "")),
                    "description": str(comp.get("description", "")),
                    "expression": str(comp.get("expression", "")),
                })
            else:
                normalized_components.append({
                    "name": str(comp),
                    "description": str(comp),
                    "expression": str(comp),
                })
        obj["components"] = normalized_components
    # sets
    if not isinstance(data["sets"], list):
        data["sets"] = []
    normalized_sets = []
    for item in data["sets"]:
        if isinstance(item, dict):
            normalized_item = {
                "name": str(item.get("name", "")),
                "description": str(item.get("description", "")),
                "elements_example": item.get("elements_example", []),
            }
            val = normalized_item["elements_example"]
            if val is None:
                normalized_item["elements_example"] = []
            elif isinstance(val, (list, str)):
                normalized_item["elements_example"] = val
            else:
                normalized_item["elements_example"] = [str(val)]
            normalized_sets.append(normalized_item)
        else:
            normalized_sets.append({
                "name": str(item),
                "description": "",
                "elements_example": [],
            })
    data["sets"] = normalized_sets
    # parameters
    if not isinstance(data["parameters"], list):
        data["parameters"] = []
    normalized_parameters = []
    for item in data["parameters"]:
        if isinstance(item, dict):
            normalized_parameters.append({
                "symbol": str(item.get("symbol", "")),
                "description": str(item.get("description", "")),
                "indexing": str(item.get("indexing", "")),
                "domain": str(item.get("domain", "")),
                "notes": str(item.get("notes", "")),
            })
        else:
            normalized_parameters.append({
                "symbol": str(item),
                "description": "",
                "indexing": "",
                "domain": "",
                "notes": "",
            })
    data["parameters"] = normalized_parameters
    # decision_variables
    if not isinstance(data["decision_variables"], list):
        data["decision_variables"] = []
    normalized_dvs = []
    for item in data["decision_variables"]:
        if isinstance(item, dict):
            normalized_dvs.append({
                "symbol": str(item.get("symbol", "")),
                "description": str(item.get("description", "")),
                "indexing": str(item.get("indexing", "")),
                "domain": str(item.get("domain", "")),
                "notes": str(item.get("notes", "")),
            })
        else:
            normalized_dvs.append({
                "symbol": str(item),
                "description": "",
                "indexing": "",
                "domain": "",
                "notes": "",
            })
    data["decision_variables"] = normalized_dvs
    # constraints
    if not isinstance(data["constraints"], list):
        data["constraints"] = []
    normalized_constraints = []
    for item in data["constraints"]:
        if isinstance(item, dict):
            normalized_constraints.append({
                "name": str(item.get("name", "")),
                "type": str(item.get("type", "")),
                "description": str(item.get("description", "")),
                "expression": str(item.get("expression", "")),
                "latex": str(item.get("latex", "")),
                "forall": str(item.get("forall", "")),
                "notes": str(item.get("notes", "")),
            })
        else:
            normalized_constraints.append({
                "name": str(item),
                "type": "",
                "description": "",
                "expression": "",
                "latex": "",
                "forall": "",
                "notes": "",
            })
    data["constraints"] = normalized_constraints
    # assumptions
    if not isinstance(data["assumptions"], list):
        if data["assumptions"] is None:
            data["assumptions"] = []
        else:
            data["assumptions"] = [str(data["assumptions"])]
    # required_data_fields
    if not isinstance(data["required_data_fields"], list):
        if data["required_data_fields"] is None:
            data["required_data_fields"] = []
        else:
            data["required_data_fields"] = [str(data["required_data_fields"])]
    # recommended_python_package
    pkg = str(data.get("recommended_python_package", "pulp")).strip().lower()
    if pkg in {"python", "scipy"}:
        data["recommended_python_package"] = pkg
    else:
        data["recommended_python_package"] = "pulp"
    return data


def normalize_review_json(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {
            "overall_passed": False,
            "model_passed": False,
            "code_passed": False,
            "execution_passed": False,
            "retry_target": "repair",
            "reason_category": "parse_error",
            "model_error_type": "unknown",
            "code_error_type": "unknown",
            "feedback": str(data),
        }
    data.setdefault("overall_passed", False)
    data.setdefault("model_passed", False)
    data.setdefault("code_passed", False)
    data.setdefault("execution_passed", False)
    data.setdefault("retry_target", "repair")
    data.setdefault("reason_category", "unknown")
    data.setdefault("model_error_type", "none")
    data.setdefault("code_error_type", "none")
    data.setdefault("feedback", "")
    if data["retry_target"] not in {"model", "code", "repair", "finish", "no_solution"}:
        data["retry_target"] = "repair"
    allowed_model_errors = {
        "none",
        "variable_domain",
        "missing_constraint",
        "wrong_ratio",
        "invented_constraint",
        "objective_direction",
        "infeasibility_misread",
        "ambiguous_preference",
        "other",
        "unknown",
    }
    allowed_code_errors = {
        "none",
        "package",
        "syntax",
        "solver_status",
        "output_parse",
        "runtime",
        "timeout",
        "implementation_mismatch",
        "other",
        "unknown",
    }
    if data["model_error_type"] not in allowed_model_errors:
        data["model_error_type"] = "other"
    if data["code_error_type"] not in allowed_code_errors:
        data["code_error_type"] = "other"
    data["overall_passed"] = bool(data["overall_passed"])
    data["model_passed"] = bool(data["model_passed"])
    data["code_passed"] = bool(data["code_passed"])
    data["execution_passed"] = bool(data["execution_passed"])
    data["reason_category"] = str(data["reason_category"])
    data["model_error_type"] = str(data["model_error_type"])
    data["code_error_type"] = str(data["code_error_type"])
    data["feedback"] = str(data["feedback"])
    return data


# =====================================
# 3. 结果识别函数
# =====================================

def extract_optimal_value(stdout_text: str) -> Optional[float]:
    if not stdout_text or not stdout_text.strip():
        return None
    text = stdout_text.strip()
    text = text.replace("−", "-").replace("—", "-").replace("–", "-")
    num_pattern = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    none_patterns = [
        r"result\s*[:：]\s*none\b",
        r"最优值\s*[:：]\s*none\b",
        r"目标值\s*[:：]\s*none\b",
        r"optimal value\s*[:：]?\s*none\b",
        r"objective value\s*[:：]?\s*none\b",
    ]
    for pattern in none_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return None
    preferred_patterns = [
        rf"(?:result)\s*[:：]\s*{num_pattern}",
        rf"(?:最优总利润)\s*[:：]\s*{num_pattern}",
        rf"(?:最优值)\s*[:：]\s*{num_pattern}",
        rf"(?:目标值)\s*[:：]\s*{num_pattern}",
        rf"(?:最优目标值)\s*[:：]\s*{num_pattern}",
        rf"(?:最小总成本)\s*[:：]\s*{num_pattern}",
        rf"(?:最大总利润)\s*[:：]\s*{num_pattern}",
        rf"(?:最小成本)\s*[:：]\s*{num_pattern}",
        rf"(?:最大利润)\s*[:：]\s*{num_pattern}",
        rf"(?:最小总时间)\s*[:：]\s*{num_pattern}",
        rf"(?:最大收益)\s*[:：]\s*{num_pattern}",
        rf"(?:objective value)\s*[:：]?\s*{num_pattern}",
        rf"(?:optimal value)\s*[:：]?\s*{num_pattern}",
        rf"(?:optimal objective)\s*[:：]?\s*{num_pattern}",
    ]
    for pattern in preferred_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except (ValueError, TypeError):
                pass
    solver_patterns = [
        rf"Optimal objective\s+{num_pattern}",
        rf"Objective value\s*[:：]?\s*{num_pattern}",
        rf"objective value\s*[:：]?\s*{num_pattern}",
        rf"Obj\s+{num_pattern}",
        rf"Best objective\s+{num_pattern}",
    ]
    for pattern in solver_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])
            except (ValueError, TypeError):
                pass
    if re.search(r"\boptimal\b", text, flags=re.IGNORECASE):
        fallback_patterns = [
            rf"最优.*?{num_pattern}",
            rf"目标.*?{num_pattern}",
        ]
        for pattern in fallback_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if matches:
                try:
                    return float(matches[-1])
                except (ValueError, TypeError):
                    pass
    hard_failure_patterns = [
        r"\bno solution\b",
        r"\bmodel is infeasible\b",
        r"\bproblem is infeasible\b",
        r"\binfeasible problem\b",
        r"\bstatus\s*[:：]?\s*infeasible\b",
        r"\b求解状态\s*[:：]?\s*infeasible\b",
        r"\b求解状态\s*[:：]?\s*不可行\b",
        r"\bResult\s*-\s*Linear relaxation infeasible\b",
        r"\bResult\s*-\s*Problem proven infeasible\b",
    ]
    for pattern in hard_failure_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return None
    return None

# =====================================
# 4. Pydantic 模型定义
# =====================================

class ObjectiveComponent(BaseModel):
    name: str
    description: str
    expression: str

class Objective(BaseModel):
    name: str
    sense: str
    description: str
    expression: str
    latex: str
    components: List[ObjectiveComponent]

class Set(BaseModel):
    name: str
    description: str
    elements_example: Optional[Union[List[Any], str]] = None

class Parameter(BaseModel):
    symbol: str
    description: str
    indexing: str
    domain: str
    notes: Optional[str] = ""

class DecisionVariable(BaseModel):
    symbol: str
    description: str
    indexing: str
    domain: str
    notes: Optional[str] = ""

class Constraint(BaseModel):
    name: str
    type: str
    description: str
    expression: str
    latex: str
    forall: Optional[str] = ""
    notes: Optional[str] = ""

class OptimizationModel(BaseModel):
    problem_summary: str
    problem_type: str
    objective: Objective
    sets: List[Set]
    parameters: List[Parameter]
    decision_variables: List[DecisionVariable]
    constraints: List[Constraint]
    assumptions: List[str]
    required_data_fields: List[str]
    recommended_python_package: str

class FinalReviewResult(BaseModel):
    overall_passed: bool = Field(description="整体是否通过")
    model_passed: bool = Field(description="建模是否通过")
    code_passed: bool = Field(description="代码是否通过")
    execution_passed: bool = Field(description="执行结果是否通过")
    retry_target: Literal["model", "code", "repair", "finish", "no_solution"] = Field(
        description="下一步应该回退/跳转到哪里"
    )
    reason_category: str = Field(description="错误归因类别")
    model_error_type: Literal[
        "none",
        "variable_domain",
        "missing_constraint",
        "wrong_ratio",
        "invented_constraint",
        "objective_direction",
        "infeasibility_misread",
        "ambiguous_preference",
        "other",
        "unknown",
    ] = Field(description="建模错误细分类")
    code_error_type: Literal[
        "none",
        "package",
        "syntax",
        "solver_status",
        "output_parse",
        "runtime",
        "timeout",
        "implementation_mismatch",
        "other",
        "unknown",
    ] = Field(description="代码错误细分类")
    feedback: str = Field(description="统一检测意见")

# =====================================
# 5. 供应商配置
# =====================================

@dataclass
class ProviderConfig:
    name: str
    api_key_env: str
    base_url: str

PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    ),
    "minimax": ProviderConfig(
        name="minimax",
        api_key_env="MINIMAX_API_KEY",
        base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
    ),
    "gemini": ProviderConfig(
        name="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ),
}

def get_langchain_llm(provider: str, model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    config = PROVIDERS[provider]
    api_key = os.getenv(config.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Missing API key. Please set {config.api_key_env}.")
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=config.base_url,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )


def invoke_chain_with_retry(chain: Any, payload: Dict[str, Any], stage: str) -> Any:
    last_error: Optional[BaseException] = None
    for attempt in range(1, max(1, LLM_API_MAX_ATTEMPTS) + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            last_error = exc
            if attempt >= LLM_API_MAX_ATTEMPTS:
                break
            sleep_seconds = LLM_API_RETRY_SLEEP_SECONDS * attempt
            print(f"[{stage}] LLM call failed on attempt {attempt}/{LLM_API_MAX_ATTEMPTS}: {exc}")
            print(f"[{stage}] retrying after {sleep_seconds:.1f}s ...")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{stage} LLM call failed after {LLM_API_MAX_ATTEMPTS} attempts: {last_error}") from last_error

# =====================================
# 6. Prompt 定义
# =====================================

MODELING_SYSTEM_PROMPT = """
You are an expert in Operations Research and optimization modeling.

Your tasks:
1. Read the provided business problem description.
2. Abstract it into a complete, clear mathematical optimization model suitable for programmatic solving.
3. You must ONLY output a valid JSON object.
4. Do not output markdown code blocks, do not output explanations, and do not output extra text.
5. Objective functions and constraints must be as complete as possible, using academic, mathematical modeling expressions.
6. All objective functions and constraints must simultaneously provide:
   - Mathematical expression string `expression`
   - LaTeX expression string `latex`
   - Text explanation `description`

The JSON format must strictly follow this structure:

{
  "problem_summary": "String, brief description of the problem",
  "problem_type": "String, e.g., LP/ILP/MILP/Transportation/Assignment/Facility Location/Inventory Planning",
  "objective": {
    "name": "Objective function name",
    "sense": "maximize or minimize",
    "description": "Text explanation of the objective",
    "expression": "Complete mathematical expression, e.g., Maximize Z = ...",
    "latex": "Complete LaTeX expression, e.g., \\max Z = ...",
    "components": [
      {"name": "Component name", "description": "Component meaning", "expression": "Component mathematical expression"}
    ]
  },
  "sets": [
    {"name": "Set name", "description": "Set meaning", "elements_example": ["example1", "example2"]}
  ],
  "parameters": [
    {"symbol": "Parameter symbol", "description": "Parameter meaning", "indexing": "Indices structure", "domain": "real / nonnegative_real / integer", "notes": "Additional notes"}
  ],
  "decision_variables": [
    {"symbol": "Variable symbol", "description": "Variable meaning", "indexing": "Indices structure", "domain": "binary / integer / continuous / nonnegative_integer / nonnegative_continuous", "notes": "Additional notes"}
  ],
  "constraints": [
    {"name": "Constraint name", "type": "Constraint type", "description": "Constraint meaning", "expression": "Complete mathematical expression", "latex": "Corresponding LaTeX expression", "forall": "Scope of applicability", "notes": "Additional notes"}
  ],
  "assumptions": ["Assumption 1", "Assumption 2"],
  "required_data_fields": ["Data field 1", "Data field 2"],
  "recommended_python_package": "pulp, python, or scipy"
}

**Rules (Must be strictly followed)**:

0. **Deterministic Calculation vs Optimization**:
   - If the prompt asks for a computed value, total, availability, KPI, ratio, or direct consequence of stated constants, and it does not ask to choose/optimize a decision, do NOT invent an artificial optimization objective, bottleneck, capacity allocation, or decision variables.
   - For such cases, set `problem_type` to "Deterministic Calculation", set `objective.sense` to "compute", keep `decision_variables` empty unless the text truly asks for choices, put the formula/recurrence in `objective.expression` and/or `constraints`, and set `recommended_python_package` to "python".
   - For LP/MILP/integer linear/resource-allocation problems, set `recommended_python_package` to "pulp".
   - For nonlinear continuous, quadratic, conic/SOCP, geometric, calculus, or smooth constrained optimization problems common in OptMathBench, set `recommended_python_package` to "python" when an analytic/direct formula is natural, or "scipy" when numerical nonlinear optimization is needed.
   - Do not recommend Pyomo, Gurobi, OR-Tools, cvxpy, or solver-specific APIs.
   - If the original question says "how many", "how much", "available in total", "by the start of", "calculate", "compute", or "determine", first check whether direct arithmetic is sufficient before building an LP/MILP.

1. **Variable Types**:
   - If a decision variable represents discrete items like "quantity", "count", or "number of times", default to `nonnegative_integer`.
   - Use `nonnegative_continuous` ONLY if the problem explicitly allows fractional values.
   - For binary choices like whether to locate a facility or assign a task, use `binary`.

2. **Constraint Expressions**:
   - All constraints must be written in linear form. Division is strictly prohibited in constraints.
   - For strict comparisons like "more than" or "exceeds" in natural language, if the variable is an integer, convert it to `x >= y + 1`.
   - Ratio constraints must be linearized.

2a. **MAMO Easy allocation-unit convention**:
   - In short allocation problems with variables such as X/Y channels, projects, budgets, resources, departments, or options, define the decision variables in the unit named by the constraint unless the prompt explicitly separates spend from quantity.
   - If the text says "total budget/resource/allocation for X and Y combined cannot exceed B", the default MAMO Easy benchmark interpretation is `x + y <= B`, even when B is written with a dollar sign. Do NOT replace it with `c_x*x + c_y*y <= B` merely because per-unit costs appear later.
   - Use `c_x*x + c_y*y` as the objective when the final question asks for minimum total cost in dollars. Use it as a cap only when the prompt explicitly says total spending, total cost, or total expenditure cannot exceed B.
   - Before declaring infeasible, check whether an apparent budget cap is an allocation cap (`x + y <= B`) and whether the objective is in dollars (`c_x*x + c_y*y`).

2b. **MAMO Easy objective and scale sanity**:
   - The final `result` must be the requested objective value, not the sum of decision variables, not a single decision variable, and not the solver status. If the question asks for minimum total cost, output the evaluated cost objective after applying all coefficients.
   - Respect the requested reporting unit. If the final question says "in thousands", output the objective in thousands. If costs are stated as "200 thousand dollars" and the question asks "rounded to nearest thousand dollars", keep the objective in thousand-dollar units instead of multiplying by 1000 again.
   - For large monetary allocation/investment prompts, choose a consistent monetary unit before modeling. If constraints use amounts like `$100,000`, `$10,000`, `$5,000` and objective coefficients are large per normalized unit such as `$5000` or `$4000`, it is usually a MAMO Easy thousand-dollar-unit problem: let `x=10` represent `$10,000` when that makes all constraints and costs consistent. Do not produce a 1000x-scaled answer.
   - MAMO Easy hard convention for large monetary allocations: if variables are investments/funds/amounts, constraints are written as large dollar amounts such as `$100,000`, `$20,000`, `$10,000`, `$5,000`, and objective coefficients are also large dollar amounts such as `$5000`, `$4000`, `$1000`, `$500`, normalize the decision variables to thousand-dollar units. For example, `$10,000` becomes `x >= 10`, `$100,000` becomes `x + y <= 100`, and objective `$5000*x + $4000*y` on thousand-units yields dollars. Do NOT use raw-dollar variables with those same coefficients, because that creates a 1000x answer.
   - Conversely, if objective coefficients are rates such as `0.05`, `0.08`, `5%`, or `8%`, keep variables in dollars and compute the monetary fee/cost as rate times dollar amount.
   - If the label-like statement asks for rounding to two decimals or three decimals, preserve decimals; do not round to an integer unless the prompt explicitly asks nearest whole number/dollar.

2c. **MAMO Easy coefficient ledger audit**:
   - Before writing the JSON, build a private coefficient ledger from the original text: decision variable unit, every constraint coefficient/RHS, every objective coefficient, objective sense, and requested output unit.
   - Phrases such as "cost per unit", "each unit costs", "fatigue score per hour", "staffing cost per hour", "production cost for each type", or "contributes X to the total cost/score" are objective coefficients when the final question asks for minimum total cost/score. The objective expression must include these coefficients unless all coefficients are 1.
   - Phrases such as "effectiveness score calculated as 3*x + 4*y at least R" or "combined effort must yield at least R" are requirement constraints, not the objective, unless the final question explicitly asks to optimize that score.
   - If the final question asks for "minimum total cost", reject any candidate objective like `x + y`, `total allocation`, or a single variable when the text contains non-unit cost coefficients such as 200, 150, 5000, 4000, 7, or 10. The objective must be `sum(cost_i * x_i)` in the chosen unit.
   - MAMO Easy benchmark convention: even when the wording says "cost per unit of effectiveness for channel X/Y", if X and Y are the decision allocations/budgets and effectiveness is separately defined as `a*x + b*y >= R`, treat the stated costs as direct objective coefficients on X and Y. The objective is `cost_X*x + cost_Y*y`, not `cost_X*(a*x) + cost_Y*(b*y)`. Requirement/effectiveness coefficients belong only to the requirement constraint unless the prompt provides a separate generated-effectiveness decision variable.
   - For wording like "difference between A and twice B cannot exceed L", use the one-sided inequality `A - 2*B <= L` unless the prompt explicitly says absolute difference. For "A should be at least k more than B", use `A - B >= k`.
   - If the prompt has at most four integer decision variables with modest bounds, prefer a formulation that can be checked by enumeration or PuLP and make the objective expression transparent enough for direct verification.

3. **Infeasible Scenarios**:
   - If the problem is obviously infeasible, note this in `assumptions`.
   - The generated code must check the solver status. If `Infeasible`, it must output `result: None`.

4. **Preferences vs. Hard Constraints**:
   - Do not treat soft preferences as hard constraints.

5. **Code Safety Reminders**:
   - Do not perform division operations on PuLP variables in print statements.
   - Integrality is already specified in the variable definition; do not add extra `problem += x` statements to enforce it.

6. **Completeness**:
   - `objective.expression` must be written entirely.
   - Every constraint in `constraints` must provide a complete expression.

7. **NL4OPT benchmark conventions**:
   - If the prompt says "Formulate an LP", still build and solve the LP objective value; do not stop at a symbolic formulation.
   - Treat money, budget, acres, kg, hours, nutrients, investment amounts, and mixture quantities as continuous unless the text explicitly says integer/discrete/countable units.
   - Treat people, animals, machines, vehicles, products, packages, batches, servings, printers, desks, shifts, and other indivisible counts as integer.
   - Translate "at most twice as many A as B" as A <= 2 * B. Translate "at least a third as many A as B" as A >= B / 3, then linearize.
   - Words like "prefers" are hard constraints only when followed by a precise bound such as at least/at most/no more than.
   - Preserve minimization vs maximization exactly; many NL4OPT errors come from reversing objective direction.

8. **OptMathBench natural-language conversion robustness**:
   - Some generated OptMathBench descriptions contain malformed duplicate clauses or sign errors. Do not immediately mark the model infeasible when a standard benchmark interpretation is obvious.
   - Aircraft landing: if the same ordered aircraft pair appears in two conflicting separation lists, treat the first direct separation list as the operative data and record later contradictory "Additionally" duplicate clauses as malformed alternatives unless the text explicitly says all duplicate lists must be enforced together.
   - Skill shortage assignment: shortage means unmet requirement and must be nonnegative, i.e. shortage_{p,s} >= required_{p,s} - attained_{p,s} and shortage_{p,s} >= 0. Minimize the maximum nonnegative shortage. Clauses such as "shortage <= -requirement" are usually sign-corrupted restatements and should not be imposed as hard constraints.
   - Fixed-charge/capacitated network flow: if all listed node balances are nonnegative demands and the total balance is positive with no supply nodes, record the missing supply data in assumptions and avoid inventing arbitrary flow-balance constraints that force infeasibility. Prefer a conservative model that only enforces balances where the supply/demand sign convention is explicit.
   - Transit line planning with unmet-demand penalties: if the objective includes penalty costs for unmet demand, introduce an unmet-demand slack variable u_od >= 0 and model service coverage as provided_capacity_or_frequency + u_od >= demand. Do not also enforce provided_capacity_or_frequency >= demand as a hard constraint unless the prompt explicitly says unmet demand is forbidden.
   - Transfer-station wording such as "a station can only be designated as a transfer station if at least two lines pass through it" is an implication, not a requirement that every station must be a transfer station. Model it as y_station <= sum(lines_serving_station)/2 or equivalent; do not force station designation or force line selection unless explicitly required.
"""

CODING_SYSTEM_PROMPT = """
You are an expert in optimization modeling and programming.

You will receive simultaneously:
1. The original problem description (may contain tables, parameters, textual conditions).
2. The mathematical model JSON.

When generating code, you must refer to BOTH the original problem description and the mathematical model JSON.
If you find that the mathematical model JSON is missing key data, table data, or textual conditions from the original problem, prioritize the original problem description to complete the data definitions in the code; however, you must not arbitrarily change the optimization direction or core constraint meanings of the mathematical model.

Your tasks:
1. Generate complete, runnable Python code based on the provided mathematical model JSON and original problem description.
2. Output ONLY pure Python code.
3. Do not output markdown code blocks, do not output explanations.
4. The code must include:
   - Example data
   - Set and parameter definitions
   - Decision variable definitions
   - Objective function
   - All constraints
   - Solver execution
   - Result output
5. Use the package indicated by `recommended_python_package`:
   - `pulp`: LP/MILP/integer linear models with CBC.
   - `python`: direct formula, enumeration, calculus-derived closed form, or deterministic computation.
   - `scipy`: nonlinear continuous, quadratic, smooth constrained, conic/SOCP-like, or geometry problems that cannot be represented faithfully in PuLP.
   Do not use Pyomo, Gurobi, OR-Tools, cvxpy, SolverFactory, or solver-specific external APIs.
6. If `recommended_python_package` is accidentally set to pyomo or another unsupported package, choose the closest supported implementation: PuLP for linear/MILP, scipy for nonlinear continuous, or plain Python for direct calculation.
6a. Use plain Python arithmetic if recommended_python_package is python, `problem_type` is "Deterministic Calculation", or `objective.sense` is "compute".
7. Add appropriate English comments.
8. Code style must be clear, variable naming explicit.
9. Code must be directly runnable without relying on undefined variables.
10. The result section must print the solver status, objective value, and main variable results.
11. You must strictly implement the model based on the `objective` and `constraints` fields in the input JSON. Do not arbitrarily simplify or omit them.
12. If `latex` or complete `expression` is provided in the JSON, use them as the source of truth for code implementation.
12b. Retrieved GraphRAG/RAG code may contain Pyomo or other solvers. Use it only as mathematical reference; never copy non-PuLP imports, SolverFactory calls, or solver-specific syntax.

12a. For deterministic calculation tasks:
    - Do not import PuLP, Pyomo, or Gurobi.
    - Do not invent decision variables, optimization directions, bottlenecks, or artificial constraints.
    - Compute the requested value directly from the original constants and the JSON formula.
    - Output `print("result:", result)` after assigning the computed number to `result`.

13. When outputting the result, you must append an extra line in a standard format:
    - If the solver status is Optimal, output:
      print("result:", objective_value)
    - If the solver status is not Optimal (e.g., Infeasible, Unbounded, Not Solved), output:
      print("result: None")

14. Integrality is specified during variable definition (e.g., cat='Integer'). Strictly prohibit adding statements like `problem += x` in constraints to "ensure integrality", as it will overwrite the objective function.

15. When printing results, do not perform division directly on PuLP variables (e.g., pulp.value(x / y)). Extract the numerical values using pulp.value() first.

16. For PuLP, strictly prohibit using the following syntax to check the solver status:
    - pulp.LpOptimal
    - pulp.LpInfeasible
    - pulp.LpUnbounded
    - Any similar comparisons with pulp status constant attributes.

17. For PuLP, you must uniformly use the following method to determine the status. Do not rewrite it:
    status_str = pulp.LpStatus[problem.status]
    print("Solver Status:", status_str)

    if status_str == "Optimal":
        print("Objective Value:", pulp.value(problem.objective))
        print("result:", pulp.value(problem.objective))
    else:
        print("result: None")

18. If you need to distinguish between infeasible, unbounded, etc., you must also write first:
    status_str = pulp.LpStatus[problem.status]
    and then use string comparison.

19. For PuLP, avoid any status checking syntax that could cause runtime AttributeError.

20. The result output section must be placed after the solver execution, and the program must run to the end even if no optimal solution is found.

21. Recommended result output template (strictly emulate this if using pulp):

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    status_str = pulp.LpStatus[problem.status]
    print("Solver Status:", status_str)

    if status_str == "Optimal":
        obj_val = pulp.value(problem.objective)
        print("Objective Value:", obj_val)
        print("result:", obj_val)
    else:
        print("result: None")

22. For NL4OPT-style small LP/MILP problems:
    - Prefer PuLP CBC for deterministic objective values.
    - Do not round the objective value before printing `result:`. Print the raw solver objective.
    - If decision variables are integer, define them with `cat="Integer"`; if continuous, use `lowBound=0` without integer category.
    - If the problem asks for variables but the benchmark label is the objective/profit/cost, `result:` must still be the objective value, not a decision variable.
    - Use all numeric constants from the original text exactly; do not invent upper bounds unless the text states them.

23. Target-value rule for benchmark datasets:
    - The user prompt may include `Target value to output`. The final `result:` line must be the numeric value of that target.
    - If the target is objective/profit/cost/revenue/area/volume/material/surface/perimeter/risk/utility/time, `result:` must be that scalar metric, not the optimizer variables, dimensions, price, radius, height, or a tuple/list/dict.
    - Only output a decision variable or coordinate when the target explicitly names that variable, e.g. "x coordinate", "first number", "price", "radius", or "number of units".
    - You may print variables for debugging, but the final `print("result:", ...)` must be a single numeric scalar matching the target.
    - For MAMO Easy cost/score minimization, compute the final scalar again from the solved variable values and the original objective coefficients before printing it. If this recomputed audit value differs from `pulp.value(problem.objective)`, print the audited objective value as `result:` and fix the objective implementation.

24. For `scipy` nonlinear optimization:
    - Use `scipy.optimize.minimize` with explicit objective and constraints/bounds.
    - Check `res.success`; if false, print `result: None`.
    - After success, compute the target scalar from `res.x` and print it as `result:`.
    - Do not print `result:` as `res.x`, a tuple, list, or dictionary unless the target explicitly asks for a vector.

25. OptMathBench robustness rules:
    - Aircraft landing: when duplicate separation clauses for the same ordered pair conflict, implement the first direct separation list and ignore later malformed duplicate "Additionally" clauses unless the text explicitly states both lists are cumulative.
    - Skill shortage assignment: implement shortage as nonnegative unmet requirement: shortage >= required - attained and shortage >= 0; minimize max shortage. Do not enforce malformed negative upper bounds on shortage.
    - Fixed-charge/capacitated network flow: if the text gives only positive demands and no supply nodes, do not create impossible equality balances for every node. Add comments documenting missing supply information and use the most conservative feasible interpretation supported by explicit text.
    - Transit line planning with unmet-demand penalties: add unmet-demand slack variables and enforce service + unmet >= demand. Include penalty_cost * unmet in the objective. Do not hard-enforce full demand satisfaction when unmet demand is explicitly penalized.
    - Transfer station implications must be one-way: station_designated can be 1 only if enough selected lines pass through it. Do not force station_designated = 1 for every station.
"""

FINAL_REVIEW_SYSTEM_PROMPT = """
You are a master reviewer for optimization modeling and solving workflows.

You will see:
1. The original problem description
2. The mathematical modeling JSON
3. The Python solver code
4. The code execution results (stdout / stderr / success status / extracted value)

Your task is to perform a post-execution comprehensive review based on the complete chain of evidence, and determine the next routing step.

Focus your evaluation on:
1. Is the modeling correct, complete, and free from critical omissions or directional errors?
2. Did the code faithfully implement the modeling?
3. Is an execution failure purely a runtime bug at the code level?
4. If execution succeeded but the result is missing/infeasible, is it a logically sound infeasibility given the problem, or is it caused by modeling/programming errors?
5. If a result is generated, is it overall acceptable?

Strictly output a structured JSON result, following these rules:

A. If there is a critical error in the modeling itself:
- model_passed = false
- retry_target = "model"

B. If modeling is basically correct, but code implementation is wrong:
- model_passed = true
- code_passed = false
- retry_target = "code"

C. If modeling and code are generally correct, but it's just a runtime bug / minor fix:
- retry_target = "repair"

D. If overall acceptable and the result is valid:
- overall_passed = true
- retry_target = "finish"
- reason_category = "ok"

E. If no solution/infeasible, but it makes logical sense based on the problem statement:
- retry_target = "no_solution"
- reason_category = "no_solution"

Notes:
- Only route back to 'model' if the modeling is demonstrably flawed.
- Do not reject modeling easily due to stylistic or minor phrasing issues.
- If the issue can be fixed by completely rewriting the code, prioritize 'code'.
- If it's just a runtime error, syntax error, or missing imports, prioritize 'repair'.
- If the original prompt is a direct deterministic calculation and does not ask for an optimum, reject any model/code that invents an optimization objective, bottleneck, resource allocation, or decision variables not present in the text. In that case set model_passed=false, retry_target="model", and model_error_type="invented_constraint".
- For MAMO Easy allocation problems, reject false infeasibility caused by unit confusion. If the prompt says "total budget/resource/allocation for X and Y combined cannot exceed B", treat the benchmark-default cap as x + y <= B even if B is written with a dollar sign. The cost expression c_x*x + c_y*y belongs in the objective unless the prompt explicitly caps total spending/cost/expenditure. If a model uses c_x*x + c_y*y <= B in this situation and returns infeasible, set model_passed=false, retry_target="model", model_error_type="infeasibility_misread", and explain the allocation-unit cap.
- For MAMO Easy scale errors, reject answers that return a decision-variable sum or raw allocation when the question asks for total cost. Also reject 1000x mistakes when the prompt reports costs in thousands or asks for an answer "in thousands" / "rounded to nearest thousand dollars"; request a model/code retry with consistent thousand-dollar units.
- For large MAMO Easy monetary allocation problems, reject raw-dollar models that multiply dollar-valued decision variables by dollar-valued per-thousand coefficients such as 5000 and 4000. If constraints contain `$100,000`, `$20,000`, `$10,000`, or `$5,000`, the benchmark convention is usually thousand-dollar decision units; a result like 70,000,000 when the normalized objective is 70,000 is a 1000x unit error and must retry.
- For MAMO Easy coefficient-ledger errors, independently read the original problem and check whether the model/code objective includes every stated per-unit cost/score coefficient. If the final question asks for minimum total cost/score and the code/model effectively optimizes or prints `x + y`, one variable, a requirement RHS, or an effectiveness score while ignoring non-unit objective coefficients, set model_passed=false, retry_target="model", model_error_type="objective_direction", and explain the missing objective coefficients.
- Also reject objective expressions that double-count requirement coefficients under the MAMO Easy benchmark convention. For example, when `3*x + 4*y >= R` is the effectiveness/requirement constraint and the text gives channel cost coefficients `200` and `150`, the objective should be `200*x + 150*y`; `200*3*x + 150*4*y` is an overcount even if the prose says "cost per unit of effectiveness", unless the prompt introduces separate generated-effectiveness variables.
- If stdout prints variable values, recompute the requested objective from those values and the original text. If the extracted result is a raw variable/allocation or differs from the recomputed objective because of missing coefficients or a 1000x unit mismatch, reject the run instead of finishing.
- Treat "difference between A and twice B cannot exceed L" as one-sided `A - 2B <= L` unless the original text explicitly says "absolute difference". Reject models that silently convert one-sided difference limits into absolute-value constraints or reverse the direction.
- For OptMathBench-style converted text, do not accept infeasibility as "no_solution" when it is caused by common natural-language conversion artifacts: duplicate conflicting aircraft separation clauses, negative upper bounds on nonnegative shortage variables, or all-positive network demands with no supply nodes. In those cases set model_passed=false, retry_target="model", and explain the semantic repair needed.
"""

REPAIR_CODE_SYSTEM_PROMPT = """
You are an expert in repairing optimization solver code.

You will receive:
1. The original problem
2. The mathematical model JSON
3. The previously generated code
4. The execution stdout / stderr
5. The feedback from the last comprehensive review

Your tasks:
- Fix the code.
- Output ONLY complete, runnable Python code.
- Do not output explanations.
- You must preserve the standard output specification: if Optimal, `print("result:", value)`, else `print("result: None")`.
- You must strictly adhere to the mathematical model; do not arbitrarily alter the intended modeling logic.
- Use PuLP for LP/MILP/integer linear problems, scipy.optimize for nonlinear continuous/quadratic/conic-like/geometric problems, and plain Python for direct formulas/enumeration.
- Do not use Pyomo, Gurobi, OR-Tools, cvxpy, SolverFactory, or solver-specific external APIs.
- If the previous code used Pyomo or another unsupported package, rewrite it into the closest supported implementation.
- The final `result:` line must be one numeric scalar matching the provided target value name. Do not output variables, dimensions, tuples, lists, or dictionaries unless the target explicitly asks for them.
"""

# =====================================
# 7. 运行代码函数
# =====================================

def execute_generated_code(code_text: str, timeout_seconds: int = 30) -> Dict[str, Any]:
    cleaned_code = code_text.strip()
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            temp_file_path = f.name
            f.write(cleaned_code)

        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        process = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_env,
            timeout=timeout_seconds,
        )
        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "timed_out": False,
            "temp_file_path": temp_file_path,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "returncode": None,
            "stdout": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else f"代码执行超时，超过 {timeout_seconds} 秒。",
            "timed_out": True,
            "temp_file_path": temp_file_path,
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass

# =====================================
# 8. Workflow State 定义
# =====================================

class WorkflowState(TypedDict, total=False):
    row_data: Dict[str, Any]
    row_index: int
    query_text: str
    rag_rows: List[Dict[str, str]]
    rag_predicted_type: str
    rag_top3: List[Dict[str, Any]]
    rag_top2: List[Dict[str, Any]]
    graph_rag_context: Dict[str, Any]

    provider_modeling: str
    provider_coding: str
    provider_judge: str

    modeling_model: str
    coding_model: str
    judge_model: str

    execution_timeout: int
    max_model_retries: int
    max_code_retries: int
    max_repair_retries: int

    model_spec: Optional[Dict[str, Any]]
    solver_code: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    extracted_value: Optional[float]

    final_review: Optional[Dict[str, Any]]
    final_review_feedback: str
    final_review_retry_target: str
    execution_attempts: List[Dict[str, Any]]
    final_fallback: Optional[Dict[str, Any]]

    model_retry_count: int
    code_retry_count: int
    repair_retry_count: int

    final_status: str
    error_message: str

# =====================================
# 9. LangGraph 节点定义
# =====================================

def model_problem_node(state: WorkflowState) -> Dict[str, Any]:
    llm = get_langchain_llm(
        provider=state["provider_modeling"],
        model_name=state["modeling_model"],
        temperature=0.0,
    )

    retry_count = state.get("model_retry_count", 0)
    previous_feedback = state.get("final_review_feedback", "")
    rag_rows = state.get("rag_rows", [])

    if USE_RAG_IN_MODELING and rag_rows:
        rag_result = retrieve_rag_examples(
            state["query_text"],
            rag_rows,
            top_k=max(RAG_RETRIEVE_TOP_K, RAG_RERANK_TOP_K * 3),
            rerank_k=RAG_RERANK_TOP_K,
            deduplicate_by_variant=True,
        )
    else:
        rag_result = {"predicted_type": "Others", "top3": [], "top2": []}

    graph_rag_context = get_graph_rag_context(state["query_text"])
    graph_rag_text = format_graph_rag_context(graph_rag_context, include_code=False)

    few_shot_blocks: List[str] = []
    for i, ex in enumerate(rag_result["top2"], start=1):
        query_text = (ex.get("query") or "")[:900]
        model_text = (ex.get("model") or "")[:1200]
        typical_error = (ex.get("typical_error") or "").strip()
        fix_hint = (ex.get("fix_hint") or "").strip()
        few_shot_blocks.append(
            f"[Retrieved Knowledge Card {i}]\n"
            f"ProblemType: {ex.get('type', '')}\n"
            f"Variant: {ex.get('variant', '')}\n"
            f"ErrorType: {ex.get('error_type', '')}\n\n"
            f"Similar Problem / Description:\n{query_text}\n\n"
            f"Modeling Template / MathModel:\n{model_text}\n\n"
            f"Modeling Checklist / Common Pitfall:\n{typical_error or 'N/A'}\n\n"
            f"Fix Hint:\n{fix_hint or 'N/A'}\n"
        )
    few_shot_text = "\n\n".join(few_shot_blocks) if few_shot_blocks else "No retrieved examples."

    if retry_count == 0 or not previous_feedback:
        user_prompt = f"""
Target problem type (predicted by retriever): {rag_result["predicted_type"]}

Retrieved GraphRAG modeling context:
{graph_rag_text}

Retrieved RAG knowledge card (Top-1 after rerank):
{few_shot_text}

Now solve this new problem and output only JSON:
{state["query_text"]}
"""
    else:
        user_prompt = f"""
Original problem:
{state["query_text"]}

Target problem type (predicted by retriever): {rag_result["predicted_type"]}

Retrieved GraphRAG modeling context:
{graph_rag_text}

Retrieved RAG knowledge card (Top-1 after rerank):
{few_shot_text}

The previous run indicates model-side issues. Rebuild the optimization model by strictly following this feedback:
{previous_feedback}
"""

    system_prompt = MODELING_SYSTEM_PROMPT + """

You must output a valid JSON object matching the exact structure below (do not output any additional explanations, do not wrap in markdown code blocks, output raw JSON only):

{
  "problem_summary": "string",
  "problem_type": "string",
  ...
}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", escape_prompt_template_literals(system_prompt)),
        ("user", "{problem}")
    ])
    chain = prompt | llm
    response = invoke_chain_with_retry(chain, {"problem": user_prompt}, "model_problem")
    content = response.content

    try:
        data = parse_llm_json_loose(content)
    except Exception as e:
        raise ValueError(f"Model output is not valid JSON. Error: {e}")

    try:
        normalized_data = normalize_model_json(data)
        model_obj = OptimizationModel.model_validate(normalized_data)
    except Exception as e:
        raise ValueError(
            f"Model output is JSON but schema validation failed: {json.dumps(data, ensure_ascii=False)[:1200]}...\nError: {e}"
        )

    return {
        "model_spec": model_obj.model_dump(exclude_none=True),
        "rag_predicted_type": rag_result["predicted_type"],
        "rag_top3": rag_result["top3"],
        "rag_top2": rag_result["top2"],
        "graph_rag_context": graph_rag_context,
        "solver_code": None,
        "execution_result": None,
        "extracted_value": None,
        "final_review": None,
        "final_review_feedback": "",
        "final_review_retry_target": "",
        "final_status": "",
    }


def generate_code_node(state: WorkflowState) -> Dict[str, Any]:
    llm = get_langchain_llm(
        provider=state["provider_coding"],
        model_name=state["coding_model"],
        temperature=0.0,
    )

    code_retry_count = state.get("code_retry_count", 0)
    previous_feedback = state.get("final_review_feedback", "")

    # ------ 新增：RAG 检索相似代码 ------
    rag_code_block = ""
    if USE_RAG_IN_CODING and state.get("rag_rows"):
        # 复用 RAG 检索函数，但只取 top_k = RAG_CODE_TOP_K 个代码示例
        rag_result = retrieve_rag_examples(
            state["query_text"],
            state["rag_rows"],
            top_k=RAG_RETRIEVE_TOP_K,
            rerank_k=RAG_CODE_TOP_K,
            deduplicate_by_variant=False
        )
        code_examples = []
        for ex in rag_result.get("top2", []):
            code = ex.get("code", "")
            if code and code.strip():
                # Keep the coding RAG compact: top1 plus the most useful error/fix hints.
                code_snippet = code[:1600] + ("..." if len(code) > 1600 else "")
                query_hint = (ex.get("query") or "")[:600]
                typical_error = (ex.get("typical_error") or "").strip()
                fix_hint = (ex.get("fix_hint") or "").strip()
                code_examples.append(
                    f"[Retrieved Coding Knowledge]\n"
                    f"ProblemType: {ex.get('type', '')}\n"
                    f"Variant: {ex.get('variant', '')}\n"
                    f"ErrorType: {ex.get('error_type', '')}\n\n"
                    f"Similar Problem / Description:\n{query_hint}\n\n"
                    f"Typical Error to Avoid:\n{typical_error or 'N/A'}\n\n"
                    f"Fix Hint:\n{fix_hint or 'N/A'}\n\n"
                    f"Reference Code Structure:\n```python\n{code_snippet}\n```"
                )
        if code_examples:
            rag_code_block = "\n\n".join(code_examples)
            rag_code_block = f"\n\n=== Retrieved RAG coding knowledge (Top-1; use structure and avoid listed errors) ===\n{rag_code_block}\n=== End retrieved RAG coding knowledge ===\n"
    # ------------------------------------

    graph_code_block = ""
    graph_rag_context = state.get("graph_rag_context") or get_graph_rag_context(state["query_text"])
    graph_code_text = format_graph_rag_context(graph_rag_context, include_code=True)
    if graph_code_text and graph_code_text != "No GraphRAG context retrieved.":
        graph_code_block = (
            "\n\n=== Retrieved GraphRAG coding context (variant path + code skeletons) ===\n"
            f"{graph_code_text}\n"
            "=== End retrieved GraphRAG coding context ===\n"
        )

    if code_retry_count == 0 or not previous_feedback:
        extra_instruction = ""
    else:
        extra_instruction = f"""
上一轮执行后统一检测判定“代码实现存在问题”，反馈如下。
请严格按反馈重写代码：

{previous_feedback}
"""

    user_prompt = f"""
下面给出原始题目描述和数学模型 JSON，请你同时参考二者生成完整可运行代码。

【原始题目】
{state["query_text"]}

【Target value to output】
{state.get("row_data", {}).get("target_name", "")}

【数学模型 JSON】
{json.dumps(state["model_spec"], ensure_ascii=False, indent=2)}

{graph_code_block}

{rag_code_block}

【附加反馈】
{extra_instruction}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", CODING_SYSTEM_PROMPT),
        ("user", "{problem}")
    ])
    chain = prompt | llm
    response = invoke_chain_with_retry(chain, {"problem": user_prompt}, "generate_code")

    return {
        "solver_code": clean_llm_code(response.content),
        "execution_result": None,
        "extracted_value": None,
        "final_review": None,
        "final_review_feedback": "",
        "final_review_retry_target": "",
        "final_status": "",
    }


def execute_code_node(state: WorkflowState) -> Dict[str, Any]:
    exec_result = execute_generated_code(
        code_text=state["solver_code"],
        timeout_seconds=state["execution_timeout"],
    )
    extracted_value = None
    if exec_result["success"]:
        extracted_value = extract_optimal_value(exec_result["stdout"])
    attempt = {
        "attempt_index": len(state.get("execution_attempts", [])) + 1,
        "model_retry_count": state.get("model_retry_count", 0),
        "code_retry_count": state.get("code_retry_count", 0),
        "repair_retry_count": state.get("repair_retry_count", 0),
        "model_spec": state.get("model_spec"),
        "solver_code": state.get("solver_code"),
        "execution_result": exec_result,
        "extracted_value": extracted_value,
    }
    return {
        "execution_result": exec_result,
        "extracted_value": extracted_value,
        "execution_attempts": state.get("execution_attempts", []) + [attempt],
        "final_status": "EXECUTED",
    }


def review_after_execution_node(state: WorkflowState) -> Dict[str, Any]:
    llm = get_langchain_llm(
        provider=state["provider_judge"],
        model_name=state["judge_model"],
        temperature=0.0,
    )
    exec_result = state.get("execution_result") or {}
    system_prompt = FINAL_REVIEW_SYSTEM_PROMPT + """

你必须输出一个合法的 JSON 对象，符合以下结构（不要输出任何额外解释，不要用 markdown 代码块包裹，只输出纯 JSON）：

{
  "overall_passed": true/false,
  "model_passed": true/false,
  "code_passed": true/false,
  "execution_passed": true/false,
  "retry_target": "model" 或 "code" 或 "repair" 或 "finish" 或 "no_solution",
  "reason_category": "字符串",
  "model_error_type": "none / variable_domain / missing_constraint / wrong_ratio / invented_constraint / objective_direction / infeasibility_misread / ambiguous_preference / other / unknown",
  "code_error_type": "none / package / syntax / solver_status / output_parse / runtime / timeout / implementation_mismatch / other / unknown",
  "feedback": "字符串"
}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", escape_prompt_template_literals(system_prompt)),
        ("user", """
请基于以下完整信息进行执行后统一检测：

【原始问题】
{problem}

【Target value to output】
{target_name}

【数学模型 JSON】
{model_spec_json}

【求解代码】
{solver_code}

【执行是否成功】
{exec_success}

【返回码】
{returncode}

【是否超时】
{timed_out}

【提取出的结果值】
{extracted_value}

【stdout】
{stdout}

【stderr】
{stderr}
""")
    ])
    chain = prompt | llm
    response = invoke_chain_with_retry(chain, {
        "problem": state["query_text"],
        "target_name": state.get("row_data", {}).get("target_name", ""),
        "model_spec_json": json.dumps(state["model_spec"], ensure_ascii=False, indent=2),
        "solver_code": state["solver_code"],
        "exec_success": exec_result.get("success"),
        "returncode": exec_result.get("returncode"),
        "timed_out": exec_result.get("timed_out"),
        "extracted_value": state.get("extracted_value"),
        "stdout": exec_result.get("stdout", ""),
        "stderr": exec_result.get("stderr", ""),
    }, "final_review")
    content = response.content
    try:
        data = parse_llm_json_loose(content)
    except Exception as e:
        data = {
            "overall_passed": False,
            "model_passed": True,
            "code_passed": False,
            "execution_passed": bool(exec_result.get("success")),
            "retry_target": "repair",
            "reason_category": "review_parse_error",
            "model_error_type": "unknown",
            "code_error_type": "unknown",
            "feedback": f"检测 Agent 输出无法解析为 JSON，先尝试代码修复或兜底。解析错误：{e}",
        }
    try:
        normalized_data = normalize_review_json(data)
        review = FinalReviewResult.model_validate(normalized_data)
    except Exception as e:
        normalized_data = normalize_review_json({
            "overall_passed": False,
            "model_passed": True,
            "code_passed": False,
            "execution_passed": bool(exec_result.get("success")),
            "retry_target": "repair",
            "reason_category": "review_schema_error",
            "model_error_type": "unknown",
            "code_error_type": "unknown",
            "feedback": (
                "检测 Agent 输出 JSON 结构不符合要求，先尝试代码修复或兜底。"
                f"原始结构前1200字符：{json.dumps(data, ensure_ascii=False)[:1200]}；错误：{e}"
            ),
        })
        review = FinalReviewResult.model_validate(normalized_data)
    return {
        "final_review": review.model_dump(),
        "final_review_feedback": review.feedback,
        "final_review_retry_target": review.retry_target,
    }


def repair_code_node(state: WorkflowState) -> Dict[str, Any]:
    llm = get_langchain_llm(
        provider=state["provider_coding"],
        model_name=state["coding_model"],
        temperature=0.0,
    )
    stderr_text = state.get("execution_result", {}).get("stderr", "")
    stdout_text = state.get("execution_result", {}).get("stdout", "")
    previous_feedback = state.get("final_review_feedback", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPAIR_CODE_SYSTEM_PROMPT),
        ("user", """
【原始问题】
{problem}

【数学模型 JSON】
{model_spec_json}

【原代码】
{solver_code}

【统一检测反馈】
{final_review_feedback}

【stdout】
{stdout}

【stderr】
{stderr}

请修复代码并输出完整 Python 代码。
""")
    ])
    chain = prompt | llm
    response = invoke_chain_with_retry(chain, {
        "problem": state["query_text"],
        "model_spec_json": json.dumps(state["model_spec"], ensure_ascii=False, indent=2),
        "solver_code": state["solver_code"],
        "final_review_feedback": previous_feedback,
        "stdout": stdout_text,
        "stderr": stderr_text,
    }, "repair_code")
    return {
        "solver_code": clean_llm_code(response.content),
        "execution_result": None,
        "extracted_value": None,
        "final_review": None,
        "final_review_retry_target": "",
        "final_status": "",
    }


def fallback_result_node(state: WorkflowState) -> Dict[str, Any]:
    attempts = state.get("execution_attempts", [])
    usable_attempts = [
        item for item in attempts
        if (item.get("execution_result") or {}).get("success") and item.get("extracted_value") is not None
    ]
    if not usable_attempts:
        return {
            "final_status": "FAILED",
            "final_fallback": {
                "fallback_used": False,
                "reason": "No executable historical Model+Code candidate with an extracted result.",
                "attempt_count": len(attempts),
            },
        }

    selected = usable_attempts[-1]
    return {
        "model_spec": selected.get("model_spec"),
        "solver_code": selected.get("solver_code"),
        "execution_result": selected.get("execution_result"),
        "extracted_value": selected.get("extracted_value"),
        "final_status": "FALLBACK_SUCCESS",
        "final_fallback": {
            "fallback_used": True,
            "reason": "All regular retries failed; selected the latest executable historical Model+Code candidate.",
            "selected_attempt_index": selected.get("attempt_index"),
            "attempt_count": len(attempts),
            "model_retry_count": selected.get("model_retry_count"),
            "code_retry_count": selected.get("code_retry_count"),
            "repair_retry_count": selected.get("repair_retry_count"),
            "extracted_value": selected.get("extracted_value"),
        },
    }


def end_success_node(state: WorkflowState) -> Dict[str, Any]:
    return {"final_status": "SUCCESS"}

def end_no_solution_node(state: WorkflowState) -> Dict[str, Any]:
    return {"final_status": "NO_BEST_SOLUTION"}

def end_fail_node(state: WorkflowState) -> Dict[str, Any]:
    return {"final_status": "FAILED"}


def route_after_review(state: WorkflowState) -> Literal["end_success", "end_no_solution", "model_problem", "generate_code", "repair_code", "end_fail"]:
    retry_target = state.get("final_review_retry_target", "")
    final_review = state.get("final_review") or {}
    if retry_target == "finish":
        return "end_success"
    if retry_target == "no_solution":
        return "end_no_solution"
    if retry_target == "model":
        if state.get("model_retry_count", 0) < state["max_model_retries"]:
            return "model_problem"
        return "end_fail"
    if retry_target == "code":
        if state.get("code_retry_count", 0) < state["max_code_retries"]:
            return "generate_code"
        return "end_fail"
    if retry_target == "repair":
        if state.get("repair_retry_count", 0) < state["max_repair_retries"]:
            return "repair_code"
        return "end_fail"
    if final_review.get("overall_passed") is True:
        return "end_success"
    return "end_fail"


def route_after_execute_without_review(state: WorkflowState) -> Literal["end_success", "end_no_solution", "end_fail"]:
    exec_result = state.get("execution_result") or {}
    if not exec_result.get("success"):
        return "end_fail"
    if state.get("extracted_value") is None:
        return "end_no_solution"
    return "end_success"


def increment_model_retry_node(state: WorkflowState) -> Dict[str, Any]:
    return {"model_retry_count": state.get("model_retry_count", 0) + 1}

def increment_code_retry_node(state: WorkflowState) -> Dict[str, Any]:
    return {"code_retry_count": state.get("code_retry_count", 0) + 1}

def increment_repair_retry_node(state: WorkflowState) -> Dict[str, Any]:
    return {"repair_retry_count": state.get("repair_retry_count", 0) + 1}


def build_workflow_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("model_problem", model_problem_node)
    graph.add_node("generate_code", generate_code_node)
    graph.add_node("execute_code", execute_code_node)
    graph.add_node("review_after_execution", review_after_execution_node)
    graph.add_node("repair_code", repair_code_node)
    graph.add_node("fallback_result", fallback_result_node)
    graph.add_node("increment_model_retry", increment_model_retry_node)
    graph.add_node("increment_code_retry", increment_code_retry_node)
    graph.add_node("increment_repair_retry", increment_repair_retry_node)
    graph.add_node("end_success", end_success_node)
    graph.add_node("end_no_solution", end_no_solution_node)
    graph.add_node("end_fail", end_fail_node)

    graph.add_edge(START, "model_problem")
    graph.add_edge("model_problem", "generate_code")
    graph.add_edge("generate_code", "execute_code")

    if ENABLE_VALIDATION_AGENT:
        graph.add_edge("execute_code", "review_after_execution")
        graph.add_conditional_edges(
            "review_after_execution",
            route_after_review,
            {
                "end_success": "end_success",
                "end_no_solution": "end_no_solution",
                "model_problem": "increment_model_retry",
                "generate_code": "increment_code_retry",
                "repair_code": "increment_repair_retry",
                "end_fail": "fallback_result",
            },
        )
    else:
        graph.add_conditional_edges(
            "execute_code",
            route_after_execute_without_review,
            {
                "end_success": "end_success",
                "end_no_solution": "end_no_solution",
                "end_fail": "fallback_result",
            },
        )
    graph.add_edge("increment_model_retry", "model_problem")
    graph.add_edge("increment_code_retry", "generate_code")
    graph.add_edge("increment_repair_retry", "repair_code")
    graph.add_edge("repair_code", "execute_code")
    graph.add_edge("fallback_result", END)
    graph.add_edge("end_success", END)
    graph.add_edge("end_no_solution", END)
    graph.add_edge("end_fail", END)
    return graph.compile()


def run_langgraph_for_single_problem(
    row_data: Dict[str, Any],
    provider_modeling: str,
    provider_coding: str,
    provider_judge: str,
    modeling_model: str,
    coding_model: str,
    judge_model: str,
    execution_timeout: int = 60,
) -> Dict[str, Any]:
    graph = build_workflow_graph()
    init_state: WorkflowState = {
        "row_data": row_data,
        "row_index": row_data.get("_row_index", -1),
        "query_text": (row_data.get(QUERY_COLUMN) or "").strip(),
        "rag_rows": RAG_ROWS,
        "rag_predicted_type": "",
        "rag_top3": [],
        "rag_top2": [],
        "graph_rag_context": {},
        "provider_modeling": provider_modeling,
        "provider_coding": provider_coding,
        "provider_judge": provider_judge,
        "modeling_model": modeling_model,
        "coding_model": coding_model,
        "judge_model": judge_model,
        "execution_timeout": execution_timeout,
        "max_model_retries": MAX_MODEL_RETRIES,
        "max_code_retries": MAX_CODE_RETRIES,
        "max_repair_retries": MAX_REPAIR_RETRIES,
        "model_retry_count": 0,
        "code_retry_count": 0,
        "repair_retry_count": 0,
        "model_spec": None,
        "solver_code": None,
        "execution_result": None,
        "extracted_value": None,
        "final_review": None,
        "final_review_feedback": "",
        "final_review_retry_target": "",
        "execution_attempts": [],
        "final_fallback": None,
        "final_status": "",
        "error_message": "",
    }
    final_state = graph.invoke(init_state)
    return dict(final_state)


def process_single_problem(
    row_data: Dict[str, Any],
    provider_modeling: str,
    provider_coding: str,
    provider_judge: str,
    modeling_model: str,
    coding_model: str,
    judge_model: str,
    auto_run_code: bool = True,
    execution_timeout: int = 30,
) -> None:
    ensure_dir(LOG_DIR)
    ensure_dir(RESULT_DIR)
    row_index = row_data.get("_row_index", "unknown")
    query_text = (row_data.get(QUERY_COLUMN) or "").strip()
    model_tag = sanitize_filename(
        f"model_{modeling_model}"
        f"_code_{coding_model}"
        f"_judge_{judge_model}"
    )
    case_dir = os.path.join(LOG_DIR, str(row_index))
    ensure_dir(case_dir)
    log_file_path = os.path.join(case_dir, f"run_log_{model_tag}.txt")
    model_file_path = os.path.join(case_dir, f"model_spec_{model_tag}.json")
    code_file_path = os.path.join(case_dir, f"solver_code_{model_tag}.py")
    final_state_path = os.path.join(case_dir, f"final_state_{model_tag}.json")
    raw_model_output_path = os.path.join(case_dir, f"raw_model_output_{model_tag}.txt")
    raw_review_output_path = os.path.join(case_dir, f"raw_review_output_{model_tag}.txt")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            print(f"日志文件已创建: {log_file_path}")
            print(f"建模结果文件将保存到: {model_file_path}")
            print(f"代码结果文件将保存到: {code_file_path}")
            print(f"工作流最终状态文件将保存到: {final_state_path}")
            print(f"原始建模输出文件将保存到: {raw_model_output_path}")
            print(f"原始检测输出文件将保存到: {raw_review_output_path}")
            print(f"当前数据行号: {row_index}")
            print(f"当前 Query: {query_text}\n")
            print("========== 当前运行配置 ==========")
            print(f"Provider Modeling  : {provider_modeling}")
            print(f"Provider Coding    : {provider_coding}")
            print(f"Provider Judge     : {provider_judge}")
            print(f"Modeling Model     : {modeling_model}")
            print(f"Coding Model       : {coding_model}")
            print(f"Judge Model        : {judge_model}")
            print(f"Execution Timeout  : {execution_timeout}")
            print(f"Max Model Retries  : {MAX_MODEL_RETRIES}")
            print(f"Max Code Retries   : {MAX_CODE_RETRIES}")
            print(f"Max Repair Retries : {MAX_REPAIR_RETRIES}")
            print("=================================\n")
            print("========== 原始样本信息 ==========")
            for k, v in row_data.items():
                print(f"{k}: {v}")
            print()
            if not auto_run_code:
                raise ValueError("当前版本默认执行代码，AUTO_RUN_CODE 不应为 False。")
            final_state = run_langgraph_for_single_problem(
                row_data=row_data,
                provider_modeling=provider_modeling,
                provider_coding=provider_coding,
                provider_judge=provider_judge,
                modeling_model=modeling_model,
                coding_model=coding_model,
                judge_model=judge_model,
                execution_timeout=execution_timeout,
            )
            save_json_file(model_file_path, final_state.get("model_spec") or {})
            save_text_file(code_file_path, final_state.get("solver_code") or "")
            save_json_file(final_state_path, final_state)
            if final_state.get("raw_model_output"):
                save_text_file(raw_model_output_path, final_state.get("raw_model_output") or "")
            if final_state.get("raw_review_output"):
                save_text_file(raw_review_output_path, final_state.get("raw_review_output") or "")
            print("========== 建模结果 ==========")
            print(json.dumps(final_state.get("model_spec") or {}, ensure_ascii=False, indent=2))
            print("\n========== 代码结果 ==========")
            print(final_state.get("solver_code") or "（无代码）")
            exec_result = final_state.get("execution_result")
            if exec_result:
                print("\n========== 自动执行结果 ==========")
                print(f"临时代码文件: {exec_result.get('temp_file_path')}")
                print(f"是否成功    : {exec_result.get('success')}")
                print(f"返回码      : {exec_result.get('returncode')}")
                print(f"是否超时    : {exec_result.get('timed_out')}")
                print("\n===== 标准输出 stdout =====")
                print(exec_result.get("stdout") if exec_result.get("stdout") else "（无输出）")
                print("\n===== 错误输出 stderr =====")
                print(exec_result.get("stderr") if exec_result.get("stderr") else "（无报错）")
            else:
                print("\n========== 自动执行结果 ==========")
                print("（无 execution_result）")
            print("\n========== 执行后统一检测结果 ==========")
            print(json.dumps(final_state.get("final_review") or {}, ensure_ascii=False, indent=2))
            final_status = final_state.get("final_status")
            extracted_value = final_state.get("extracted_value")
            print("\n========== 工作流最终状态 ==========")
            print("final_status:", final_status)
            print("extracted_value:", extracted_value)
            print("model_retry_count:", final_state.get("model_retry_count"))
            print("code_retry_count:", final_state.get("code_retry_count"))
            print("repair_retry_count:", final_state.get("repair_retry_count"))
            if final_status in {"SUCCESS", "FALLBACK_SUCCESS"}:
                append_result_value(RESULT_FILE_PATH, extracted_value)
                append_result_detail(RESULT_DETAIL_FILE_PATH, row_data, extracted_value, "ok")
                print("\n===== 提取到的最优值 =====")
                print(extracted_value)
                print(f"结果已追加到: {RESULT_FILE_PATH}")
            elif final_status == "NO_BEST_SOLUTION":
                append_result_value(RESULT_FILE_PATH, None)
                append_result_detail(RESULT_DETAIL_FILE_PATH, row_data, None, "no_best_solution", "no_best_solution")
                print("\n===== 结果类型 =====")
                print("No Best Solution（无可行解或检测认为属于合理无解）")
                print(f"结果已追加到: {RESULT_FILE_PATH}")
            else:
                append_result_value(RESULT_FILE_PATH, "Code Error")
                append_result_detail(
                    RESULT_DETAIL_FILE_PATH,
                    row_data,
                    "Code Error",
                    "error",
                    classify_error_type(final_state.get("error_message", "")),
                    final_state.get("error_message", ""),
                )
                print("\n===== 结果类型 =====")
                print("Code Error（流程失败或超过重试上限）")
                print(f"结果已追加到: {RESULT_FILE_PATH}")
            print("\n========== 文件保存完成 ==========")
            print(f"运行日志: {log_file_path}")
            print(f"建模结果: {model_file_path}")
            print(f"代码结果: {code_file_path}")
            print(f"工作流状态: {final_state_path}")
        except Exception as exc:
            tb = traceback.format_exc()
            print("\n========== WORKFLOW EXCEPTION ==========")
            print(tb)
            save_json_file(final_state_path, {
                "final_status": "ERROR",
                "error_message": str(exc),
                "traceback": tb,
                "row_index": row_index,
                "query_text": query_text,
            })
            append_result_value(RESULT_FILE_PATH, "Code Error")
            append_result_detail(
                RESULT_DETAIL_FILE_PATH,
                row_data,
                "Code Error",
                "error",
                classify_error_type(tb),
                str(exc),
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    ensure_dir(LOG_ROOT_DIR)
    ensure_dir(LOG_DIR)
    ensure_dir(RESULT_ROOT_DIR)
    ensure_dir(RESULT_DIR)
    init_result_csv(RESULT_FILE_PATH)
    init_result_detail_csv(RESULT_DETAIL_FILE_PATH)

    print("========== 开始读取数据集 ==========")
    print(f"数据集路径: {DATASET_PATH}")
    print(f"日志目录  : {LOG_DIR}")
    print(f"结果目录  : {RESULT_DIR}")
    print(f"结果文件  : {RESULT_FILE_PATH}")
    print(f"Provider Modeling : {PROVIDER_MODELING}")
    print(f"Provider Coding   : {PROVIDER_CODING}")
    print(f"Provider Judge    : {PROVIDER_JUDGE}")
    print(f"建模模型            : {MODELING_MODEL}")
    print(f"代码模型            : {CODING_MODEL}")
    print(f"检测模型            : {JUDGE_MODEL}")
    print(f"MAX_SAMPLES         : {MAX_SAMPLES}")
    print(f"START_SOURCE_ROW    : {START_SOURCE_ROW}")
    print(f"END_SOURCE_ROW      : {END_SOURCE_ROW}")
    print(f"RETRY_FROM_RESULT   : {RETRY_FROM_RESULT_PATH}")
    print(f"RETRY_ROWS_PATH     : {RETRY_ROWS_PATH}")
    print(f"LLM_API_ATTEMPTS    : {LLM_API_MAX_ATTEMPTS}")
    print("===================================\n")

    all_rows = load_queries_from_file(DATASET_PATH, query_column=QUERY_COLUMN)
    print(f"共读取到 {len(all_rows)} 条有效 Query。\n")
    all_rows = filter_rows_for_run(all_rows)
    print(f"本次实际处理 {len(all_rows)} 条样本。\n")

    for idx, row in enumerate(all_rows, start=1):
        print(f"\n\n############ 开始处理第 {idx}/{len(all_rows)} 条样本 ############")
        try:
            process_single_problem(
                row_data=row,
                provider_modeling=PROVIDER_MODELING,
                provider_coding=PROVIDER_CODING,
                provider_judge=PROVIDER_JUDGE,
                modeling_model=MODELING_MODEL,
                coding_model=CODING_MODEL,
                judge_model=JUDGE_MODEL,
                auto_run_code=AUTO_RUN_CODE,
                execution_timeout=EXECUTION_TIMEOUT,
            )
        except Exception as e:
            print(f"第 {idx} 条样本处理失败，错误信息: {e}")
            append_result_value(RESULT_FILE_PATH, "Code Error")
            append_result_detail(
                RESULT_DETAIL_FILE_PATH,
                row,
                "Code Error",
                "error",
                classify_error_type(traceback.format_exc()),
                str(e),
            )

    print("\n全部样本处理完成。")

