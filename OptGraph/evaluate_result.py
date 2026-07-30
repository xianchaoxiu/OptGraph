import csv
import os
from typing import List, Optional, Dict, Any

# =====================================
# 0. 路径配置
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
RESULT_ROOT_DIR = os.path.join(BASE_DIR, "RESULT")

# 在这里手动指定要评测的结果CSV文件路径及对应的数据集名称
# 每个元素是一个元组: (result_file_path, dataset_name)
# dataset_name 应与数据集文件名（不含扩展名）一致，例如 "NL4OPT_NEW"
RESULT_FILES = [
    (r"RESULT/IndustryOR_deepseek_deepseek-chat_20260410_223119/IndustryOR_deepseek_deepseek-chat_result_20260410_223119.csv", "IndustryOR"),
    (r"RESULT/IndustryOR_gemini_gemini-3.1-pro-preview_20260410_222532/IndustryOR_gemini_gemini-3.1-pro-preview_result_20260410_222532.csv", "IndustryOR"),
    (r"RESULT/IndustryOR_model_gemini-3.1-pro-preview__judge_gpt-4.1_20260410_220855/IndustryOR_model_gemini-3.1-pro-preview__judge_gpt-4.1_result_20260410_220855.csv", "IndustryOR"),
    (r"RESULT/NL4OPT_deepseek_deepseek-chat_20260410_223059/NL4OPT_deepseek_deepseek-chat_result_20260410_223059.csv", "NL4OPT"),
    (r"RESULT/NL4OPT_gemini_gemini-3.1-pro-preview_20260410_222451/NL4OPT_gemini_gemini-3.1-pro-preview_result_20260410_222451.csv", "NL4OPT"),
    (r"RESULT/NL4OPT_model_gemini-3.1-pro-preview__judge_gpt-4.1_20260410_214102/NL4OPT_model_gemini-3.1-pro-preview__judge_gpt-4.1_result_20260410_214102.csv", "NL4OPT"),

]

QUERY_COLUMN = "Query"
LABEL_COLUMN = "Label"

# 评测结果输出文件
SUMMARY_OUTPUT_PATH = os.path.join(RESULT_ROOT_DIR, "evaluation_summary.csv")

# 容差设置
EXACT_TOL = 1e-6
REL_TOL_1 = 0.01   # 1%
REL_TOL_5 = 0.05   # 5%

# 预测失败关键词
NO_BEST_SOLUTION_TOKENS = {
    "no best solution",
    "no solution",
    "no optimal solution",
    "infeasible",
    "unbounded",
    "infeasible or unbounded",
}

CODE_ERROR_TOKENS = {
    "code error",
    "execution error",
    "runtime error",
    "traceback",
    "syntax error",
    "nameerror",
    "typeerror",
    "valueerror",
    "zerodivisionerror",
    "attributeerror",
    "importerror",
    "indexerror",
    "keyerror",
    "oserror",
    "exception",
}

NULL_TOKENS = {
    "",
    "none",
    "nan",
    "null",
    "n/a",
    "na",
}

# =====================================
# 1. 工具函数
# =====================================

def safe_float(x) -> Optional[float]:
    """将字符串转换为 float，无法转换时返回 None"""
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def normalize_text(x: Any) -> str:
    """统一转小写并去掉首尾空白"""
    if x is None:
        return ""
    return str(x).strip().lower()


def classify_prediction_text(raw_text: Any) -> str:
    """
    将预测原始文本分类为：
    - numeric
    - no_best_solution
    - code_error
    - null
    - invalid_text
    """
    text = normalize_text(raw_text)

    if text in NULL_TOKENS:
        return "null"

    # 先判断是否是数值
    if safe_float(text) is not None:
        return "numeric"

    # 再判断特殊错误类型
    for token in NO_BEST_SOLUTION_TOKENS:
        if token in text:
            return "no_best_solution"

    for token in CODE_ERROR_TOKENS:
        if token in text:
            return "code_error"

    return "invalid_text"


def read_labels(dataset_path: str, label_column: str = "Label") -> List[Optional[float]]:
    """读取数据集中的标签列，支持多种编码"""
    encodings = ['utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'utf-8']
    last_error = None

    for enc in encodings:
        try:
            with open(dataset_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"数据集没有表头: {dataset_path}")
                if label_column not in reader.fieldnames:
                    raise ValueError(f"数据集 {dataset_path} 中未找到列 {label_column}")

                labels = []
                for row in reader:
                    labels.append(safe_float(row.get(label_column)))
                return labels
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise ValueError(f"无法用尝试的编码读取文件: {dataset_path}，最后错误: {last_error}")


def read_predictions(result_path: str) -> List[Dict[str, Any]]:
    """
    读取结果文件中的预测列（默认第1列），保留：
    - raw: 原始字符串
    - value: 数值（若可转）
    - type: 分类
    """
    encodings = ['utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'utf-8']
    last_error = None

    for enc in encodings:
        try:
            with open(result_path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    raise ValueError(f"结果文件为空: {result_path}")

                preds: List[Dict[str, Any]] = []
                for row in reader:
                    raw = ""
                    if row and len(row) > 0:
                        raw = str(row[0]).strip()

                    pred_type = classify_prediction_text(raw)
                    pred_value = safe_float(raw)

                    preds.append({
                        "raw": raw,
                        "value": pred_value,
                        "type": pred_type,
                    })
                return preds
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise ValueError(f"无法用尝试的编码读取文件: {result_path}，最后错误: {last_error}")


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def evaluate_one_dataset(
    dataset_name: str,
    labels: List[Optional[float]],
    preds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """评估单个数据集，返回各种指标和错误行号列表"""
    total_samples = len(labels)
    total_preds = len(preds)
    compare_len = min(total_samples, total_preds)

    exact_match_count = 0
    rel_1_count = 0
    rel_5_count = 0

    # valid_count: label 和 pred 都是数值的样本数
    valid_count = 0

    # 应评测样本数：label 有效的样本都应该进入分母
    expected_count = sum(1 for y in labels if y is not None)

    # 错误行号分类
    error_rows: List[int] = []
    numeric_mismatch_rows: List[int] = []
    invalid_pred_rows: List[int] = []
    missing_pred_rows: List[int] = []
    code_error_rows: List[int] = []
    no_best_solution_rows: List[int] = []

    abs_errors: List[float] = []
    rel_errors: List[float] = []

    # 先处理有预测覆盖到的部分
    for i in range(compare_len):
        row_num = i + 2  # CSV 中数据行号从 2 开始
        y_true = labels[i]
        pred_info = preds[i]
        y_pred = pred_info["value"]
        pred_type = pred_info["type"]

        # 标签无效则跳过，不纳入 expected_count（前面已经排除）
        if y_true is None:
            continue

        # 预测不是数值，直接记为错误
        if pred_type != "numeric" or y_pred is None:
            error_rows.append(row_num)
            invalid_pred_rows.append(row_num)

            if pred_type == "code_error":
                code_error_rows.append(row_num)
            elif pred_type == "no_best_solution":
                no_best_solution_rows.append(row_num)

            continue

        # 到这里，说明 y_true 和 y_pred 都是有效数字
        valid_count += 1
        abs_err = abs(y_true - y_pred)
        rel_err = abs_err / max(abs(y_true), 1e-12)

        abs_errors.append(abs_err)
        rel_errors.append(rel_err)

        # 精确匹配
        if abs_err <= EXACT_TOL:
            exact_match_count += 1
        else:
            error_rows.append(row_num)
            numeric_mismatch_rows.append(row_num)

        if rel_err <= REL_TOL_1:
            rel_1_count += 1
        if rel_err <= REL_TOL_5:
            rel_5_count += 1

    # 结果文件比标签短：缺失预测也算错误
    if total_preds < total_samples:
        for i in range(total_preds, total_samples):
            if labels[i] is not None:
                row_num = i + 2
                error_rows.append(row_num)
                missing_pred_rows.append(row_num)

    exact_error_count = expected_count - exact_match_count
    rel_1_error_count = expected_count - rel_1_count
    rel_5_error_count = expected_count - rel_5_count

    result = {
        "dataset_name": dataset_name,
        "label_count": total_samples,
        "pred_count": total_preds,
        "compared_count": compare_len,
        "expected_count": expected_count,
        "valid_count": valid_count,

        "exact_match_count": exact_match_count,
        "exact_error_count": exact_error_count,
        "rel_1_count": rel_1_count,
        "rel_1_error_count": rel_1_error_count,
        "rel_5_count": rel_5_count,
        "rel_5_error_count": rel_5_error_count,

        "exact_match_accuracy": (exact_match_count / expected_count) if expected_count > 0 else None,
        "rel_1_accuracy": (rel_1_count / expected_count) if expected_count > 0 else None,
        "rel_5_accuracy": (rel_5_count / expected_count) if expected_count > 0 else None,

        "mae": mean(abs_errors),
        "mre": mean(rel_errors),

        "error_rows": sorted(set(error_rows)),
        "numeric_mismatch_rows": sorted(set(numeric_mismatch_rows)),
        "invalid_pred_rows": sorted(set(invalid_pred_rows)),
        "missing_pred_rows": sorted(set(missing_pred_rows)),
        "code_error_rows": sorted(set(code_error_rows)),
        "no_best_solution_rows": sorted(set(no_best_solution_rows)),
    }
    return result


def print_row_list(title: str, rows: List[int], max_show: Optional[int] = None) -> None:
    """打印行号列表"""
    if not rows:
        return

    if max_show is not None and len(rows) > max_show:
        shown = rows[:max_show]
        print(f"  {title:<24}: {shown} ... 共 {len(rows)} 行")
    else:
        print(f"  {title:<24}: {rows}")


def print_dataset_result(result: Dict[str, Any]) -> None:
    """打印单个数据集的评测结果（包含错误行号分类）"""
    print(f"数据集: {result['dataset_name']}")
    print(f"  Label 数量                : {result['label_count']}")
    print(f"  Prediction 数量           : {result['pred_count']}")
    print(f"  实际比较数量              : {result['compared_count']}")
    print(f"  应评测样本数              : {result['expected_count']}")
    print(f"  数值有效预测数量          : {result['valid_count']}")

    if result["expected_count"] == 0:
        print("  没有可评测标签，无法计算准确率。")
        print()
        return

    print(f"  错误数 (Exact Match)      : {result['exact_error_count']}")
    print(f"  Exact Match 准确率        : {result['exact_match_accuracy']:.4f}")
    print(f"  1% 相对误差准确率         : {result['rel_1_accuracy']:.4f}")
    print(f"  5% 相对误差准确率         : {result['rel_5_accuracy']:.4f}")

    if result["mae"] is not None:
        print(f"  Mean Absolute Error       : {result['mae']:.6f}")
    else:
        print("  Mean Absolute Error       : None")

    if result["mre"] is not None:
        print(f"  Mean Relative Error       : {result['mre']:.6f}")
    else:
        print("  Mean Relative Error       : None")

    if result["error_rows"]:
        print_row_list("所有错误行号", result["error_rows"])
        print_row_list("数值不匹配行号", result["numeric_mismatch_rows"])
        print_row_list("无效预测行号", result["invalid_pred_rows"])
        print_row_list("缺失预测行号", result["missing_pred_rows"])
        print_row_list("Code Error 行号", result["code_error_rows"])
        print_row_list("No Best Solution 行号", result["no_best_solution_rows"])
    else:
        print("  无错误行")

    print()


def save_summary_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """保存汇总结果到 CSV 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "dataset_name",
        "label_count",
        "pred_count",
        "compared_count",
        "expected_count",
        "valid_count",
        "exact_match_count",
        "exact_error_count",
        "rel_1_count",
        "rel_1_error_count",
        "rel_5_count",
        "rel_5_error_count",
        "exact_match_accuracy",
        "rel_1_accuracy",
        "rel_5_accuracy",
        "mae",
        "mre",
        "error_rows",
        "numeric_mismatch_rows",
        "invalid_pred_rows",
        "missing_pred_rows",
        "code_error_rows",
        "no_best_solution_rows",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            out_row = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, list):
                    out_row[k] = ";".join(map(str, v))
                else:
                    out_row[k] = v
            writer.writerow(out_row)


def find_dataset_path(dataset_name: str) -> Optional[str]:
    """根据数据集名查找对应的 CSV 文件路径"""
    dataset_path = os.path.join(DATASET_DIR, f"{dataset_name}.csv")
    if os.path.exists(dataset_path):
        return dataset_path
    return None


# =====================================
# 2. 主程序
# =====================================

if __name__ == "__main__":
    if not RESULT_FILES:
        print("请在 RESULT_FILES 中填写至少一个要评测的结果文件路径及对应的数据集名称。")
        raise SystemExit(0)

    all_results: List[Dict[str, Any]] = []

    print("========== 开始评测 ==========\n")

    for result_path, dataset_name in RESULT_FILES:
        print(f"正在评测: {result_path}")

        # 支持相对路径
        abs_result_path = result_path
        if not os.path.isabs(abs_result_path):
            abs_result_path = os.path.join(BASE_DIR, result_path)

        if not os.path.exists(abs_result_path):
            print(f"  ❌ 结果文件不存在，跳过。\n")
            continue

        dataset_path = find_dataset_path(dataset_name)
        if dataset_path is None:
            print(f"  ❌ 未找到对应的数据集文件（{dataset_name}.csv），跳过。\n")
            continue

        try:
            labels = read_labels(dataset_path, LABEL_COLUMN)
            preds = read_predictions(abs_result_path)

            result = evaluate_one_dataset(dataset_name, labels, preds)
            all_results.append(result)

            print_dataset_result(result)

        except Exception as e:
            print(f"  ❌ 评测失败: {e}\n")

    if not all_results:
        print("没有成功评测的数据集。")
    else:
        print("========== 汇总 ==========\n")
        for res in all_results:
            print(f"数据集: {res['dataset_name']}")
            if res["expected_count"] > 0:
                print(f"  错误数 (Exact Match) : {res['exact_error_count']}")
                print(f"  Exact Match 准确率   : {res['exact_match_accuracy']:.4f}")
                print(f"  MAE: {res['mae']:.6f}" if res['mae'] is not None else "  MAE: None")
                print(f"  MRE: {res['mre']:.6f}" if res['mre'] is not None else "  MRE: None")
                print_row_list("所有错误行号", res["error_rows"])
                print_row_list("数值不匹配行号", res["numeric_mismatch_rows"])
                print_row_list("无效预测行号", res["invalid_pred_rows"])
                print_row_list("缺失预测行号", res["missing_pred_rows"])
                print_row_list("Code Error 行号", res["code_error_rows"])
                print_row_list("No Best Solution 行号", res["no_best_solution_rows"])
            else:
                print("  无有效标签可评测")
            print()

        save_summary_csv(all_results, SUMMARY_OUTPUT_PATH)
        print(f"汇总结果已保存到: {SUMMARY_OUTPUT_PATH}")
