"""
SF-12 / SF-36 在线评分工具
Flask Web 应用 - Vercel 兼容版
"""

import os
import io
import uuid
import hashlib
import base64
import tempfile
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for
from scoring import score_sf12, score_sf36
from questions import get_question_table

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# === 内存结果存储（Vercel serverless 兼容） ===
# 格式: { token: {"data": bytes, "filename": str, "expires": float} }
_result_store = {}
_store_lock = threading.Lock()

def _cleanup_store():
    """清理过期结果"""
    now = time.time()
    with _store_lock:
        expired = [k for k, v in _result_store.items() if v["expires"] < now]
        for k in expired:
            del _result_store[k]

def store_result(excel_bytes: bytes, filename: str, ttl: int = 600) -> str:
    """存储结果，返回下载 token"""
    _cleanup_store()
    token = uuid.uuid4().hex
    with _store_lock:
        _result_store[token] = {
            "data": excel_bytes,
            "filename": filename,
            "expires": time.time() + ttl,
        }
    return token

def get_result(token: str):
    """获取存储的结果"""
    _cleanup_store()
    with _store_lock:
        return _result_store.pop(token, None)

# === 激活码系统 ===
ACTIVATION_CODES = {
    hashlib.sha256("DEMO2026".encode()).hexdigest(): {"used": False, "max_uses": 1000},
    hashlib.sha256("TRIAL-SF".encode()).hexdigest(): {"used": False, "max_uses": 500},
}

def generate_code(prefix="SF"):
    code = f"{prefix}-{uuid.uuid4().hex[:8].upper()}"
    ACTIVATION_CODES[hashlib.sha256(code.encode()).hexdigest()] = {
        "used": False, "max_uses": 1
    }
    return code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tool/<scale_type>")
def tool(scale_type):
    if scale_type not in ("sf12", "sf36"):
        return redirect(url_for("index"))
    questions = get_question_table(scale_type)
    title = "SF-12" if scale_type == "sf12" else "SF-36"
    return render_template("calculator.html", scale_type=scale_type, title=title, questions=questions)


@app.route("/api/preview", methods=["POST"])
def preview():
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400
    file = request.files["file"]
    scale_type = request.form.get("scale_type", "sf12")
    if file.filename == "":
        return jsonify({"error": "请选择文件"}), 400

    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "不支持的文件格式，请上传 CSV 或 Excel (.xlsx/.xls)"}), 400
    except Exception as e:
        return jsonify({"error": f"文件读取失败: {str(e)}"}), 400

    questions = get_question_table(scale_type)
    expected_cols = len(questions)

    if len(df.columns) < expected_cols:
        return jsonify({
            "error": f"数据列数不足。需要 {expected_cols} 列（{scale_type.upper()}量表），您的数据只有 {len(df.columns)} 列。"
        }), 400

    mapping = []
    for i, q in enumerate(questions):
        col_name = df.columns[i] if i < len(df.columns) else ""
        preview_values = df.iloc[:3, i].tolist() if i < len(df.columns) else []
        preview_values = [str(v) if pd.notna(v) else "" for v in preview_values]
        mapping.append({
            "position": i + 1,
            "col_name": str(col_name),
            "question_id": q["id"],
            "domain": q["domain"],
            "question_text": q["text"],
            "options": q.get("options", []),
            "values": q["values"],
            "note": q.get("note", ""),
            "preview_values": preview_values,
        })

    return jsonify({
        "success": True,
        "scale_type": scale_type,
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "expected_cols": expected_cols,
        "mapping": mapping,
        "col_names": [str(c) for c in df.columns[:expected_cols]],
    })


@app.route("/api/calculate", methods=["POST"])
def calculate():
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    scale_type = request.form.get("scale_type", "sf12")
    code = request.form.get("code", "")

    code_hash = hashlib.sha256(code.strip().encode()).hexdigest()
    if code_hash not in ACTIVATION_CODES:
        return jsonify({"error": "激活码无效，请输入正确的激活码"}), 403

    code_info = ACTIVATION_CODES[code_hash]
    if code_info["used"] and code_info["max_uses"] <= 0:
        return jsonify({"error": "该激活码已被使用"}), 403

    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file)
        elif ext in (".xls", ".xlsx"):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "不支持的文件格式"}), 400
    except Exception as e:
        return jsonify({"error": f"文件读取失败: {str(e)}"}), 400

    try:
        result = score_sf12(df) if scale_type == "sf12" else score_sf36(df)
    except Exception as e:
        return jsonify({"error": f"计分过程出错: {str(e)}"}), 500

    code_info["used"] = True
    code_info["max_uses"] -= 1

    # 合并原始数据 + 得分
    combined = pd.concat([df.iloc[:, :len(df.columns)], result.reset_index(drop=True)], axis=1)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name=f"{scale_type.upper()}_原始数据+得分", index=False)
        result.to_excel(writer, sheet_name=f"{scale_type.upper()}_仅得分", index=False)
    excel_bytes = output.getvalue()

    # 存储到内存
    filename = f"{scale_type.upper()}_计算结果.xlsx"
    token = store_result(excel_bytes, filename)

    summary = {}
    for col in result.columns:
        col_data = result[col].dropna()
        if len(col_data) > 0:
            summary[col] = {
                "mean": round(float(col_data.mean()), 2),
                "std": round(float(col_data.std()), 2),
                "min": round(float(col_data.min()), 2),
                "max": round(float(col_data.max()), 2),
                "count": len(col_data),
            }

    return jsonify({
        "success": True,
        "scale_type": scale_type,
        "total_records": len(result),
        "summary": summary,
        "download_token": token,
    })


@app.route("/api/download/<token>")
def download(token):
    result = get_result(token)
    if not result:
        return jsonify({"error": "结果已过期（超过10分钟），请重新计算"}), 404
    return send_file(
        io.BytesIO(result["data"]),
        as_attachment=True,
        download_name=result["filename"],
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/api/generate_template/<scale_type>")
def generate_template(scale_type):
    questions = get_question_table(scale_type)
    df = pd.DataFrame(columns=[q["id"] for q in questions])
    example_row = {}
    for q in questions:
        vals = q["values"].split("-")
        example_row[q["id"]] = int(vals[0]) if len(vals) == 2 else ""
    df = pd.concat([df, pd.DataFrame([example_row])], ignore_index=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="数据", index=False)
        q_df = pd.DataFrame(questions)
        q_df.to_excel(writer, sheet_name="题目说明", index=False)

    output.seek(0)
    filename = f"{scale_type.upper()}_模板.xlsx"
    return send_file(output, as_attachment=True, download_name=filename,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/api/activate", methods=["POST"])
def activate():
    data = request.get_json()
    if data.get("admin_key") != "admin123":
        return jsonify({"error": "管理员密钥错误"}), 403
    codes = [generate_code(data.get("prefix", "SF")) for _ in range(data.get("count", 1))]
    return jsonify({"codes": codes})



@app.route("/admin")
def admin_page():
    return render_template("admin.html")


@app.route("/api/admin/codes")
def admin_codes():
    key = request.args.get("key", "")
    if key != "admin123":
        return jsonify({"error": "密码错误"}), 403
    codes_list = []
    for h, info in ACTIVATION_CODES.items():
        codes_list.append({
            "code": h[:16] + "...",
            "used": info["used"],
            "max_uses": info["max_uses"],
        })
    return jsonify({"codes": codes_list})
if __name__ == "__main__":
    app.run(debug=True, port=5000)