# -*- coding: utf-8 -*-
"""
embroidery_autopunch のブラウザUI。
posterize済み画像をアップロード → 刺しゅうデータ(.pes)を生成 →
針落ち経路のプレビュー・トリム数などの統計を表示 → .pes をダウンロードできる。
"""
import io
import sys
import os
import uuid
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file, abort

import pyembroidery
from embroidery_generator import (
    split_bgr_and_background_mask,
    compute_px_per_mm,
    process_image,
    build_pattern,
    HOOP_SIZE_MM,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)

STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


def _cleanup_old_tmp_files(max_age_sec=3600):
    now = time.time()
    for name in os.listdir(TMP_DIR):
        path = os.path.join(TMP_DIR, name)
        if now - os.path.getmtime(path) > max_age_sec:
            try:
                os.remove(path)
            except OSError:
                pass


@app.post("/api/generate")
def api_generate():
    _cleanup_old_tmp_files()

    if "image" not in request.files:
        return jsonify(error="image ファイルがありません"), 400
    file = request.files["image"]
    hoop_mm = float(request.form.get("hoop_mm", HOOP_SIZE_MM))

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw_image is None:
        return jsonify(error="画像を読み込めませんでした"), 400

    bgr_image, background_mask = split_bgr_and_background_mask(raw_image)
    px_per_mm = compute_px_per_mm(bgr_image, hoop_mm)

    blocks = process_image(bgr_image, background_mask, px_per_mm)
    if not blocks:
        return jsonify(error="有効なパーツが検出できませんでした。色分解済みの画像か確認してください。"), 400

    pattern = build_pattern(blocks, px_per_mm, bgr_image.shape)

    token = uuid.uuid4().hex
    pes_path = os.path.join(TMP_DIR, f"{token}.pes")
    pyembroidery.write(pattern, pes_path)

    # PEC書き出し後に読み直すことで、実機で実際に発生するトリム/ジャンプ回数・
    # スナップ後の糸色を正確に取得する（trim()呼び出し回数と実際のトリム回数は一致しない）。
    encoded = pyembroidery.read(pes_path)

    threads = []
    for t in encoded.threadlist:
        r, g, b = t.get_red(), t.get_green(), t.get_blue()
        threads.append({
            "r": r, "g": g, "b": b,
            "hex": "#%02x%02x%02x" % (r, g, b),
            "name": t.description or "",
        })

    stitch_count = jump_count = trim_count = color_change_count = 0
    strokes = []
    thread_idx = 0
    current_stroke = None

    def flush_stroke():
        nonlocal current_stroke
        if current_stroke is not None and len(current_stroke["points"]) >= 2:
            strokes.append(current_stroke)
        current_stroke = None

    for x, y, cmd in encoded.stitches:
        if cmd == STITCH:
            stitch_count += 1
            if current_stroke is None:
                current_stroke = {"thread": thread_idx, "points": []}
            current_stroke["points"].append([round(x / 10.0, 2), round(-y / 10.0, 2)])
        elif cmd == JUMP:
            jump_count += 1
            flush_stroke()
        elif cmd == TRIM:
            trim_count += 1
            flush_stroke()
        elif cmd == COLOR_CHANGE:
            color_change_count += 1
            flush_stroke()
            thread_idx += 1
    flush_stroke()

    return jsonify(
        token=token,
        hoop_mm=hoop_mm,
        px_per_mm=round(px_per_mm, 3),
        stats={
            "parts": len(blocks),
            "stitches": stitch_count,
            "jumps": jump_count,
            "trims": trim_count,
            "color_changes": color_change_count,
        },
        threads=threads,
        strokes=strokes,
    )


@app.get("/api/download/<token>")
def api_download(token):
    if not token.isalnum():
        abort(400)
    path = os.path.join(TMP_DIR, f"{token}.pes")
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name="embroidery.pes")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
