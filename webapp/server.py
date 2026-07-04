# -*- coding: utf-8 -*-
"""
embroidery_autopunch のブラウザUI。
posterize済み画像をアップロード → 刺しゅうデータ(.pes)を生成 →
針落ち経路のプレビュー・トリム数などの統計を表示 → .pes をダウンロードできる。
パーツ単位の縫い方変更・回転・元画像との比較にも対応する。
"""
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
    measure_thickness_px,
    generate_satin_stitches,
    generate_tatami_stitches,
    generate_radial_fill_stitches,
    HOOP_SIZE_MM,
    SATIN_PITCH_MM,
    TATAMI_PITCH_MM,
    TATAMI_OVERLAP_MM,
    MAX_STITCH_LENGTH_MM,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)

STITCH = pyembroidery.STITCH
JUMP = pyembroidery.JUMP
TRIM = pyembroidery.TRIM
COLOR_CHANGE = pyembroidery.COLOR_CHANGE

# token -> {"blocks": [StitchBlock], "px_per_mm": float, "image_shape": (h,w,..),
#           "hoop_mm": float, "created": time.time()}
# 生成済みパーツ(マスク含む)をメモリに保持し、パーツ単位の縫い方変更を
# 画像の再アップロードなしで再計算できるようにする。
SESSIONS = {}
SESSION_MAX_AGE_SEC = 3600


def _cleanup_sessions():
    now = time.time()
    for token in list(SESSIONS.keys()):
        if now - SESSIONS[token]["created"] > SESSION_MAX_AGE_SEC:
            del SESSIONS[token]
    for name in os.listdir(TMP_DIR):
        path = os.path.join(TMP_DIR, name)
        if now - os.path.getmtime(path) > SESSION_MAX_AGE_SEC:
            try:
                os.remove(path)
            except OSError:
                pass


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


def _full_mask_from_block(block, image_shape):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    bx, by = block.bbox_origin
    bh, bw = block.mask_crop.shape[:2]
    mask[by:by + bh, bx:bx + bw] = block.mask_crop
    return mask


def _regenerate_block_points(block, stitch_type, px_per_mm, image_shape):
    """block.mask_crop から指定の縫い方(stitch_type)で points_px を作り直す。"""
    full_mask = _full_mask_from_block(block, image_shape)
    satin_pitch_px = SATIN_PITCH_MM * px_per_mm
    tatami_pitch_px = TATAMI_PITCH_MM * px_per_mm
    max_stitch_px = MAX_STITCH_LENGTH_MM * px_per_mm

    if stitch_type == "satin":
        points = generate_satin_stitches(full_mask, satin_pitch_px)
    elif stitch_type == "radial":
        points = generate_radial_fill_stitches(full_mask, satin_pitch_px)
    else:
        overlap_px = TATAMI_OVERLAP_MM * px_per_mm
        thickness_px = measure_thickness_px(full_mask)
        local_overlap_px = min(overlap_px, thickness_px * 0.15)
        points = generate_tatami_stitches(full_mask, tatami_pitch_px, local_overlap_px, max_stitch_px)
        stitch_type = "tatami"

    if not points:
        return False
    block.points_px = points
    block.stitch_type = stitch_type
    return True


def _rotate_image(raw_image, rotation_deg):
    if rotation_deg == 90:
        return cv2.rotate(raw_image, cv2.ROTATE_90_CLOCKWISE)
    if rotation_deg == 180:
        return cv2.rotate(raw_image, cv2.ROTATE_180)
    if rotation_deg == 270:
        return cv2.rotate(raw_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return raw_image


def _build_response(token, blocks, px_per_mm, image_shape, hoop_mm):
    pattern = build_pattern(blocks, px_per_mm, image_shape)
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
            # pyembroideryの内部座標はY下向き正＝画像のY軸と同じ向きなので、
            # HTML canvas(Y下向き正)にそのまま渡せばよい（符号反転しない）。
            current_stroke["points"].append([round(x / 10.0, 2), round(y / 10.0, 2)])
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

    parts = []
    for i, b in enumerate(blocks):
        r, g, bch = b.color_bgr[2], b.color_bgr[1], b.color_bgr[0]
        parts.append({
            "index": i,
            "hex": "#%02x%02x%02x" % (r, g, bch),
            "area_mm2": round(b.area_px / (px_per_mm ** 2), 2),
            "stitch_type": b.stitch_type,
        })

    return {
        "token": token,
        "hoop_mm": hoop_mm,
        "px_per_mm": round(px_per_mm, 3),
        "stats": {
            "parts": len(blocks),
            "stitches": stitch_count,
            "jumps": jump_count,
            "trims": trim_count,
            "color_changes": color_change_count,
        },
        "threads": threads,
        "strokes": strokes,
        "parts_list": parts,
    }


@app.post("/api/generate")
def api_generate():
    _cleanup_sessions()

    if "image" not in request.files:
        return jsonify(error="image ファイルがありません"), 400
    file = request.files["image"]
    hoop_mm = float(request.form.get("hoop_mm", HOOP_SIZE_MM))
    rotation_deg = int(request.form.get("rotation_deg", 0))

    file_bytes_raw = file.read()
    file_bytes = np.frombuffer(file_bytes_raw, dtype=np.uint8)
    raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if raw_image is None:
        return jsonify(error="画像を読み込めませんでした"), 400

    raw_image = _rotate_image(raw_image, rotation_deg)

    bgr_image, background_mask = split_bgr_and_background_mask(raw_image)
    px_per_mm = compute_px_per_mm(bgr_image, hoop_mm)

    blocks = process_image(bgr_image, background_mask, px_per_mm)
    if not blocks:
        return jsonify(error="有効なパーツが検出できませんでした。色分解済みの画像か確認してください。"), 400

    token = uuid.uuid4().hex

    # 比較表示用に、実際に処理した(回転後の)画像をPNGとして保存しておく
    ok, png_bytes = cv2.imencode(".png", raw_image)
    if ok:
        with open(os.path.join(TMP_DIR, f"{token}.png"), "wb") as f:
            f.write(png_bytes.tobytes())

    SESSIONS[token] = {
        "blocks": blocks,
        "px_per_mm": px_per_mm,
        "image_shape": bgr_image.shape,
        "hoop_mm": hoop_mm,
        "created": time.time(),
    }

    return jsonify(**_build_response(token, blocks, px_per_mm, bgr_image.shape, hoop_mm))


@app.get("/api/parts/<token>")
def api_parts(token):
    session = SESSIONS.get(token)
    if session is None:
        return jsonify(error="セッションが見つかりません。再生成してください。"), 404
    parts = []
    for i, b in enumerate(session["blocks"]):
        r, g, bch = b.color_bgr[2], b.color_bgr[1], b.color_bgr[0]
        parts.append({
            "index": i,
            "hex": "#%02x%02x%02x" % (r, g, bch),
            "area_mm2": round(b.area_px / (session["px_per_mm"] ** 2), 2),
            "stitch_type": b.stitch_type,
        })
    return jsonify(parts=parts)


@app.post("/api/parts/<token>/<int:part_index>")
def api_update_part(token, part_index):
    session = SESSIONS.get(token)
    if session is None:
        return jsonify(error="セッションが見つかりません。再生成してください。"), 404
    blocks = session["blocks"]
    if not (0 <= part_index < len(blocks)):
        return jsonify(error="パーツが見つかりません"), 404

    stitch_type = (request.json or {}).get("stitch_type")
    if stitch_type not in ("satin", "tatami", "radial"):
        return jsonify(error="stitch_type は satin / tatami / radial のいずれかです"), 400

    ok = _regenerate_block_points(blocks[part_index], stitch_type, session["px_per_mm"], session["image_shape"])
    if not ok:
        return jsonify(error="この縫い方では有効なステッチが生成できませんでした（パーツが小さすぎる可能性があります）"), 400

    resp = _build_response(token, blocks, session["px_per_mm"], session["image_shape"], session["hoop_mm"])
    return jsonify(**resp)


@app.get("/api/image/<token>")
def api_image(token):
    if not token.isalnum():
        abort(400)
    path = os.path.join(TMP_DIR, f"{token}.png")
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="image/png")


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
