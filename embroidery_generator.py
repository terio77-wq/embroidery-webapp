"""
刺しゅうオートパンチデータ生成スクリプト
=======================================

色分解・減色・境界線のクッキリ化が完了しているイラスト（グラデーション/ノイズなし）を読み込み、
画像処理と幾何学計算のみ（AI/機械学習は不使用）で刺しゅうデータ（.pes / .dst）を自動生成する。

入力画像の前提:
    - 背景を除く各色の領域は完全に塗り分けられ、境界線がクッキリしている
    - 背景色は画像四隅のピクセル色から自動判定する
    - 画像は正方形（Skitch PP1 の刺しゅう枠 10cm×10cm にそのままフィットさせる前提）

処理フロー:
    ① 画像を刺しゅう枠サイズ(10cm×10cm)にフィッティング → 1mmあたりのピクセル数(px_per_mm)を自動算出
    ② 色（レイヤー）ごとにマスクを抽出 → 孤立パーツを個別に分離
    ③ distanceTransform で各パーツの太さを算出し、5mm を境にステッチ方式を自動分岐
         - 5mm未満 → サテン縫い（中心線から輪郭の左右を交互にジグザグ）
         - 5mm以上 → タタミ縫い（1mm膨張させてから水平スライスをジグザグ一筆書き）
    ④ 面積の大きい順にステッチブロックをソートし、下地→細部の順に重ね縫いされるようにする
    ⑤ pyembroidery で色替えコマンドを挟みながら .pes / .dst として書き出す（座標は枠中心を原点(0,0)に配置）

Usage:
    python embroidery_generator.py input.png output.pes

    入力画像は正方形で渡す前提。画像の一辺のピクセル数を HOOP_SIZE_MM(10cm) に
    フィッティングして px_per_mm を自動算出するため、通常は追加の指定は不要。
    正方形でない画像や、枠サイズを変更したい場合のみ --hoop-size-mm で上書きできる。
"""

import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
import pyembroidery


# ============================================================
# パラメーター
# ============================================================

SATIN_PITCH_MM = 0.45          # サテン縫いの針落ちピッチ（0.4〜0.5mm）
TATAMI_PITCH_MM = 0.4          # タタミ縫いのスライス間隔
TATAMI_OVERLAP_MM = 1.0        # タタミ縫いのオーバーラップ（膨張量）
MAX_STITCH_LENGTH_MM = 3.0     # 1針あたりの最大針幅。タタミの行内分割、および
                                # これを超える移動を区間分割(run分割)する際の閾値として使う
TRIM_THRESHOLD_MM = 5.0        # 区間の切れ目でこの距離以下の移動ならトリムせずジャンプのみで渡る
                                # （浮き糸として許容し、トリム回数・機械停止時間を削減する）
THICKNESS_THRESHOLD_MM = 5.0   # サテン/タタミの分岐閾値（パーツの太さ）
MIN_PART_AREA_PX = 15          # これ未満の面積の孤立パーツはノイズとして無視
MIN_SATIN_STITCH_POINTS = 16   # サテン縫いの生成点数がこれ未満なら塗り残しが出るとみなし
                                # タタミ縫い（スキャンライン塗りつぶし）にフォールバックする
MAX_SATIN_RUN_DENSITY = 0.5    # サテン縫いを生成してみた結果、区間(run)の数が
                                # 面積1mm2あたりこれを超えて断片化する形状は、
                                # タタミ縫い（スキャンライン方式、形状の凹凸に
                                # 影響されにくい）にフォールバックする。
                                # （例: もこもこした毛の陰影のような、輪郭に多数の
                                # 凹凸がある三日月状のパーツ。サテンは中心線を1本の
                                # ジグザグでたどる方式のため、凹凸だらけの形状では
                                # 小さな出っ張りに入っては引き返す動作を繰り返し、
                                # ジグザグが不自然に飛び回って見た目が悪化する）
                                # ※スケルトンの分岐点数で判定していたが、実際の
                                # run数との相関が弱く（分岐点が少なくても激しく
                                # 断片化する形状、逆に分岐点が多くても実際は
                                # あまり断片化しない形状の両方があった）、
                                # 実際にサテンを生成してrun数を直接調べる方式に変更した。
MIN_SATIN_RUN_ALLOWANCE = 3    # 面積が小さいパーツでは上のrun密度基準だと
                                # 1run増えるだけで簡単に超過してしまうため、
                                # 面積に関わらず最低限許容するrun数
MERGE_GAP_MM = 0.2              # 同じ色でこの距離以内に隣接する極小パーツは1つのグループ
                                # （1つの自動パンチ対象）として統合する。
                                # 文字の字間（実測: 約0.39mm〜）より必ず小さくすること。
                                # そうしないと文字同士まで1つのパーツに結合されてしまう。
                                # あくまでアンチエイリアス由来の1〜2px程度のノイズ的な
                                # 分断を橋渡しするための値であり、意匠として意図的に
                                # 離れている部位（文字・目など）を結合する用途ではない。
MERGE_MAX_AREA_MM2 = 8.0       # 統合の対象は面積がこれ未満の「極小パーツ」同士に限定する。
                                # 既に十分な大きさを持つパーツ（縁取りや本体など）を
                                # 巻き込んでしまわないようにするため
HOLE_FILL_MAX_AREA_MM2 = 8.0   # タグの中のロゴ文字のように、ある色の領域の中に完全に
                                # 囲まれた小さな別パーツ（穴）がある場合、その下地側では
                                # 精密にくり抜こうとしない。面積がこれ未満で、かつ背景に
                                # 接していない（＝他の前景色に完全に囲まれている）穴は、
                                # 下地の段階では埋めて一色で塗ってしまい、後から面積の
                                # 小さい方（＝穴の中身の色）を上から重ね縫いする。
                                # ステッチブロックは面積降順で処理されるため、下地が先に
                                # 縫われ、細かい文字などは自動的にその上から縫われる。
                                # こうすることで、下地と上物の境界を画素単位で精密に
                                # 一致させる必要がなくなり（隙間・ズレのリスクを回避）、
                                # 下地パーツの形状自体も単純になりステッチ品質も上がる。
                                # 背景に接する穴（意匠として本来背景を見せたい部分、
                                # 例: 文字の"O"の内側の抜きなど）は対象外とし、塗り残す。
PES_UNITS_PER_MM = 10.0        # pyembroidery の内部座標単位（1 unit = 0.1mm）
HOOP_SIZE_MM = 100.0           # Skitch PP1 の刺しゅう枠サイズ（10cm×10cm 正方形）
SQUARE_TOLERANCE_RATIO = 0.02  # 縦横比がこの割合を超えて異なる場合は警告する


@dataclass
class StitchBlock:
    """1つの色パーツ分の針落ち点データ"""
    color_bgr: tuple
    points_px: list        # [(x, y), ...] 画像ピクセル座標系
    area_px: float
    stitch_type: str       # "satin" または "tatami"
    mask_crop: np.ndarray  # このパーツのマスク（bbox_originで切り出し済み）。
                            # 同一パーツ内でのrun間の橋渡し経路探索に使う
    bbox_origin: tuple     # mask_crop の左上が画像全体座標系でどこにあるか (x, y)


# ============================================================
# ステップ①: 刺しゅう枠(10cm×10cm)へのフィッティング
# ============================================================

def compute_px_per_mm(image: np.ndarray, hoop_size_mm: float) -> float:
    """
    正方形画像を hoop_size_mm × hoop_size_mm の刺しゅう枠にフィッティングし、
    1mmあたりのピクセル数(px_per_mm)を算出する。
    画像が正方形でない場合は警告を出し、短辺を枠に収まる基準として使う。
    """
    h, w = image.shape[:2]
    if abs(w - h) > max(w, h) * SQUARE_TOLERANCE_RATIO:
        print(
            f"警告: 入力画像が正方形ではありません ({w}x{h}px)。"
            f"短辺を基準に {hoop_size_mm}mm 枠へフィッティングします。",
            file=sys.stderr,
        )
    side_px = min(w, h)
    return side_px / hoop_size_mm


# ============================================================
# ステップ②: 色（レイヤー）の分離と輪郭抽出
# ============================================================

def imread_unicode(path: str, flags=cv2.IMREAD_UNCHANGED):
    """
    日本語などの非ASCII文字を含むパスでも読み込めるcv2.imread代替。
    cv2.imread()はWindowsで非ASCIIパスを渡すと無言でNoneを返すため、
    np.fromfile + cv2.imdecode 経由で読み込む。
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def split_bgr_and_background_mask(raw_image: np.ndarray, alpha_threshold: int = 128):
    """
    読み込んだ画像からBGR画像と「背景であるか」を示すbool maskを作る。

    - アルファチャンネルがある場合（透過PNG）は alpha < alpha_threshold を背景とする。
      透過部分はRGB値が(0,0,0)などデザイン内の黒色パーツと衝突しうるため、
      背景判定には必ずアルファチャンネルを優先して使う。
    - アルファチャンネルがない場合は、四隅のピクセル色の最頻値を背景色とみなす。
    """
    if raw_image.ndim == 3 and raw_image.shape[2] == 4:
        bgr = raw_image[:, :, :3]
        alpha = raw_image[:, :, 3]
        background_mask = alpha < alpha_threshold
        return bgr, background_mask

    bgr = raw_image[:, :, :3] if raw_image.ndim == 3 else cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    corners = [
        tuple(int(c) for c in bgr[0, 0]),
        tuple(int(c) for c in bgr[0, w - 1]),
        tuple(int(c) for c in bgr[h - 1, 0]),
        tuple(int(c) for c in bgr[h - 1, w - 1]),
    ]
    bg_color = np.array(max(set(corners), key=corners.count), dtype=int)
    diff = np.abs(bgr.astype(int) - bg_color)
    background_mask = np.all(diff <= 10, axis=2)
    return bgr, background_mask


def extract_color_masks(bgr_image: np.ndarray, background_mask: np.ndarray):
    """
    背景（background_mask）を除く色ごとのマスクを抽出する。
    戻り値: [(color_bgr, mask), ...]
    """
    foreground_pixels = bgr_image[~background_mask]
    if foreground_pixels.size == 0:
        return []
    unique_colors = np.unique(foreground_pixels.reshape(-1, 3), axis=0)

    results = []
    for color in unique_colors:
        mask = cv2.inRange(bgr_image, color, color)
        mask[background_mask] = 0
        if cv2.countNonZero(mask) < MIN_PART_AREA_PX:
            continue
        results.append((tuple(int(c) for c in color), mask))
    return results


def fill_enclosed_holes(color_mask: np.ndarray, background_mask: np.ndarray, max_hole_area_px: float) -> np.ndarray:
    """
    color_mask 内部にある小さな穴のうち、背景に接していないもの（＝他の前景色パーツに
    完全に囲まれている、タグの中のロゴ文字のような部分）を埋める。
    下地パーツを精密にくり抜かなくても、後から面積の小さい方（穴の中身の色）が
    面積降順ソートにより自動的に上から重ね縫いされるため、仕上がりは変わらない。
    """
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return color_mask

    background_u8 = background_mask.astype(np.uint8) * 255
    filled = color_mask.copy()
    for i, h in enumerate(hierarchy[0]):
        if h[3] == -1:
            continue  # 外側輪郭（穴ではない）はスキップ

        hole_mask = np.zeros_like(color_mask)
        cv2.drawContours(hole_mask, contours, i, 255, thickness=cv2.FILLED)
        area_px = cv2.countNonZero(hole_mask)
        if area_px == 0 or area_px > max_hole_area_px:
            continue
        if cv2.countNonZero(cv2.bitwise_and(hole_mask, background_u8)) > 0:
            continue  # 背景に接する穴は意匠上の抜きなので塗り残す

        filled = cv2.bitwise_or(filled, hole_mask)
    return filled


def split_into_parts(mask: np.ndarray, merge_gap_px: float = 0.0, merge_max_area_px: float = None):
    """
    同一色のマスクを孤立パーツに分離する。

    merge_gap_px > 0 を指定すると、面積が merge_max_area_px 未満の「極小パーツ」同士
    に限り、その距離以内で隣接していれば同一グループ（1つのStitchBlock）として統合する。
    目のハイライトや黒目のような細かい別色パーツを挟んで、本来は1つの意匠（例: 赤目2つ）
    であるはずの同色領域が、アンチエイリアスや重なりの都合で無数の極小パーツに
    分断されてしまうのを防ぐための処理。
    統合の対象を「極小パーツ同士」に限定しているのは、縁取りや本体のような
    既に十分大きいパーツまで巻き込んで無関係な位置と繋がってしまうのを防ぐため。
    """
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    raw_parts = []
    for label in range(1, num_labels):
        part_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        if cv2.countNonZero(part_mask) < MIN_PART_AREA_PX:
            continue
        raw_parts.append(part_mask)

    if merge_gap_px <= 0 or merge_max_area_px is None:
        return raw_parts

    small_parts = [p for p in raw_parts if cv2.countNonZero(p) < merge_max_area_px]
    large_parts = [p for p in raw_parts if cv2.countNonZero(p) >= merge_max_area_px]
    if not small_parts:
        return large_parts

    small_mask = np.zeros_like(mask)
    for p in small_parts:
        small_mask = cv2.bitwise_or(small_mask, p)

    # 膨張は両側から同時に効くため、半径 merge_gap_px/2 で膨張させれば
    # 隙間 merge_gap_px 以内のパーツ同士が連結する（半径=merge_gap_pxだと
    # 実際には2倍の距離まで統合されてしまうので注意）
    kernel_radius = max(1, int(round(merge_gap_px / 2.0)))
    kernel_size = kernel_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grouping_mask = cv2.dilate(small_mask, kernel)

    num_labels2, labels2 = cv2.connectedComponents(grouping_mask, connectivity=8)
    merged_small_parts = []
    for label in range(1, num_labels2):
        group_mask = np.where(labels2 == label, 255, 0).astype(np.uint8)
        part_mask = cv2.bitwise_and(small_mask, group_mask)
        if cv2.countNonZero(part_mask) < MIN_PART_AREA_PX:
            continue
        merged_small_parts.append(part_mask)

    return large_parts + merged_small_parts


# ============================================================
# ステップ③: パーツの「太さ」によるステッチ自動分岐ロジック
# ============================================================

def measure_thickness_px(part_mask: np.ndarray) -> float:
    """distanceTransform による最大内接円の直径 = パーツの太さ(px)"""
    dist = cv2.distanceTransform(part_mask, cv2.DIST_L2, 5)
    return float(dist.max()) * 2.0


# ---- 分岐A: サテン縫い（太さ5mm未満） ----------------------------------

def _order_skeleton_points(skeleton: np.ndarray):
    """
    スケルトン画素を1本の経路として順序付ける（分岐対応のDFSバックトラック走査）。

    文字やループ状の輪郭（例: 顔の輪郭全体を縫う縫い枠線）は、スケルトンが
    分岐点（Y字路）や閉ループを含む「グラフ」になる。単純な貪欲最近傍法で
    行き止まりに達すると、離れた未訪問画素へ直線的に「ジャンプ」してしまい、
    デザインの内部を突っ切る誤った針目になる（実際にこの問題で長い斜め線が
    生成されることを確認済み）。
    そこで、隣接画素のみを辿るDFSを行い、行き止まりでは直前の分岐点まで
    後戻り（同じ経路を逆再生）することで、常に隣接画素間の移動のみで
    全画素を1本の経路として辿れるようにする。分岐やループがある場合は
    一部区間を往復するが、隣接画素間の移動しか発生しないため、
    デザインを突っ切るような不正な針目は原理的に発生しない。
    """
    ys, xs = np.nonzero(skeleton)
    points = list(zip(xs.tolist(), ys.tolist()))
    if not points:
        return []

    point_set = set(points)

    def neighbors(p):
        x, y = p
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in point_set:
                    yield q

    endpoints = [p for p in points if sum(1 for _ in neighbors(p)) == 1]
    start = endpoints[0] if endpoints else points[0]

    visited = set()
    path = []
    remaining = set(points)

    # 万一スケルトンが複数の孤立した断片に分かれている場合も取りこぼさないよう、
    # 未訪問の画素が残っていれば新たな起点からDFSを繰り返す
    # （断片間の移動は後段の split_into_runs が距離判定してジャンプに変換する）
    while remaining:
        stack = [start]
        visited.add(start)
        path.append(start)
        remaining.discard(start)

        while stack:
            current = stack[-1]
            nxt = next((q for q in neighbors(current) if q not in visited), None)
            if nxt is not None:
                visited.add(nxt)
                remaining.discard(nxt)
                stack.append(nxt)
                path.append(nxt)
            else:
                stack.pop()
                if stack:
                    # 行き止まりから直前の分岐点まで後戻り（隣接画素間の移動のみ）
                    path.append(stack[-1])

        if remaining:
            start = next(iter(remaining))

    return path


def _resample_path(path, pitch_px: float):
    """経路を一定間隔(pitch_px)で再サンプリングする"""
    if len(path) < 2:
        return path

    pts = np.array(path, dtype=float)
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum_len[-1]
    if total_len < pitch_px:
        return [tuple(pts[0]), tuple(pts[-1])]

    n_samples = max(2, int(total_len // pitch_px) + 1)
    sample_dists = np.linspace(0, total_len, n_samples)

    resampled = []
    for d in sample_dists:
        idx = np.searchsorted(cum_len, d)
        idx = min(max(idx, 1), len(cum_len) - 1)
        d0, d1 = cum_len[idx - 1], cum_len[idx]
        t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
        p = pts[idx - 1] + t * (pts[idx] - pts[idx - 1])
        resampled.append((float(p[0]), float(p[1])))
    return resampled


def _find_edge_along_ray(mask: np.ndarray, origin, direction, max_len: float):
    """origin から direction 方向へ進み、mask の外に出る直前の座標を返す"""
    x0, y0 = origin
    dx, dy = direction
    h, w = mask.shape
    last = (x0, y0)
    step = 0.5
    dist = 0.0
    while dist < max_len:
        x = x0 + dx * dist
        y = y0 + dy * dist
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= w or iy >= h or mask[iy, ix] == 0:
            break
        last = (x, y)
        dist += step
    return last


def generate_satin_stitches(part_mask: np.ndarray, pitch_px: float):
    """
    領域の中心線（スケルトン）を抽出し、中心線に直交する左右の輪郭を
    交互に行き来する密なジグザグの針落ち点リストを生成する。
    """
    skeleton = skeletonize(part_mask > 0)
    path = _order_skeleton_points(skeleton)
    if len(path) < 2:
        return []

    centerline = _resample_path(path, pitch_px)

    dist = cv2.distanceTransform(part_mask, cv2.DIST_L2, 5)
    max_half_width = float(dist.max()) + 2.0  # 探索の安全マージン

    stitches = []
    for i, (cx, cy) in enumerate(centerline):
        if i == 0:
            tangent = np.array(centerline[1]) - np.array(centerline[0])
        elif i == len(centerline) - 1:
            tangent = np.array(centerline[-1]) - np.array(centerline[-2])
        else:
            tangent = np.array(centerline[i + 1]) - np.array(centerline[i - 1])

        norm = np.linalg.norm(tangent)
        if norm == 0:
            continue
        tangent = tangent / norm
        perp = np.array([-tangent[1], tangent[0]])

        left = _find_edge_along_ray(part_mask, (cx, cy), tuple(perp), max_half_width)
        right = _find_edge_along_ray(part_mask, (cx, cy), tuple(-perp), max_half_width)

        if i % 2 == 0:
            stitches.append(left)
            stitches.append(right)
        else:
            stitches.append(right)
            stitches.append(left)

    return stitches


def generate_radial_fill_stitches(part_mask: np.ndarray, pitch_px: float):
    """
    目のハイライトのような丸い/コンパクトな極小パーツ向けの塗りつぶし方式。

    サテン縫いは形状の中心線（スケルトン）を1本の軸としてジグザグに辿る
    方式のため、丸い点のような「中心線」と呼べる軸が存在しない形状では
    スケルトンがほぼ1点に潰れてしまい、縫い目を生成できない。
    タタミ縫いは水平スキャンラインで塗りつぶすため任意の形状に対応できるが、
    塗り残し防止のオーバーラップ(膨張)が必要で、パーツ自体が数mm角の
    タタミの行間隔(TATAMI_PITCH_MM)程度まで小さいと、そのオーバーラップで
    意匠より明らかに大きく膨張したり、逆に0にすると数本しかない
    スキャンラインの量子化でサイズ・形が不安定になったりする。

    この方式は、重心から輪郭までの放射状の線（スポーク）を交互に
    ジグザグで辿ることで、中心線に頼らず・膨張も使わずに、実際の
    輪郭ぴったりのサイズで塗りつぶす。丸くもコンパクトな凹凸のない
    形状であれば、円・楕円に限らずどんな形でも輪郭に忠実に塗れる。
    """
    ys, xs = np.nonzero(part_mask)
    if len(xs) == 0:
        return []
    cx, cy = float(xs.mean()), float(ys.mean())

    dist = cv2.distanceTransform(part_mask, cv2.DIST_L2, 5)
    max_r = float(dist.max()) + 2.0  # 探索の安全マージン

    n_spokes = max(8, int(round(2 * np.pi * max_r / pitch_px)))

    stitches = []
    for i in range(n_spokes):
        angle = 2 * np.pi * i / n_spokes
        direction = (np.cos(angle), np.sin(angle))
        edge = _find_edge_along_ray(part_mask, (cx, cy), direction, max_r)
        if i % 2 == 0:
            stitches.append((cx, cy))
            stitches.append(edge)
        else:
            stitches.append(edge)
            stitches.append((cx, cy))

    return stitches


# ---- 分岐B: タタミ縫い（太さ5mm以上） ----------------------------------

def _find_runs(xs_sorted: np.ndarray):
    """ソート済みインデックス配列から連続する区間 [(start, end), ...] を求める"""
    runs = []
    start = prev = int(xs_sorted[0])
    for x in xs_sorted[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    return runs


def generate_tatami_stitches(part_mask: np.ndarray, pitch_px: float, overlap_px: float, max_stitch_px: float):
    """
    領域を overlap_px 分だけ膨張（オーバーラップ処理）させたのち、
    水平方向のスライスと輪郭の交点をジグザグに一筆書きで繋ぎ、
    面全体を塗りつぶす針落ち点リストを生成する。

    各スライス（行）の区間は run_start→run_end を1本の針落ちで結ぶのではなく、
    max_stitch_px を超えないよう細かく分割する。実機・刺しゅうデータ形式は
    1針あたりの最大針幅に制限があり、分割しないと広い面で極端に長い
    （渡り糸状の）針目になってしまうため。
    """
    kernel_size = max(1, int(round(overlap_px))) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(part_mask, kernel)

    ys, _ = np.nonzero(dilated)
    if len(ys) == 0:
        return []
    y_min, y_max = int(ys.min()), int(ys.max())

    stitches = []
    direction = 1  # 1: 左→右, -1: 右→左
    row = float(y_min)
    while row <= y_max:
        y = int(round(row))
        xs_on = np.nonzero(dilated[y, :])[0]
        if len(xs_on) > 0:
            runs = _find_runs(xs_on)
            row_points = []
            for run_start, run_end in runs:
                run_length = abs(run_end - run_start)
                n_steps = max(1, int(np.ceil(run_length / max_stitch_px)))
                xs_interp = np.linspace(run_start, run_end, n_steps + 1)
                row_points.extend((float(x), float(y)) for x in xs_interp)
            if direction == -1:
                row_points = row_points[::-1]
            stitches.extend(row_points)
            direction *= -1
        row += pitch_px

    return stitches


# ============================================================
# ステップ④: 分岐処理・面積によるソート
# ============================================================

def process_image(bgr_image: np.ndarray, background_mask: np.ndarray, px_per_mm: float) -> list:
    """画像全体を処理し、色パーツごとの StitchBlock リストを面積降順で返す"""
    color_masks = extract_color_masks(bgr_image, background_mask)

    satin_pitch_px = SATIN_PITCH_MM * px_per_mm
    tatami_pitch_px = TATAMI_PITCH_MM * px_per_mm
    overlap_px = TATAMI_OVERLAP_MM * px_per_mm
    max_stitch_px = MAX_STITCH_LENGTH_MM * px_per_mm
    thickness_threshold_px = THICKNESS_THRESHOLD_MM * px_per_mm
    merge_gap_px = MERGE_GAP_MM * px_per_mm
    merge_max_area_px = MERGE_MAX_AREA_MM2 * (px_per_mm ** 2)
    hole_fill_max_area_px = HOLE_FILL_MAX_AREA_MM2 * (px_per_mm ** 2)

    blocks = []
    for color_bgr, color_mask in color_masks:
        color_mask = fill_enclosed_holes(color_mask, background_mask, hole_fill_max_area_px)
        for part_mask in split_into_parts(color_mask, merge_gap_px, merge_max_area_px):
            thickness_px = measure_thickness_px(part_mask)
            area_px = cv2.countNonZero(part_mask)
            area_mm2 = area_px / (px_per_mm ** 2)

            is_thin_enough = thickness_px < thickness_threshold_px

            # タタミ縫いのオーバーラップ(TATAMI_OVERLAP_MM=1.0mm)は、本来は隣接する
            # 大きな塗りつぶし同士の継ぎ目を隠すための値。目のハイライトのような
            # 太さ1mm程度の極小パーツにそのまま適用すると、全方向に大きく
            # 膨張してしまい、本来の意匠より明らかに大きく見えてしまう
            # （半径をパーツの太さの0.5倍にする案も試したが、それでも
            # 1mmの点が2mm超まで膨張し、意匠より一回り大きく見えてしまった）。
            # パーツ自身の太さの0.15倍を上限にすることで、大きい塗りつぶしでは
            # 従来通りの継ぎ目処理を保ちつつ、小さいパーツの膨張を実寸に近い
            # レベルまで抑える。
            local_overlap_px = min(overlap_px, thickness_px * 0.15)

            if is_thin_enough:
                points = generate_satin_stitches(part_mask, satin_pitch_px)
                stitch_type = "satin"
                if len(points) < MIN_SATIN_STITCH_POINTS:
                    # 目のハイライトのような丸い/コンパクトな極小パーツは、
                    # 「中心線」と呼べる軸が存在しないためスケルトンがほぼ1点に
                    # 潰れてしまい、サテンの生成点数が足りず塗り残しになる。
                    # タタミ縫いはオーバーラップ(膨張)なしでは小さい形状ほど
                    # スキャンライン量子化でサイズ・形が不安定になるため、
                    # 重心からの放射状ジグザグで輪郭ぴったりに塗る方式に切り替える。
                    points = generate_radial_fill_stitches(part_mask, satin_pitch_px)
                    stitch_type = "radial"
                else:
                    run_allowance = max(MIN_SATIN_RUN_ALLOWANCE, area_mm2 * MAX_SATIN_RUN_DENSITY)
                    n_runs = len(split_into_runs(points, max_stitch_px))
                    if n_runs > run_allowance:
                        # サテンの生成自体は成立するが、区間(run)が面積の割に
                        # 断片化しすぎている（もこもこした毛の陰影のような、輪郭に
                        # 多数の凹凸がある形状。サテンのジグザグが不自然に飛び回って
                        # 見た目が悪化する）場合は、スキャンライン方式のタタミ縫いに
                        # 切り替える。
                        points = generate_tatami_stitches(part_mask, tatami_pitch_px, local_overlap_px, max_stitch_px)
                        stitch_type = "tatami"
            else:
                points = generate_tatami_stitches(part_mask, tatami_pitch_px, local_overlap_px, max_stitch_px)
                stitch_type = "tatami"

            if not points:
                continue

            bx, by, bw, bh = cv2.boundingRect(part_mask)
            blocks.append(StitchBlock(
                color_bgr=color_bgr,
                points_px=points,
                area_px=area_px,
                stitch_type=stitch_type,
                mask_crop=part_mask[by:by + bh, bx:bx + bw].copy(),
                bbox_origin=(bx, by),
            ))

    # 色ごとにグループ化して連続して縫うことで、色変え回数を実際の色数まで抑える。
    # パーツ単位で単純に面積降順に並べると、同じ色でも他の色のパーツより
    # 小さければ順番がバラバラになり、同じ色に何度も戻ってきてしまう
    # （実際の色数は8色でも、色変えが50回以上に増えてしまう不具合があった）。
    # 色グループ同士の順序は、各色の最大パーツ面積（＝その色が担う一番大きな
    # 役割）で決める。最大面積が大きい色ほど「下地」的な色とみなして先に縫い、
    # 小さい色ほど「上物」的な色とみなして後に縫う。これにより、タグの中の
    # 文字のように大きい色の上に小さい色が重ね縫いされる構図を壊さずに、
    # 色グループをまとめられる。各色グループの内部では、パーツ同士を
    # 面積降順に並べる（大きい方を先に縫うのは同色内でも変わらない）。
    max_area_by_color = {}
    for b in blocks:
        max_area_by_color[b.color_bgr] = max(max_area_by_color.get(b.color_bgr, 0), b.area_px)
    blocks.sort(key=lambda b: (max_area_by_color[b.color_bgr], b.area_px), reverse=True)
    return blocks


# ============================================================
# ステップ⑤: 刺しゅうデータの書き出し（pyembroidery）
# ============================================================

def split_into_runs(points: list, max_gap_px: float) -> list:
    """
    点列を「連続して針を落として問題ない区間（run）」ごとに分割する。
    タタミ縫いで凹形状・穴（例: 顔の輪郭が目の周りで途切れる部分）を
    またぐ際、隣り合う交点同士の距離が max_gap_px を超える場合は
    生地の上を長い1針で突っ切ってしまうため、そこで区間を分けてジャンプさせる。
    """
    if not points:
        return []

    runs = [[points[0]]]
    for x, y in points[1:]:
        px, py = runs[-1][-1]
        dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        if dist > max_gap_px:
            runs.append([(x, y)])
        else:
            runs[-1].append((x, y))
    return runs


def _order_runs_nearest(runs: list, start_point) -> list:
    """
    区間(run)群を貪欲最近傍法で並べ替える。各runは始点・終点どちらからでも
    縫えるため、現在位置から見て近い方の端を選んで向きも決める。
    完全なTSP最適解ではないが、隣接する区間同士を優先的に連続させることで
    移動距離（延いてはトリム対象になる長距離ジャンプ）を減らせる。
    """
    remaining = list(runs)
    ordered = []
    current = start_point
    while remaining:
        if current is None:
            best_idx, best_reverse = 0, False
        else:
            best_idx, best_reverse, best_dist = None, False, None
            for i, r in enumerate(remaining):
                d_start = (current[0] - r[0][0]) ** 2 + (current[1] - r[0][1]) ** 2
                d_end = (current[0] - r[-1][0]) ** 2 + (current[1] - r[-1][1]) ** 2
                if best_dist is None or d_start < best_dist:
                    best_dist, best_idx, best_reverse = d_start, i, False
                if d_end < best_dist:
                    best_dist, best_idx, best_reverse = d_end, i, True
        run = remaining.pop(best_idx)
        if best_reverse:
            run = list(reversed(run))
        ordered.append(run)
        current = run[-1]
    return ordered


def _bridge_within_mask(mask_crop: np.ndarray, bbox_origin: tuple, p0: tuple, p1: tuple, pitch_px: float) -> list:
    """
    同一パーツ内で p0 から p1 へ、パーツの内部だけを通る経路で繋ぐ針落ち点列を返す
    （p0 自体は含まない。p1 で終わる）。

    .pes/.pec 形式は最初の1回を除きJUMPコマンドが必ず自動的にトリムとして
    エンコードされる（pattern.trim()の呼び出し自体はPEC側で無視される）ため、
    同一パーツ内のrunの切れ目でJUMPを使うと不要なトリムが大量発生する。
    パーツは connectedComponents で分離済み＝内部で必ず連結しているため、
    マスク内部のコストを低く・外側のコストを高く設定した経路探索
    （route_through_array、ダイクストラ法）で、同色領域からはみ出さない
    経路を必ず見つけられる。これを一定間隔で再サンプリングしSTITCHとして
    繋ぐことで、意匠を崩さずJUMP（＝トリム）を回避する。
    """
    x0, y0 = bbox_origin
    h, w = mask_crop.shape
    costs = np.where(mask_crop > 0, 1.0, 1000.0).astype(np.float64)

    def to_rc(p):
        r = min(max(int(round(p[1])) - y0, 0), h - 1)
        c = min(max(int(round(p[0])) - x0, 0), w - 1)
        return r, c

    start_rc = to_rc(p0)
    end_rc = to_rc(p1)
    indices, _ = route_through_array(costs, start_rc, end_rc, fully_connected=True)
    path_px = [(x0 + c, y0 + r) for r, c in indices]

    resampled = _resample_path(path_px, pitch_px)
    return resampled[1:]  # 始点(=現在の針位置)は既に縫われているので除く


def build_pattern(blocks: list, px_per_mm: float, image_shape: tuple) -> "pyembroidery.EmbPattern":
    """
    image_shape: 元画像の (height, width)。
    刺しゅう枠の中心 = 画像の中心 が原点(0,0)になるよう座標をオフセットする
    （Artspira / 刺しゅうミシンは通常、枠の中心にデザインの中心を合わせて配置するため）。
    """
    pattern = pyembroidery.EmbPattern()
    px_to_units = PES_UNITS_PER_MM / px_per_mm  # px -> 0.1mm単位（pyembroidery内部座標）
    max_gap_px = MAX_STITCH_LENGTH_MM * px_per_mm
    trim_threshold_px = TRIM_THRESHOLD_MM * px_per_mm

    h, w = image_shape[:2]
    center_x_px, center_y_px = w / 2.0, h / 2.0

    def to_units(x_px, y_px):
        return (x_px - center_x_px) * px_to_units, (y_px - center_y_px) * px_to_units

    prev_color = None
    is_first_run_overall = True
    current_point_px = None  # パターン全体を通した直前の針落ち位置（run並べ替え・トリム判定用）

    for block in blocks:
        is_first_block = prev_color is None
        color_changed = (not is_first_block) and block.color_bgr != prev_color

        if color_changed:
            pattern.color_change()
        if is_first_block or color_changed:
            b, g, r = block.color_bgr
            thread = pyembroidery.EmbThread()
            thread.set_color(r, g, b)
            pattern.add_thread(thread)

        runs = split_into_runs(block.points_px, max_gap_px)
        # ブロック内の区間を貪欲最近傍で並べ替え、直前位置からの移動距離を短縮する
        # → 短い移動が増え、トリム不要（閾値以下のジャンプ）で渡れる区間切り替えが増える
        runs = _order_runs_nearest(runs, current_point_px)

        block_first_run = True
        for run in runs:
            if is_first_run_overall or block_first_run:
                # 別パーツ（別の色 or 別の孤立部位）への移動は、実際に糸を切る必要がある。
                # pyembroideryの正規化処理は「trim()やcolor_change()の直後に打った
                # 最初のSTITCH」に対して自動でJUMPを補ってくれる仕組みのため、
                # ここで自分からmove_abs(JUMP)は呼ばない（呼ぶとJUMPが二重に
                # 入り、.pesでは非first JUMPは全て自動トリムされるため
                # トリムも二重発生してしまう）。
                if not is_first_run_overall and not color_changed:
                    gap_px = ((current_point_px[0] - run[0][0]) ** 2
                              + (current_point_px[1] - run[0][1]) ** 2) ** 0.5
                    if gap_px > trim_threshold_px:
                        pattern.trim()
                    # 閾値以下なら、トリムもJUMPも使わず通常のSTITCHで渡る
                    # （同色の別パーツ同士が近接している場合、浮き糸ジャンプより
                    # 素直に縫い進むほうが仕上がり・トリム回数の両面で有利）
                # color_changed の場合は、直前の pattern.color_change() 呼び出しで
                # 既にトリム相当の状態になっているため、ここで重ねてtrim()しない
                ux0, uy0 = to_units(*run[0])
                pattern.stitch_abs(ux0, uy0)
            else:
                # 同一パーツ内のrun間移動は、JUMPを使わずパーツ内部を通る経路の
                # STITCHで繋ぎ、トリムを回避する（同色領域内なので仕上がりに影響しない）
                bridge = _bridge_within_mask(
                    block.mask_crop, block.bbox_origin, current_point_px, run[0], max_gap_px,
                )
                for bx, by in bridge:
                    ux, uy = to_units(bx, by)
                    pattern.stitch_abs(ux, uy)

            for x, y in run[1:]:
                ux, uy = to_units(x, y)
                pattern.stitch_abs(ux, uy)

            current_point_px = run[-1]
            is_first_run_overall = False
            block_first_run = False

        prev_color = block.color_bgr

    pattern.end()
    return pattern


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="色分解済みイラストから刺しゅうデータ(.pes / .dst)を自動生成する",
    )
    parser.add_argument("input", help="入力画像パス（色分解・減色・境界クッキリ化済みPNG、正方形）")
    parser.add_argument("output", help="出力ファイルパス（.pes または .dst）")
    parser.add_argument(
        "--hoop-size-mm", type=float, default=HOOP_SIZE_MM,
        help=f"刺しゅう枠の一辺のサイズ(mm)。Skitch PP1のデフォルトは{HOOP_SIZE_MM}mm(10cm四方)",
    )
    args = parser.parse_args()

    raw_image = imread_unicode(args.input)
    if raw_image is None:
        print(f"画像を読み込めませんでした: {args.input}", file=sys.stderr)
        sys.exit(1)

    bgr_image, background_mask = split_bgr_and_background_mask(raw_image)
    has_alpha = raw_image.ndim == 3 and raw_image.shape[2] == 4
    print(f"画像を読み込みました: {bgr_image.shape[1]}x{bgr_image.shape[0]}px (alpha={'あり' if has_alpha else 'なし'})")

    px_per_mm = compute_px_per_mm(bgr_image, args.hoop_size_mm)
    print(f"{args.hoop_size_mm}mm×{args.hoop_size_mm}mm 枠にフィッティング → px_per_mm={px_per_mm:.3f}")

    blocks = process_image(bgr_image, background_mask, px_per_mm)
    print(f"検出パーツ数: {len(blocks)}")
    for b in blocks:
        print(f"  color(BGR)={b.color_bgr} type={b.stitch_type} area={b.area_px}px points={len(b.points_px)}")

    if not blocks:
        print("有効なパーツが検出できませんでした。入力画像・背景判定を確認してください。", file=sys.stderr)
        sys.exit(1)

    pattern = build_pattern(blocks, px_per_mm, bgr_image.shape)
    pyembroidery.write(pattern, args.output)
    print(f"書き出し完了: {args.output}")


if __name__ == "__main__":
    main()
