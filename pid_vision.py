"""YOLOv5 + Tesseract P&ID inference, tag association and review flags.

Only engineer-approved custom P&ID weights are accepted. Generic COCO weights are
not a substitute for P&ID symbol training. Supply a local YOLOv5 clone through
``YOLOV5_REPO`` or let PyTorch Hub load the official Ultralytics repository.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pandas as pd


TAG_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{1,5}(?:[-_/ ]?[A-Z]{0,3})?[-_/ ]?\d{1,4}[A-Z]?(?![A-Z0-9])", re.I)


def _normalise_tag(value: str) -> str:
    value = re.sub(r"\s+", "-", value.strip().upper()).replace("_", "-").replace("/", "-")
    return re.sub(r"-+", "-", value)


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union


def _nms(rows: list[dict], threshold: float = 0.5) -> list[dict]:
    keep: list[dict] = []
    for row in sorted(rows, key=lambda x: x["symbol_confidence"], reverse=True):
        box = [row[k] for k in ("x1", "y1", "x2", "y2")]
        if all(
            row["symbol_class"] != old["symbol_class"]
            or _iou(box, [old[k] for k in ("x1", "y1", "x2", "y2")]) < threshold
            for old in keep
        ):
            keep.append(row)
    return keep


def _load_yolov5(weights: str | Path, yolov5_repo: str | Path | None = None):
    import torch

    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError("Engineer-approved custom YOLOv5 P&ID weights (best.pt) are required")
    repo_text = str(yolov5_repo or os.getenv("YOLOV5_REPO", "")).strip()
    repo = Path(repo_text).expanduser() if repo_text else None
    if repo and (repo / "hubconf.py").is_file():
        return torch.hub.load(str(repo.resolve()), "custom", path=str(weights), source="local", trust_repo=True)
    return torch.hub.load("ultralytics/yolov5", "custom", path=str(weights), trust_repo=True)


def detect_symbols(
    image_path: str | Path,
    weights: str | Path,
    conf: float = 0.15,
    tile: int = 1280,
    overlap: float = 0.2,
    yolov5_repo: str | Path | None = None,
) -> pd.DataFrame:
    """Run custom YOLOv5 on overlapping tiles and de-duplicate page coordinates."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    model = _load_yolov5(weights, yolov5_repo)
    model.conf = float(conf)
    model.iou = 0.45
    model.max_det = 3000
    h, w = image.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    rows: list[dict] = []
    names = model.names
    for y in range(0, h, step):
        for x in range(0, w, step):
            crop = image[y : min(y + tile, h), x : min(x + tile, w)]
            if crop.shape[0] < 64 or crop.shape[1] < 64:
                continue
            prediction = model(crop, size=tile)
            for x1, y1, x2, y2, score, class_id in prediction.xyxy[0].detach().cpu().tolist():
                class_id = int(class_id)
                label = names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, str(class_id))
                rows.append(
                    {
                        "symbol_class": str(label), "class_id": class_id,
                        "symbol_confidence": float(score), "x1": round(x1 + x), "y1": round(y1 + y),
                        "x2": round(x2 + x), "y2": round(y2 + y),
                    }
                )
    columns = ["symbol_class", "class_id", "symbol_confidence", "x1", "y1", "x2", "y2"]
    return pd.DataFrame(_nms(rows), columns=columns)


def _line_rows_from_tesseract(image, orientation: int, language: str, min_conf: float) -> list[dict]:
    import cv2
    import pytesseract

    working = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) if orientation == 90 else image
    original_h, original_w = image.shape[:2]
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    data = pytesseract.image_to_data(
        threshold, lang=language, config="--oem 1 --psm 11 -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DATAFRAME,
    )
    if data is None or data.empty:
        return []
    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[(data["text"] != "") & (pd.to_numeric(data["conf"], errors="coerce") >= 0)]
    rows: list[dict] = []
    for _, group in data.groupby(["page_num", "block_num", "par_num", "line_num"], sort=False):
        line_text = " ".join(group["text"].tolist())
        matches = [_normalise_tag(m.group(0)) for m in TAG_RE.finditer(line_text)]
        if not matches:
            continue
        confidence = float(pd.to_numeric(group["conf"], errors="coerce").mean()) / 100.0
        if confidence < min_conf:
            continue
        x1 = float(group["left"].min()) / 2
        y1 = float(group["top"].min()) / 2
        x2 = float((group["left"] + group["width"]).max()) / 2
        y2 = float((group["top"] + group["height"]).max()) / 2
        if orientation == 90:
            x1, x2, y1, y2 = max(0, y1), min(original_w, y2), max(0, original_h - x2), min(original_h, original_h - x1)
        rows.append(
            {
                "ocr_text": line_text, "ocr_confidence": confidence,
                "candidate_tag": "|".join(dict.fromkeys(matches)), "orientation": orientation,
                "tx1": round(x1), "ty1": round(y1), "tx2": round(x2), "ty2": round(y2),
            }
        )
    return rows


def read_text(image_path: str | Path, language: str = "eng", min_conf: float = 0.25, include_vertical: bool = True) -> pd.DataFrame:
    """Read horizontal and optional vertical P&ID tags with Tesseract 5."""
    import cv2
    import pytesseract

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError("Tesseract binary is missing. In Colab run: apt-get install -y tesseract-ocr") from exc
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    rows = _line_rows_from_tesseract(image, 0, language, min_conf)
    if include_vertical:
        rows.extend(_line_rows_from_tesseract(image, 90, language, min_conf))
    rows = sorted(rows, key=lambda row: row["ocr_confidence"], reverse=True)
    keep: list[dict] = []
    for row in rows:
        box = [row[k] for k in ("tx1", "ty1", "tx2", "ty2")]
        tags = set(row["candidate_tag"].split("|"))
        if not any(tags & set(old["candidate_tag"].split("|")) and _iou(box, [old[k] for k in ("tx1", "ty1", "tx2", "ty2")]) > 0.35 for old in keep):
            keep.append(row)
    columns = ["ocr_text", "ocr_confidence", "candidate_tag", "orientation", "tx1", "ty1", "tx2", "ty2"]
    return pd.DataFrame(keep, columns=columns)


def _point_to_box_distance(px: float, py: float, box) -> float:
    x1, y1, x2, y2 = box
    return math.hypot(max(x1 - px, 0, px - x2), max(y1 - py, 0, py - y2))


def _confidence_level(value) -> str:
    if pd.isna(value): return "NOT_AVAILABLE"
    if value >= 0.8: return "HIGH"
    if value >= 0.55: return "MEDIUM"
    return "LOW"


def associate(symbols: pd.DataFrame, text: pd.DataFrame, max_distance: float = 220) -> pd.DataFrame:
    """One-to-one, nearest feasible association between symbol boxes and valid tags."""
    symbol_records = list(symbols.reset_index(drop=True).to_dict("records"))
    text_records = list(text[text["candidate_tag"].astype(str).str.len() > 0].reset_index(drop=True).to_dict("records"))
    candidates = []
    for si, symbol in enumerate(symbol_records):
        for ti, tag in enumerate(text_records):
            tcx, tcy = (tag["tx1"] + tag["tx2"]) / 2, (tag["ty1"] + tag["ty2"]) / 2
            distance = _point_to_box_distance(tcx, tcy, [symbol[k] for k in ("x1", "y1", "x2", "y2")])
            if distance <= max_distance: candidates.append((distance, si, ti))
    matches, used_symbols, used_text = {}, set(), set()
    for distance, si, ti in sorted(candidates):
        if si not in used_symbols and ti not in used_text:
            matches[si] = (ti, distance); used_symbols.add(si); used_text.add(ti)

    output: list[dict] = []
    for si, symbol in enumerate(symbol_records):
        tag_row = text_records[matches[si][0]] if si in matches else None
        distance = matches[si][1] if si in matches else None
        if tag_row:
            association = max(0.0, 1.0 - float(distance) / max_distance)
            ocr_conf = float(tag_row["ocr_confidence"])
            overall = float((max(symbol["symbol_confidence"], 1e-6) * max(ocr_conf, 1e-6) * max(association, 1e-6)) ** (1 / 3))
            result_type, tag = "TAGGED_ASSET", tag_row["candidate_tag"].split("|")[0]
            text_box = {k: tag_row[k] for k in ("tx1", "ty1", "tx2", "ty2")}
        else:
            association, ocr_conf, overall = 0.0, None, float(symbol["symbol_confidence"])
            result_type, tag = "SYMBOL_ONLY", ""
            text_box = {k: None for k in ("tx1", "ty1", "tx2", "ty2")}
        output.append(
            {
                "detection_id": f"DET-{len(output) + 1:04d}", "result_type": result_type,
                "symbol_class": symbol["symbol_class"], "tag": tag,
                "symbol_confidence": symbol["symbol_confidence"], "ocr_confidence": ocr_conf,
                "association_confidence": association, "overall_confidence": overall,
                "confidence_level": _confidence_level(overall), "association_distance_px": distance,
                **{k: symbol[k] for k in ("x1", "y1", "x2", "y2")}, **text_box,
            }
        )
    for ti, tag_row in enumerate(text_records):
        if ti in used_text: continue
        output.append(
            {
                "detection_id": f"TXT-{len(output) + 1:04d}", "result_type": "TEXT_ONLY",
                "symbol_class": "Expected symbol", "tag": tag_row["candidate_tag"].split("|")[0],
                "symbol_confidence": None, "ocr_confidence": tag_row["ocr_confidence"],
                "association_confidence": 0.0, "overall_confidence": tag_row["ocr_confidence"],
                "confidence_level": _confidence_level(tag_row["ocr_confidence"]), "association_distance_px": None,
                "x1": tag_row["tx1"], "y1": tag_row["ty1"], "x2": tag_row["tx2"], "y2": tag_row["ty2"],
                **{k: tag_row[k] for k in ("tx1", "ty1", "tx2", "ty2")},
            }
        )
    columns = ["detection_id", "result_type", "symbol_class", "tag", "symbol_confidence", "ocr_confidence",
               "association_confidence", "overall_confidence", "confidence_level", "association_distance_px",
               "x1", "y1", "x2", "y2", "tx1", "ty1", "tx2", "ty2"]
    return pd.DataFrame(output, columns=columns)


def find_flaws(results: pd.DataFrame, image_shape) -> pd.DataFrame:
    h, w = image_shape[:2]; flaws: list[dict] = []
    def add(row, kind: str, severity: str, evidence: str) -> None:
        flaws.append({"flaw_id": f"FLW-{len(flaws)+1:04d}", "detection_id": row.detection_id, "tag": row.tag,
                      "symbol_class": row.symbol_class, "flaw_type": kind, "severity": severity,
                      "evidence": evidence, "review_status": "ENGINEERING_REVIEW_REQUIRED"})
    for row in results.itertuples(index=False):
        if row.result_type == "SYMBOL_ONLY": add(row, "MISSING_OR_UNASSOCIATED_TAG", "HIGH", "YOLOv5 symbol has no valid nearby Tesseract tag")
        if row.result_type == "TEXT_ONLY": add(row, "TAG_WITHOUT_SYMBOL", "HIGH", "Tesseract tag has no associated YOLOv5 symbol")
        if pd.notna(row.symbol_confidence) and row.symbol_confidence < 0.4: add(row, "LOW_SYMBOL_CONFIDENCE", "MEDIUM", f"YOLOv5 confidence {row.symbol_confidence:.3f}")
        if pd.notna(row.ocr_confidence) and row.ocr_confidence < 0.6: add(row, "LOW_OCR_CONFIDENCE", "MEDIUM", f"Tesseract confidence {row.ocr_confidence:.3f}")
        if row.result_type == "TAGGED_ASSET" and row.association_confidence < 0.4: add(row, "WEAK_TAG_SYMBOL_ASSOCIATION", "MEDIUM", f"Association confidence {row.association_confidence:.3f}")
        if min(row.x1, row.y1) <= 2 or row.x2 >= w - 2 or row.y2 >= h - 2: add(row, "CLIPPED_AT_IMAGE_BOUNDARY", "MEDIUM", "Box touches page or tile boundary")
    tagged = results[results["tag"].astype(str).str.len() > 0]
    for tag, count in tagged["tag"].value_counts().items():
        if count > 1:
            for row in tagged[tagged["tag"].eq(tag)].itertuples(index=False): add(row, "DUPLICATE_TAG", "HIGH", f"Tag appears {count} times")
    return pd.DataFrame(flaws, columns=["flaw_id", "detection_id", "tag", "symbol_class", "flaw_type", "severity", "evidence", "review_status"])


def annotate(image_path: str | Path, results: pd.DataFrame, flaws: pd.DataFrame, output_path: str | Path) -> None:
    import cv2
    image = cv2.imread(str(image_path)); flaw_ids = set(flaws["detection_id"]) if len(flaws) else set()
    palette = {"TAGGED_ASSET": (42, 170, 42), "SYMBOL_ONLY": (0, 0, 220), "TEXT_ONLY": (255, 120, 0)}
    for row in results.itertuples(index=False):
        color = (0, 140, 255) if row.detection_id in flaw_ids else palette.get(row.result_type, (150, 0, 150))
        cv2.rectangle(image, (int(row.x1), int(row.y1)), (int(row.x2), int(row.y2)), color, 2)
        if row.result_type == "TAGGED_ASSET" and pd.notna(row.tx1):
            cv2.rectangle(image, (int(row.tx1), int(row.ty1)), (int(row.tx2), int(row.ty2)), (255, 90, 0), 1)
            cv2.line(image, (int((row.x1+row.x2)/2), int((row.y1+row.y2)/2)), (int((row.tx1+row.tx2)/2), int((row.ty1+row.ty2)/2)), (42, 170, 42), 1)
        label = f"{row.tag or row.symbol_class} | {row.result_type} | {row.overall_confidence:.2f}"
        cv2.putText(image, label, (int(row.x1), max(15, int(row.y1)-4)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image)


def run(image_path: str | Path, weights: str | Path, output_dir: str | Path, conf: float = 0.15, yolov5_repo: str | Path | None = None) -> dict:
    import cv2
    from asset_hierarchy import build_from_vision
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None: raise ValueError(f"Cannot read image: {image_path}")
    symbols = detect_symbols(image_path, weights, conf=conf, yolov5_repo=yolov5_repo)
    text = read_text(image_path)
    results = associate(symbols, text, max_distance=max(220, int(math.hypot(*image.shape[:2]) * .025)))
    flaws = find_flaws(results, image.shape)
    hierarchy_input = results.rename(columns={"detection_id": "asset_id", "symbol_class": "detected_name"}).copy()
    flagged = set(flaws["detection_id"].astype(str)) if len(flaws) else set()
    hierarchy_input["flaw_flag"] = hierarchy_input["asset_id"].astype(str).apply(lambda value: "REVIEW" if value in flagged else "PASS")
    hierarchy = build_from_vision(hierarchy_input)
    symbols.to_csv(out/"symbol_detections.csv", index=False); text.to_csv(out/"ocr_text.csv", index=False)
    results.to_csv(out/"detection_results.csv", index=False); flaws.to_csv(out/"flaw_report.csv", index=False)
    hierarchy.to_csv(out/"asset_hierarchy.csv", index=False); annotate(image_path, results, flaws, out/"annotated_pid.png")
    summary = {"detector": "YOLOv5 custom weights", "ocr": "Tesseract 5", "symbols": len(symbols),
               "ocr_tag_regions": len(text), "tagged_assets": int((results.result_type=="TAGGED_ASSET").sum()) if len(results) else 0,
               "symbol_only": int((results.result_type=="SYMBOL_ONLY").sum()) if len(results) else 0,
               "text_only": int((results.result_type=="TEXT_ONLY").sum()) if len(results) else 0,
               "flaws_for_review": len(flaws), "weights": str(weights),
               "flaw_scope": "Visible detection, OCR and association checks only; process design and topology require engineering review."}
    (out/"inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
