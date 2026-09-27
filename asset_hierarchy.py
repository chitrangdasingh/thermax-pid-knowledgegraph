"""Create an auditable asset hierarchy from DXF or vision inference outputs.

The current DXF hierarchy is intentionally conservative. Site, area and process-unit
fields remain explicit engineering inputs, while the drawing/system title and asset
rows are source-backed. CNN confidence is never substituted with DXF rule confidence.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


DEFAULT_CONTEXT = {
    "enterprise_id": "ENT-THERMAX",
    "enterprise_name": "Thermax",
    "site_id": "SITE-UNASSIGNED",
    "site_name": "Engineering input required",
    "area_id": "AREA-UNASSIGNED",
    "area_name": "Engineering input required",
    "process_unit_id": "UNIT-UNASSIGNED",
    "process_unit_name": "Engineering input required",
    "system_id": "SYS-STR-001",
    "system_name": "Single Tube Reactor System",
    "drawing_id": "PID-THERMAX-STR-001",
}


def _relationship_metrics(assets: pd.DataFrame, relationships: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame({"asset_id": assets["asset_id"].astype(str)})
    if relationships.empty:
        metrics["candidate_connection_count"] = 0
        metrics["mean_topology_confidence"] = pd.NA
        metrics["unknown_direction_count"] = 0
        return metrics
    endpoints = pd.concat(
        [
            relationships.rename(columns={"source_id": "asset_id"}),
            relationships.rename(columns={"target_id": "asset_id"}),
        ],
        ignore_index=True,
    )
    grouped = endpoints.groupby("asset_id", dropna=False).agg(
        candidate_connection_count=("relationship", "size"),
        mean_topology_confidence=("confidence", "mean"),
        unknown_direction_count=("direction", lambda s: int((s.astype(str).str.upper() == "UNKNOWN").sum())),
    )
    return metrics.merge(grouped.reset_index(), on="asset_id", how="left").fillna(
        {"candidate_connection_count": 0, "unknown_direction_count": 0}
    )


def build_from_dxf(
    assets: pd.DataFrame,
    relationships: pd.DataFrame,
    context: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx = {**DEFAULT_CONTEXT, **(context or {})}
    metrics = _relationship_metrics(assets, relationships)
    hierarchy = assets.merge(metrics, on="asset_id", how="left")

    hierarchy.insert(0, "enterprise_id", ctx["enterprise_id"])
    hierarchy.insert(1, "enterprise_name", ctx["enterprise_name"])
    hierarchy.insert(2, "site_id", ctx["site_id"])
    hierarchy.insert(3, "site_name", ctx["site_name"])
    hierarchy.insert(4, "area_id", ctx["area_id"])
    hierarchy.insert(5, "area_name", ctx["area_name"])
    hierarchy.insert(6, "process_unit_id", ctx["process_unit_id"])
    hierarchy.insert(7, "process_unit_name", ctx["process_unit_name"])
    hierarchy.insert(8, "system_id", ctx["system_id"])
    hierarchy.insert(9, "system_name", ctx["system_name"])
    hierarchy.insert(10, "drawing_id", ctx["drawing_id"])
    hierarchy.insert(11, "parent_id", ctx["system_id"])
    hierarchy.insert(12, "hierarchy_level", "ASSET")
    hierarchy["hierarchy_path"] = (
        hierarchy["enterprise_id"] + "/" + hierarchy["site_id"] + "/" + hierarchy["area_id"]
        + "/" + hierarchy["process_unit_id"] + "/" + hierarchy["system_id"] + "/" + hierarchy["asset_id"]
    )
    hierarchy["object_status"] = "TAGGED_ASSET_CANDIDATE"
    hierarchy["tag_status"] = "NATIVE_DXF_TEXT"
    hierarchy["symbol_status"] = "CLASS_INFERRED_FROM_TAG_PREFIX_AND_LOCAL_GEOMETRY"
    hierarchy["cnn_symbol_confidence"] = pd.NA
    hierarchy["association_confidence"] = pd.NA
    hierarchy["model_stage"] = "CNN_NOT_TRAINED"
    hierarchy["engineering_action"] = "Confirm site/area/unit, symbol box/class, and process connectivity"

    ordered = [
        "enterprise_id", "enterprise_name", "site_id", "site_name", "area_id", "area_name",
        "process_unit_id", "process_unit_name", "system_id", "system_name", "drawing_id",
        "parent_id", "hierarchy_level", "asset_id", "tag", "asset_class", "object_status",
        "tag_status", "symbol_status", "cnn_symbol_confidence", "text_confidence",
        "association_confidence", "overall_confidence", "confidence_level", "evidence_coverage",
        "digital_twin_readiness", "readiness_level", "candidate_connection_count",
        "mean_topology_confidence", "unknown_direction_count", "cad_layer", "x", "y",
        "model_stage", "review_status", "engineering_action", "hierarchy_path", "confidence_basis",
    ]
    hierarchy = hierarchy[ordered].sort_values(["asset_class", "tag"], kind="stable").reset_index(drop=True)

    nodes = pd.DataFrame(
        [
            {"node_id": ctx["enterprise_id"], "node_name": ctx["enterprise_name"], "node_type": "ENTERPRISE", "parent_id": ""},
            {"node_id": ctx["site_id"], "node_name": ctx["site_name"], "node_type": "SITE", "parent_id": ctx["enterprise_id"]},
            {"node_id": ctx["area_id"], "node_name": ctx["area_name"], "node_type": "AREA", "parent_id": ctx["site_id"]},
            {"node_id": ctx["process_unit_id"], "node_name": ctx["process_unit_name"], "node_type": "PROCESS_UNIT", "parent_id": ctx["area_id"]},
            {"node_id": ctx["system_id"], "node_name": ctx["system_name"], "node_type": "SYSTEM", "parent_id": ctx["process_unit_id"]},
        ]
    )
    asset_nodes = hierarchy[["asset_id", "tag", "parent_id"]].rename(
        columns={"asset_id": "node_id", "tag": "node_name"}
    )
    asset_nodes.insert(2, "node_type", "ASSET")
    nodes = pd.concat([nodes, asset_nodes], ignore_index=True)
    return hierarchy, nodes


def build_from_vision(detections: pd.DataFrame, context: dict | None = None) -> pd.DataFrame:
    """Convert post-training vision results into the same hierarchy contract."""
    ctx = {**DEFAULT_CONTEXT, **(context or {})}
    df = detections.copy()
    if "asset_id" not in df:
        df["asset_id"] = [f"VIS-{i:05d}" for i in range(1, len(df) + 1)]
    defaults = {
        "tag": "", "detected_name": "Unknown", "result_type": "UNRESOLVED",
        "symbol_confidence": pd.NA, "ocr_confidence": pd.NA,
        "association_confidence": pd.NA, "flaw_flag": "REVIEW",
    }
    for col, value in defaults.items():
        if col not in df:
            df[col] = value
    df["enterprise_id"] = ctx["enterprise_id"]
    df["enterprise_name"] = ctx["enterprise_name"]
    df["site_id"] = ctx["site_id"]
    df["site_name"] = ctx["site_name"]
    df["area_id"] = ctx["area_id"]
    df["area_name"] = ctx["area_name"]
    df["process_unit_id"] = ctx["process_unit_id"]
    df["process_unit_name"] = ctx["process_unit_name"]
    df["system_id"] = ctx["system_id"]
    df["system_name"] = ctx["system_name"]
    df["drawing_id"] = ctx["drawing_id"]
    df["parent_id"] = ctx["system_id"]
    df["hierarchy_level"] = "ASSET_CANDIDATE"
    df["asset_class"] = df["detected_name"]
    df["object_status"] = df["result_type"]
    df["tag_status"] = df["tag"].astype(str).apply(lambda x: "OCR_TAG" if x.strip() else "TAG_MISSING")
    df["symbol_status"] = df["symbol_confidence"].apply(lambda x: "CNN_DETECTED" if pd.notna(x) else "SYMBOL_MISSING")
    df["review_status"] = df["flaw_flag"].astype(str).apply(
        lambda x: "ENGINEERING_REVIEW_REQUIRED" if x.upper() not in {"", "NONE", "PASS"} else "READY_FOR_APPROVAL"
    )
    df["hierarchy_path"] = (
        df["enterprise_id"] + "/" + df["site_id"] + "/" + df["area_id"] + "/"
        + df["process_unit_id"] + "/" + df["system_id"] + "/" + df["asset_id"]
    )
    return df[
        [
            "enterprise_id", "enterprise_name", "site_id", "site_name", "area_id", "area_name",
            "process_unit_id", "process_unit_name", "system_id", "system_name", "drawing_id", "parent_id",
            "hierarchy_level", "asset_id", "tag", "asset_class", "object_status", "tag_status", "symbol_status",
            "symbol_confidence", "ocr_confidence", "association_confidence", "flaw_flag", "review_status",
            "hierarchy_path",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hierarchy, nodes = build_from_dxf(
        pd.read_csv(data_dir / "asset_master.csv"),
        pd.read_csv(data_dir / "relationships.csv"),
    )
    hierarchy.to_csv(out / "asset_hierarchy.csv", index=False)
    nodes.to_csv(out / "hierarchy_nodes.csv", index=False)
    print({"assets": len(hierarchy), "nodes": len(nodes), "output_dir": str(out)})


if __name__ == "__main__":
    main()
