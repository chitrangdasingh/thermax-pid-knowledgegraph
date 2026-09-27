"""Local evidence store and opt-in historian connectors for the Thermax P&ID POC."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DB = BASE / "runtime" / "pid.sqlite3"
REQUIRED = {"timestamp", "sensor_tag", "asset_id", "value", "unit", "quality_code"}


def csv_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def connect(path=DB):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(p)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    return db


SCHEMA = """
CREATE TABLE IF NOT EXISTS drawings (drawing_id TEXT PRIMARY KEY, name TEXT, source_hash TEXT, source_kind TEXT);
CREATE TABLE IF NOT EXISTS assets (asset_id TEXT PRIMARY KEY, tag TEXT, asset_class TEXT, x REAL, y REAL, cad_layer TEXT,
 overall_confidence REAL, evidence_coverage REAL, digital_twin_readiness REAL, review_status TEXT, provenance TEXT);
CREATE TABLE IF NOT EXISTS relationships (edge_id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT REFERENCES assets(asset_id),
 target_id TEXT REFERENCES assets(asset_id), kind TEXT, confidence REAL, evidence TEXT, direction TEXT, review_status TEXT,
 UNIQUE(source_id,target_id,kind));
CREATE TABLE IF NOT EXISTS geometry (geometry_id TEXT PRIMARY KEY, geometry_type TEXT, x REAL, y REAL, cad_layer TEXT);
CREATE TABLE IF NOT EXISTS line_segments (segment_id TEXT PRIMARY KEY, x1 REAL, y1 REAL, x2 REAL, y2 REAL, cad_layer TEXT,
 classification TEXT DEFAULT 'DRAWING_PRIMITIVE', review_status TEXT DEFAULT 'UNCLASSIFIED');
CREATE TABLE IF NOT EXISTS cad_blocks (block_id TEXT PRIMARY KEY, block_name TEXT, x REAL, y REAL, cad_layer TEXT, attributes_json TEXT, drawing_region TEXT,
 review_status TEXT DEFAULT 'UNCLASSIFIED');
CREATE TABLE IF NOT EXISTS tag_mappings (sensor_tag TEXT PRIMARY KEY, asset_id TEXT REFERENCES assets(asset_id), property TEXT,
 unit TEXT, lower_limit REAL, upper_limit REAL, engineer_status TEXT DEFAULT 'PENDING');
CREATE TABLE IF NOT EXISTS observations (event_id TEXT PRIMARY KEY, timestamp TEXT, sensor_tag TEXT, asset_id TEXT REFERENCES assets(asset_id),
 property TEXT, value REAL, unit TEXT, quality_code TEXT, source TEXT, ingested_at TEXT);
CREATE INDEX IF NOT EXISTS obs_asset_time ON observations(asset_id,timestamp);
CREATE TABLE IF NOT EXISTS alerts (alert_id TEXT PRIMARY KEY, event_id TEXT, asset_id TEXT, rule TEXT, severity TEXT,
 measured REAL, threshold REAL, status TEXT DEFAULT 'OPEN', created_at TEXT);
CREATE TABLE IF NOT EXISTS decisions (asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id), symbol_decision TEXT,
 connectivity_decision TEXT, reviewer TEXT, reviewed_at TEXT, comment TEXT);
CREATE TABLE IF NOT EXISTS hierarchy (asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id), enterprise TEXT, site TEXT,
 area TEXT, process_unit TEXT, status TEXT DEFAULT 'DRAFT');
CREATE TABLE IF NOT EXISTS derived_channels (channel_id TEXT PRIMARY KEY, first_tag TEXT, second_tag TEXT,
 operation TEXT, unit TEXT, status TEXT DEFAULT 'DRAFT');
CREATE TABLE IF NOT EXISTS audit (audit_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, action TEXT, source TEXT, details TEXT);
"""


def log(db, action, source, details):
    db.execute("INSERT INTO audit(timestamp,action,source,details) VALUES (?,?,?,?)",
               (datetime.now(timezone.utc).isoformat(), action, source, json.dumps(details, default=str)))


def bootstrap(path=DB, data_dir=DATA):
    data_dir = Path(data_dir)
    db = connect(path)
    db.executescript(SCHEMA)
    summary = json.loads((data_dir / "summary.json").read_text())
    manifest_path=data_dir/"source_manifest.json"
    source_hash=json.loads(manifest_path.read_text()).get("sha256","") if manifest_path.exists() else ""
    provenance = f"DXF:{summary['source_file']}"
    with db:
        db.execute("INSERT OR REPLACE INTO drawings VALUES (?,?,?,?)", ("DRAWING-001", summary["source_file"],
                   source_hash, summary["mode"]))
        for a in csv_rows(data_dir / "asset_master.csv"):
            db.execute("INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (a["asset_id"], a["tag"], a["asset_class"], float(a["x"]), float(a["y"]), a["cad_layer"],
                        float(a["overall_confidence"]), float(a["evidence_coverage"]), float(a["digital_twin_readiness"]),
                        a["review_status"], provenance))
            db.execute("INSERT OR IGNORE INTO hierarchy(asset_id,enterprise,site,area,process_unit,status) VALUES (?,?,?,?,?,?)",
                       (a["asset_id"],"Thermax","RTIC","Single Tube Reactor","Unassigned","DRAFT"))
        for r in csv_rows(data_dir / "relationships.csv"):
            db.execute("INSERT OR REPLACE INTO relationships(source_id,target_id,kind,confidence,evidence,direction,review_status) VALUES (?,?,?,?,?,?,?)",
                       (r["source_id"],r["target_id"],r["relationship"],float(r["confidence"]),r["evidence"],r["direction"],r["review_status"]))
        for g in csv_rows(data_dir / "geometry_inventory.csv"):
            db.execute("INSERT OR REPLACE INTO geometry VALUES (?,?,?,?,?)",
                       (g["geometry_id"],g["geometry_type"],float(g["x"]),float(g["y"]),g["cad_layer"]))
        for s in csv_rows(data_dir / "line_segments.csv"):
            db.execute("INSERT OR IGNORE INTO line_segments(segment_id,x1,y1,x2,y2,cad_layer) VALUES (?,?,?,?,?,?)",
                       (s["segment_id"],float(s["x1"]),float(s["y1"]),float(s["x2"]),float(s["y2"]),s["cad_layer"]))
        block_path=data_dir/"cad_blocks.csv"
        if block_path.exists():
            for b in csv_rows(block_path):
                db.execute("INSERT OR REPLACE INTO cad_blocks(block_id,block_name,x,y,cad_layer,attributes_json,drawing_region) VALUES (?,?,?,?,?,?,?)",
                           (b["block_id"],b["block_name"],float(b["x"]),float(b["y"]),b["cad_layer"],b["attributes_json"],b["drawing_region"]))
        log(db,"BOOTSTRAP","bundled_dxf_results", {"assets":summary["process_assets"],"candidate_edges":summary["preliminary_connections"]})
    db.close()
    return summary


def read_table(name, path=DB, where="", params=(), limit=2000):
    allowed={"drawings","assets","relationships","geometry","line_segments","cad_blocks","tag_mappings","observations","alerts","decisions","hierarchy","derived_channels","audit"}
    if name not in allowed:
        raise ValueError("Unknown table")
    with connect(path) as db:
        return pd.read_sql_query(f"SELECT * FROM {name} {where} LIMIT ?",db,params=params+(limit,))


def upsert_mapping(sensor_tag, asset_id, prop, unit, lower=None, upper=None, path=DB):
    if not sensor_tag or not asset_id or not prop or not unit:
        raise ValueError("sensor_tag, asset_id, property and unit are required")
    with connect(path) as db:
        if db.execute("SELECT 1 FROM assets WHERE asset_id=?",(asset_id,)).fetchone() is None:
            raise ValueError(f"Unknown asset_id: {asset_id}")
        db.execute("INSERT OR REPLACE INTO tag_mappings VALUES (?,?,?,?,?,?, 'PENDING')",
                   (sensor_tag,asset_id,prop,unit,lower,upper))
        log(db,"MAPPING_UPSERT","user",{"sensor_tag":sensor_tag,"asset_id":asset_id})


def approve_mapping(sensor_tag, reviewer, path=DB):
    if not reviewer.strip():raise ValueError("Reviewer identity required")
    with connect(path) as db:
        result=db.execute("UPDATE tag_mappings SET engineer_status='APPROVED' WHERE sensor_tag=?",(sensor_tag,))
        if result.rowcount!=1:raise ValueError("Mapping not found")
        log(db,"MAPPING_APPROVED",reviewer,{"sensor_tag":sensor_tag})


def record_review(asset_id, symbol_decision, connectivity_decision, reviewer, comment="",path=DB):
    allowed={"APPROVED","REJECTED","PENDING"}
    if symbol_decision not in allowed or connectivity_decision not in allowed:raise ValueError("Invalid decision")
    if not reviewer.strip():raise ValueError("Reviewer identity required")
    with connect(path) as db:
        if db.execute("SELECT 1 FROM assets WHERE asset_id=?",(asset_id,)).fetchone() is None:raise ValueError("Asset not found")
        db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)",(asset_id,symbol_decision,connectivity_decision,reviewer,
            datetime.now(timezone.utc).isoformat(),comment))
        log(db,"ASSET_REVIEW",reviewer,{"asset_id":asset_id,"symbol":symbol_decision,"connectivity":connectivity_decision})


def add_manual_asset(tag, asset_class, x, y, reviewer, path=DB):
    if not tag.strip() or not asset_class.strip() or not reviewer.strip():raise ValueError("Tag, class and reviewer required")
    with connect(path) as db:
        if db.execute("SELECT 1 FROM assets WHERE UPPER(tag)=UPPER(?)",(tag.strip(),)).fetchone():raise ValueError("Tag already exists")
        current=db.execute("SELECT asset_id FROM assets WHERE asset_id LIKE 'MAN-%' ORDER BY asset_id DESC LIMIT 1").fetchone()
        number=int(current[0].split('-')[1])+1 if current else 1
        aid=f"MAN-{number:04d}"
        db.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (aid,tag.strip(),asset_class.strip(),float(x),float(y),"MANUAL",0.0,0.0,0.0,"ENGINEERING_REVIEW_REQUIRED","MANUAL:"+reviewer))
        db.execute("INSERT INTO hierarchy(asset_id,enterprise,site,area,process_unit,status) VALUES (?,?,?,?,?,?)",
            (aid,"Thermax","RTIC","Single Tube Reactor","Unassigned","DRAFT"))
        log(db,"MANUAL_ASSET_CREATE",reviewer,{"asset_id":aid,"tag":tag})
    return aid


def set_hierarchy(asset_id,enterprise,site,area,process_unit,reviewer,path=DB):
    if not all(s.strip() for s in [enterprise,site,area,process_unit,reviewer]):raise ValueError("All hierarchy fields and reviewer required")
    with connect(path) as db:
        result=db.execute("UPDATE hierarchy SET enterprise=?,site=?,area=?,process_unit=?,status='REVIEWED' WHERE asset_id=?",
                          (enterprise,site,area,process_unit,asset_id))
        if result.rowcount!=1:raise ValueError("Asset hierarchy record missing")
        log(db,"HIERARCHY_REVIEW",reviewer,{"asset_id":asset_id,"unit":process_unit})


def add_derived_channel(channel_id,first_tag,second_tag,path=DB):
    if not channel_id.strip() or first_tag==second_tag:raise ValueError("Unique channel ID and two different tags required")
    with connect(path) as db:
        rows=db.execute("SELECT sensor_tag,unit,engineer_status FROM tag_mappings WHERE sensor_tag IN (?,?)",(first_tag,second_tag)).fetchall()
        if len(rows)!=2 or any(r['engineer_status']!='APPROVED' for r in rows):raise ValueError("Both inputs need approved mappings")
        if len({r['unit'] for r in rows})!=1:raise ValueError("Inputs require matching units")
        db.execute("INSERT INTO derived_channels(channel_id,first_tag,second_tag,operation,unit,status) VALUES (?,?,?,'DIFFERENCE',?,'DRAFT')",
                   (channel_id,first_tag,second_tag,rows[0]['unit']))
        log(db,"DERIVED_CHANNEL_CREATE","user",{"channel_id":channel_id})


def compute_derived(channel_id,path=DB):
    with connect(path) as db:
        c=db.execute("SELECT * FROM derived_channels WHERE channel_id=?",(channel_id,)).fetchone()
        if not c:raise ValueError("Unknown derived channel")
        samples=[]
        for tag in [c['first_tag'],c['second_tag']]:
            row=db.execute("SELECT * FROM observations WHERE sensor_tag=? AND quality_code='GOOD' ORDER BY timestamp DESC LIMIT 1",(tag,)).fetchone()
            if row is None:return {"status":"NO_MATCHED_MEASUREMENTS","channel_id":channel_id}
            samples.append(row)
        dt=abs((pd.Timestamp(samples[0]['timestamp'])-pd.Timestamp(samples[1]['timestamp'])).total_seconds())
        if dt>300:return {"status":"TIMESTAMPS_NOT_ALIGNED","seconds_apart":dt,"tolerance_seconds":300}
        return {"status":"DERIVED","channel_id":channel_id,"value":samples[0]['value']-samples[1]['value'],
                "unit":c['unit'],"timestamp":max(r['timestamp'] for r in samples),
                "source_event_ids":[r['event_id'] for r in samples],"operation":"A_MINUS_B"}


def ingest_observations(frame, source="historian_csv", path=DB):
    """Reject unmapped/invalid data; idempotent event IDs. No fabricated telemetry."""
    missing=REQUIRED-set(frame.columns)
    if missing:raise ValueError("Missing columns: "+", ".join(sorted(missing)))
    accepted=0; rejected=[]
    with connect(path) as db:
        mappings={r["sensor_tag"]:r for r in db.execute("SELECT * FROM tag_mappings")}
        for number,row in enumerate(frame.to_dict("records"),start=2):
            try:
                tag=str(row["sensor_tag"]).strip(); asset=str(row["asset_id"]).strip()
                m=mappings.get(tag)
                if not m or m["asset_id"]!=asset: raise ValueError("Sensor and asset are not in approved mapping table")
                if str(m["engineer_status"]).upper()!="APPROVED":raise ValueError("Mapping awaiting engineer approval")
                if str(row["unit"]).strip()!=m["unit"]:raise ValueError("Unit differs from mapping")
                t=pd.to_datetime(row["timestamp"],utc=True,errors="raise")
                if pd.isna(t):raise ValueError("Invalid timestamp")
                value=float(row["value"])
                if not math.isfinite(value):raise ValueError("Nonfinite reading")
                quality=str(row["quality_code"]).strip().upper()
                if quality not in {"GOOD","BAD","UNCERTAIN"}:raise ValueError("quality_code must be GOOD, BAD or UNCERTAIN")
                ident=hashlib.sha256(f"{source}|{t.isoformat()}|{tag}|{value}|{quality}".encode()).hexdigest()
                before=db.total_changes
                db.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (ident,t.isoformat(),tag,asset,m["property"],value,m["unit"],quality,source,datetime.now(timezone.utc).isoformat()))
                accepted+=db.total_changes-before
            except (ValueError,TypeError,OverflowError) as ex:
                rejected.append({"row":number,"reason":str(ex)})
        log(db,"OBSERVATIONS_INGEST",source,{"accepted":accepted,"rejected":len(rejected)})
    return {"accepted":accepted,"rejected":rejected}


def evaluate_rules(path=DB):
    """Threshold rules for approved, GOOD, mapped telemetry only."""
    created=0
    with connect(path) as db:
        q="""SELECT o.*, m.lower_limit, m.upper_limit FROM observations o JOIN tag_mappings m
        ON o.sensor_tag=m.sensor_tag AND o.asset_id=m.asset_id WHERE o.quality_code='GOOD' AND m.engineer_status='APPROVED'"""
        for row in db.execute(q).fetchall():
            for rule,limit,test in [("LOW",row["lower_limit"],lambda a,b:a<b),("HIGH",row["upper_limit"],lambda a,b:a>b)]:
                if limit is None or not test(row["value"],limit):continue
                alert_id=hashlib.sha256(f"{row['event_id']}|{rule}".encode()).hexdigest()
                before=db.total_changes
                db.execute("INSERT OR IGNORE INTO alerts(alert_id,event_id,asset_id,rule,severity,measured,threshold,created_at) VALUES (?,?,?,?,?,?,?,?)",
                           (alert_id,row["event_id"],row["asset_id"],rule,"WARNING",row["value"],limit,datetime.now(timezone.utc).isoformat()))
                created+=db.total_changes-before
        log(db,"RULE_EVALUATION","thresholds",{"created":created})
    return created


def asset_context(asset_id, path=DB):
    with connect(path) as db:
        asset=db.execute("SELECT * FROM assets WHERE asset_id=?",(asset_id,)).fetchone()
        if not asset:return None
        edges=[dict(r) for r in db.execute("SELECT * FROM relationships WHERE source_id=? OR target_id=?",(asset_id,asset_id))]
        obs=[dict(r) for r in db.execute("SELECT * FROM observations WHERE asset_id=? ORDER BY timestamp DESC LIMIT 30",(asset_id,))]
        alerts=[dict(r) for r in db.execute("SELECT * FROM alerts WHERE asset_id=? ORDER BY created_at DESC LIMIT 30",(asset_id,))]
        mappings=[dict(r) for r in db.execute("SELECT * FROM tag_mappings WHERE asset_id=?",(asset_id,))]
        return {"asset":dict(asset),"relationships":edges,"observations":obs,"alerts":alerts,"mappings":mappings}


def summarize_asset(asset_id,path=DB):
    """Grounded, deterministic copilot response. No external LLM calls."""
    ctx=asset_context(asset_id,path)
    if ctx is None:return {"answer":"Unknown asset ID","sources":[]}
    a=ctx["asset"]
    text=f"{a['tag']} is classified as {a['asset_class']} from DXF tag and nearby geometry. Evidence score {a['overall_confidence']:.1%}. "
    text+=f"{len(ctx['relationships'])} candidate connections await engineering review. "
    if ctx["observations"]:
        o=ctx["observations"][0];text+=f"Latest mapped reading: {o['value']} {o['unit']} at {o['timestamp']} ({o['quality_code']}). "
    else:text+="No approved historian observations are available. "
    text+=f"{len(ctx['alerts'])} logged alerts."
    return {"answer":text,"sources":[a["provenance"]]+[f"event:{o['event_id']}" for o in ctx["observations"][:1]],"review_status":a["review_status"]}


def namespaces(path=DB):
    """Simple unified namespace projection; IDs stay stable across adapters."""
    with connect(path) as db:
        return [{"topic":f"thermax/rtic/reactor_pid/{r['asset_id']}/{r['property']}","sensor_tag":r["sensor_tag"],
                 "unit":r["unit"],"approved":r["engineer_status"]=="APPROVED"} for r in db.execute("SELECT * FROM tag_mappings")]
