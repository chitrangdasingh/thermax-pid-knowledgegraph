"""Flattened-vector ingestion for the supplied Thermax P&ID DXF.

The drawing's process symbols are exploded geometry. Tags are therefore the
primary asset anchors and nearby vector primitives provide symbol evidence.
All inferred topology is explicitly marked for engineering review.
"""
from __future__ import annotations

import json, math, re
from collections import Counter
from pathlib import Path

import ezdxf
import networkx as nx
import numpy as np
import pandas as pd
from ezdxf import bbox

TAG_RE = re.compile(
    r"\b(?:HE|HX|FRN|MFC|FIC|PIC|TIC|LIC|PSV|PRV|FCV|PCV|TCV|LCV|"
    r"CH|LG|SM|FI|FT|PI|PT|TI|TT|LI|LT|XV|FV|TV|PV|SV|RV|P|V|R|E|C|K|F|M|I)"
    r"[-_ ]?\d{2,4}[A-Z]?\b", re.I)

CLASS = {
    "HE":"Heat Exchanger","HX":"Heat Exchanger","FRN":"Furnace","R":"Reactor",
    "V":"Vessel","P":"Pump","SM":"Static Mixer","CH":"Chiller","LG":"Level Gauge",
    "MFC":"Mass Flow Controller","FI":"Flow Indicator","FT":"Flow Transmitter",
    "FIC":"Flow Controller","PI":"Pressure Indicator","PT":"Pressure Transmitter",
    "PIC":"Pressure Controller","TI":"Temperature Indicator","TT":"Temperature Transmitter",
    "TIC":"Temperature Controller","LI":"Level Indicator","LT":"Level Transmitter",
    "LIC":"Level Controller","PSV":"Safety Valve","PRV":"Pressure Regulator",
    "XV":"On-Off Valve","FV":"Flow Control Valve","TV":"Temperature Control Valve",
    "PV":"Pressure Control Valve","SV":"Solenoid Valve","RV":"Relief Valve",
    "I":"Interlock","M":"Motor"
}

def clean(v): return re.sub(r"\s+", " ", str(v).replace("\\P", " ")).strip()
def cls(tag):
    m=re.match(r"[A-Z]+",tag.upper()); return CLASS.get(m.group(0),"Tagged Component") if m else "Tagged Component"

def point_segment_dist(px,py,s):
    x1=s.x1.to_numpy(float); y1=s.y1.to_numpy(float); x2=s.x2.to_numpy(float); y2=s.y2.to_numpy(float)
    dx=x2-x1; dy=y2-y1; den=dx*dx+dy*dy
    u=np.where(den>0,((px-x1)*dx+(py-y1)*dy)/den,0); u=np.clip(u,0,1)
    return np.hypot(px-(x1+u*dx),py-(y1+u*dy))

def run(dxf_path, output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    doc=ezdxf.readfile(str(dxf_path)); msp=doc.modelspace()

    text=[]; raw_tags=[]
    for e in msp:
        if e.dxftype() not in {"TEXT","MTEXT"}: continue
        raw=e.dxf.text if e.dxftype()=="TEXT" else e.plain_text(); val=clean(raw)
        try: x,y=float(e.dxf.insert.x),float(e.dxf.insert.y)
        except Exception: continue
        matches=[m.group(0).replace(" ","-").replace("_","-").upper() for m in TAG_RE.finditer(val)]
        text.append({"text_id":f"TXT-{len(text)+1:05d}","text":val,"x":x,"y":y,"cad_layer":e.dxf.layer,"candidate_tags":"|".join(matches)})
        for tag in matches:
            if 80<=x<=4700 and 35<=y<=303 and tag!="P-47": raw_tags.append({"tag":tag,"x":x,"y":y,"cad_layer":e.dxf.layer})
    text=pd.DataFrame(text)
    tags=pd.DataFrame(raw_tags).sort_values("y",ascending=False).drop_duplicates("tag").reset_index(drop=True)
    tags["asset_class"]=tags.tag.map(cls)

    geometry=[]; segments=[]
    for e in msp:
        kind=e.dxftype()
        if kind in {"LINE","LWPOLYLINE","CIRCLE","ARC","ELLIPSE","SOLID"}:
            try:
                ex=bbox.extents([e],fast=True); c=ex.center
                if math.isfinite(c.x) and math.isfinite(c.y):
                    geometry.append({"geometry_id":f"GEO-{len(geometry)+1:06d}","geometry_type":kind,"x":float(c.x),"y":float(c.y),"cad_layer":e.dxf.layer})
            except Exception: pass
        pts=[]
        try:
            if kind=="LINE": pts=[(e.dxf.start.x,e.dxf.start.y),(e.dxf.end.x,e.dxf.end.y)]
            elif kind=="LWPOLYLINE": pts=[(p[0],p[1]) for p in e.get_points("xy")]
        except Exception: pts=[]
        for a,b in zip(pts,pts[1:]):
            mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
            if 80<=mx<=4700 and 0<=my<=303 and math.dist(a,b)>0:
                segments.append({"segment_id":f"SEG-{len(segments)+1:06d}","x1":a[0],"y1":a[1],"x2":b[0],"y2":b[1],"length":math.dist(a,b),"cad_layer":e.dxf.layer})
    geo=pd.DataFrame(geometry); seg=pd.DataFrame(segments)

    assets=[]
    for i,t in tags.iterrows():
        d=np.hypot(geo.x-t.x,geo.y-t.y); near=geo[d<=20]
        n=len(near); gconf=float(1-np.exp(-n/18)); semantic=.92 if t.asset_class!="Tagged Component" else .65
        conf=.40*.99+.25*semantic+.25*gconf+.10*.95; coverage=.50+.25*min(1,n/20); ready=conf*coverage
        assets.append({"asset_id":f"AST-{i+1:04d}","tag":t.tag,"asset_class":t.asset_class,"x":t.x,"y":t.y,
          "cad_layer":t.cad_layer,"symbol_source":"FLATTENED_DXF_VECTOR_GEOMETRY","local_geometry_count":n,
          "local_line_count":int(near.geometry_type.isin(["LINE","LWPOLYLINE"]).sum()),
          "local_circle_arc_count":int(near.geometry_type.isin(["CIRCLE","ARC","ELLIPSE"]).sum()),
          "text_confidence":.99,"semantic_confidence":round(semantic,3),"geometry_confidence":round(gconf,3),
          "overall_confidence":round(conf,3),"evidence_coverage":round(coverage,3),"digital_twin_readiness":round(ready,3),
          "confidence_level":"HIGH" if conf>=.8 else "MEDIUM" if conf>=.6 else "LOW",
          "readiness_level":"HIGH" if ready>=.8 else "MEDIUM" if ready>=.55 else "LOW",
          "review_status":"REVIEW_SYMBOL_AND_CONNECTIVITY","confidence_basis":"Native DXF tag plus nearby flattened vector geometry"})
    assets=pd.DataFrame(assets)

    # Conservative spatial-line candidates: both assets must be close to the same CAD layer
    touches={}
    for _,a in assets.iterrows():
        cand=seg[(np.minimum(seg.x1,seg.x2)<=a.x+12)&(np.maximum(seg.x1,seg.x2)>=a.x-12)&
                 (np.minimum(seg.y1,seg.y2)<=a.y+12)&(np.maximum(seg.y1,seg.y2)>=a.y-12)].copy()
        if cand.empty: continue
        cand["d"]=point_segment_dist(a.x,a.y,cand)
        for layer,dmin in cand[cand.d<=12].groupby("cad_layer").d.min().items():
            touches.setdefault(str(layer),[]).append((a.asset_id,float(dmin)))
    rel=[]; seen=set()
    for layer,items in touches.items():
        # Within each layer, connect only the nearest spatial neighbour, not every pair.
        ids=[x[0] for x in items]
        for aid,_ in items:
            a=assets[assets.asset_id==aid].iloc[0]
            choices=[]
            for bid,_ in items:
                if bid==aid: continue
                b=assets[assets.asset_id==bid].iloc[0]; choices.append((math.hypot(a.x-b.x,a.y-b.y),bid))
            if not choices: continue
            dist,bid=min(choices); key=tuple(sorted((aid,bid)))
            if key in seen or dist>500: continue
            seen.add(key); conf=max(.20,min(.60,.60-dist/1250))
            rel.append({"source_id":key[0],"target_id":key[1],"relationship":"CONNECTED_TO_CANDIDATE",
                        "confidence":round(conf,3),"evidence":f"Shared CAD layer {layer} + spatial proximity",
                        "direction":"UNKNOWN","review_status":"ENGINEERING_REVIEW_REQUIRED"})
    rel=pd.DataFrame(rel,columns=["source_id","target_id","relationship","confidence","evidence","direction","review_status"])

    g=nx.MultiDiGraph(); g.add_node("DRAWING-001",node_type="Drawing",name=Path(dxf_path).name)
    for c in sorted(assets.asset_class.unique()): g.add_node("CLASS-"+re.sub(r"\W+","-",c).upper(),node_type="AssetClass",name=c)
    for _,a in assets.iterrows():
        g.add_node(a.asset_id,node_type="Asset",tag=a.tag,asset_class=a.asset_class,confidence=float(a.overall_confidence),readiness=float(a.digital_twin_readiness),x=float(a.x),y=float(a.y))
        g.add_edge(a.asset_id,"DRAWING-001",relationship="SHOWN_ON",confidence=1.0)
        g.add_edge(a.asset_id,"CLASS-"+re.sub(r"\W+","-",a.asset_class).upper(),relationship="TYPE_OF",confidence=float(a.semantic_confidence))
    for _,r in rel.iterrows(): g.add_edge(r.source_id,r.target_id,relationship=r.relationship,confidence=float(r.confidence),review_status=r.review_status)

    review=assets[["asset_id","tag","asset_class","overall_confidence","evidence_coverage","digital_twin_readiness","review_status"]].copy()
    review["symbol_decision"]="PENDING"; review["connectivity_decision"]="PENDING"; review["comment"]=""
    text.to_csv(out/"text_inventory.csv",index=False); tags.to_csv(out/"process_tags.csv",index=False)
    geo.to_csv(out/"geometry_inventory.csv",index=False); seg.to_csv(out/"line_segments.csv",index=False)
    assets.to_csv(out/"asset_master.csv",index=False); rel.to_csv(out/"relationships.csv",index=False); review.to_csv(out/"validation_queue.csv",index=False)
    nx.write_graphml(g,out/"knowledge_graph.graphml")
    (out/"knowledge_graph.json").write_text(json.dumps(nx.node_link_data(g,edges="links"),indent=2,default=str),encoding="utf-8")
    summary={"source_file":Path(dxf_path).name,"mode":"FLATTENED_VECTOR_DXF","entity_count":len(msp),"entity_types":dict(Counter(e.dxftype() for e in msp)),
      "text_records":len(text),"process_assets":len(assets),"asset_classes":int(assets.asset_class.nunique()),"geometry_objects":len(geo),"process_line_segments":len(seg),
      "preliminary_connections":len(rel),"graph_nodes":g.number_of_nodes(),"graph_edges":g.number_of_edges(),"pending_reviews":len(review),
      "warning":"Symbol class is inferred from tag prefix and local vector evidence; topology requires engineering review."}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary

if __name__=="__main__":
    import sys
    print(json.dumps(run(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else "flattened_results"),indent=2))
