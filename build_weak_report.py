"""Build an honest pre-training review table from current DXF evidence."""
from pathlib import Path
import sys
import pandas as pd


def run(data_dir,preannotation_dir,output_dir):
    data=Path(data_dir);pre=Path(preannotation_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    assets=pd.read_csv(data/"asset_master.csv")
    relations=pd.read_csv(data/"relationships.csv")
    boxes=pd.read_csv(pre/"preannotation_manifest.csv")
    linked=set(relations.source_id)|set(relations.target_id)
    detections=assets.merge(boxes[["asset_id","x1","y1","x2","y2","annotation_status"]],on="asset_id",how="left")
    detections["result_type"]="TAGGED_ASSET_PREANNOTATION"
    detections["detected_name"]=detections.asset_class
    detections["symbol_confidence"]=pd.NA
    detections["ocr_confidence"]=detections.text_confidence
    detections["association_confidence"]=pd.NA
    detections["flaw_flag"]="REVIEW"
    detections["model_status"]="CNN_NOT_TRAINED"
    detections["review_status"]="ENGINEERING_BOX_AND_CLASS_REVIEW_REQUIRED"
    columns=["asset_id","tag","result_type","detected_name","symbol_confidence","ocr_confidence",
             "association_confidence","x1","y1","x2","y2","annotation_status","flaw_flag","model_status","review_status"]
    detections[columns].to_csv(out/"weak_detection_results.csv",index=False)
    flaws=[]
    def add(kind,severity,asset_id,tag,evidence):
        flaws.append({"flaw_id":f"PRE-FLW-{len(flaws)+1:04d}","flaw_type":kind,"severity":severity,"asset_id":asset_id,
                      "tag":tag,"evidence":evidence,"detection_method":"DXF_RULE_PRETRAINING",
                      "review_status":"ENGINEERING_REVIEW_REQUIRED"})
    for a in assets.itertuples(index=False):
        add("SYMBOL_BOUNDARY_UNVERIFIED","MEDIUM",a.asset_id,a.tag,"Weak box inferred from tag and local vector centres")
        if a.asset_id not in linked:add("ORPHAN_ASSET_CANDIDATE","HIGH",a.asset_id,a.tag,"No candidate relationship generated")
    lookup=assets.set_index("asset_id").tag.to_dict()
    for r in relations.itertuples(index=False):
        pair=f"{lookup.get(r.source_id,r.source_id)} ↔ {lookup.get(r.target_id,r.target_id)}"
        add("UNKNOWN_FLOW_DIRECTION","HIGH",r.source_id,pair,"Candidate connection direction is UNKNOWN")
        if r.confidence<.5:add("LOW_TOPOLOGY_CONFIDENCE","MEDIUM",r.source_id,pair,f"Rule confidence {r.confidence:.3f}")
    pd.DataFrame(flaws).to_csv(out/"weak_flaw_report.csv",index=False)
    return {"weak_detections":len(detections),"review_flags":len(flaws),"orphan_candidates":sum(f["flaw_type"]=="ORPHAN_ASSET_CANDIDATE" for f in flaws),
            "unknown_directions":sum(f["flaw_type"]=="UNKNOWN_FLOW_DIRECTION" for f in flaws),
            "low_topology_confidence":sum(f["flaw_type"]=="LOW_TOPOLOGY_CONFIDENCE" for f in flaws)}


if __name__=="__main__":print(run(sys.argv[1] if len(sys.argv)>1 else "data",sys.argv[2] if len(sys.argv)>2 else "cnn_preannotations",
                                   sys.argv[3] if len(sys.argv)>3 else "cnn_preannotations/reports"))
