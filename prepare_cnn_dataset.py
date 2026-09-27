"""Create reviewable weak P&ID symbol boxes from the DXF evidence.

These are PRE-ANNOTATIONS, not ground truth. Import the generated image and
labels into CVAT, correct every box/class, then export an approved YOLO dataset.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _to_pixels(x, y, bounds, width, height):
    px=(x-bounds["xmin"])/(bounds["xmax"]-bounds["xmin"])*width
    py=(bounds["ymax"]-y)/(bounds["ymax"]-bounds["ymin"])*height
    return px,py


def run(result_dir, output_dir):
    data=Path(result_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    image=Image.open(data/"vector_drawing.png").convert("RGB")
    width,height=image.size
    bounds=json.loads((data/"drawing_bounds.json").read_text())
    assets=pd.read_csv(data/"asset_master.csv")
    geometry=pd.read_csv(data/"geometry_inventory.csv")
    classes=sorted(assets.asset_class.unique().tolist())
    class_id={name:i for i,name in enumerate(classes)}
    rows=[];labels=[]
    preview=image.copy();draw=ImageDraw.Draw(preview)
    colors=["#e53935","#00897b","#5e35b1","#fb8c00","#039be5","#7cb342","#6d4c41","#d81b60","#3949ab","#00acc1"]
    for asset in assets.itertuples(index=False):
        nearby=geometry[((geometry.x-asset.x)**2+(geometry.y-asset.y)**2)<=25**2]
        # The tag insertion point anchors the weak box; nearby primitive centres
        # expand it. Engineering must move/resize this around the actual symbol.
        xs=[asset.x]+nearby.x.tolist();ys=[asset.y]+nearby.y.tolist()
        x1=min(xs)-4;x2=max(xs)+4;y1=min(ys)-4;y2=max(ys)+4
        if x2-x1<14:x1-=7;x2+=7
        if y2-y1<10:y1-=5;y2+=5
        px1,py2=_to_pixels(x1,y1,bounds,width,height)
        px2,py1=_to_pixels(x2,y2,bounds,width,height)
        px1=max(0,min(width-1,px1));px2=max(0,min(width-1,px2))
        py1=max(0,min(height-1,py1));py2=max(0,min(height-1,py2))
        if px2<=px1 or py2<=py1:continue
        cx=(px1+px2)/2/width;cy=(py1+py2)/2/height
        bw=(px2-px1)/width;bh=(py2-py1)/height
        labels.append(f"{class_id[asset.asset_class]} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        rows.append({"annotation_id":f"PRE-{len(rows)+1:04d}","asset_id":asset.asset_id,"tag":asset.tag,
            "proposed_class":asset.asset_class,"class_id":class_id[asset.asset_class],"x1":round(px1),"y1":round(py1),
            "x2":round(px2),"y2":round(py2),"source":"DXF_TAG_PLUS_NEARBY_VECTOR_CENTRES",
            "annotation_status":"WEAK_PREANNOTATION","engineer_decision":"PENDING","engineer_comment":""})
        color=colors[class_id[asset.asset_class]%len(colors)]
        draw.rectangle((px1,py1,px2,py2),outline=color,width=3)
        draw.text((px1,max(0,py1-14)),f"{asset.tag} | {asset.asset_class}",fill=color)
    (out/"images").mkdir(exist_ok=True);(out/"labels").mkdir(exist_ok=True)
    image.save(out/"images"/"thermax_pid_source.png")
    preview.save(out/"preannotation_preview.png")
    (out/"labels"/"thermax_pid_source.txt").write_text("\n".join(labels)+"\n",encoding="utf-8")
    pd.DataFrame(rows).to_csv(out/"preannotation_manifest.csv",index=False)
    (out/"classes.txt").write_text("\n".join(classes)+"\n",encoding="utf-8")
    (out/"README_REVIEW_REQUIRED.md").write_text(
        "# Weak pre-annotations — engineering review required\n\n"
        "These boxes were inferred from DXF tag positions and nearby primitive centres. They are not CNN detections or ground truth. "
        "Import the image/label into CVAT, move every box to the actual symbol, correct the class, add missed symbols, and mark symbol-only objects. "
        "Do not train until multiple drawings are reviewed and split by drawing into train/validation/test.\n",encoding="utf-8")
    summary={"image":str(out/"images"/"thermax_pid_source.png"),"weak_boxes":len(rows),"classes":classes,
             "training_ready":False,"reason":"All boxes require engineering review; one drawing cannot provide an independent test set."}
    (out/"preannotation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


if __name__=="__main__":
    print(json.dumps(run(sys.argv[1] if len(sys.argv)>1 else "data",
                         sys.argv[2] if len(sys.argv)>2 else "cnn_preannotations"),indent=2))
