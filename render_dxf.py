"""Create a vector background and locate CAD INSERT blocks for a DXF result directory."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import ezdxf
import pandas as pd
from PIL import Image, ImageDraw


def run(dxf_file, result_dir):
    folder=Path(result_dir);folder.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(folder/"line_segments.csv")
    xmin=min(s.x1.min(),s.x2.min())-10;xmax=max(s.x1.max(),s.x2.max())+10
    ymin=min(s.y1.min(),s.y2.min())-10;ymax=max(s.y1.max(),s.y2.max())+10
    w=8500;h=max(200,round(w*(ymax-ymin)/(xmax-xmin)))
    image=Image.new('RGB',(w,h),'white');draw=ImageDraw.Draw(image)
    for v in s.itertuples(index=False):
        draw.line(((round((v.x1-xmin)*w/(xmax-xmin)),round((ymax-v.y1)*h/(ymax-ymin))),
                   (round((v.x2-xmin)*w/(xmax-xmin)),round((ymax-v.y2)*h/(ymax-ymin)))),fill=(45,63,74),width=1)
    image.save(folder/'vector_drawing.png',optimize=True)
    bounds=dict(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
    (folder/'drawing_bounds.json').write_text(json.dumps(bounds),encoding='utf-8')
    blocks=[]
    for e in ezdxf.readfile(str(dxf_file)).modelspace().query('INSERT'):
        x,y=float(e.dxf.insert.x),float(e.dxf.insert.y)
        blocks.append({'block_id':f'BLK-{len(blocks)+1:04d}','block_name':e.dxf.name,
            'x':x,'y':y,'cad_layer':e.dxf.layer,
            'attributes_json':json.dumps({a.dxf.tag:a.dxf.text for a in e.attribs}),
            'drawing_region':'INSIDE_DRAWING' if xmin<=x<=xmax and ymin<=y<=ymax else 'OUTSIDE_DRAWING_OR_LEGEND'})
    with (folder/'cad_blocks.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=['block_id','block_name','x','y','cad_layer','attributes_json','drawing_region'])
        writer.writeheader();writer.writerows(blocks)
    digest=hashlib.sha256()
    with Path(dxf_file).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
    (folder/'source_manifest.json').write_text(json.dumps({'filename':Path(dxf_file).name,'sha256':digest.hexdigest(),
        'note':'Checksum of the exact uploaded source DXF.'},indent=2),encoding='utf-8')
    return {'blocks':len(blocks),'outside':sum(b['drawing_region']!='INSIDE_DRAWING' for b in blocks),'image_size':[w,h]}


if __name__=='__main__':print(json.dumps(run(sys.argv[1],sys.argv[2]),indent=2))
