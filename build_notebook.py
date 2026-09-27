"""Build the complete Google Colab runbook shipped with this platform."""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# Thermax P&ID: DXF → YOLOv5/Tesseract → graph → dashboard

Use **Google Colab / Python 3**. Before running, choose **Runtime → Change runtime type → T4 GPU**. Use one notebook/runtime from top to bottom; do not open repeated Streamlit or training sessions.

This runbook has two honest stages:

1. DXF-assisted extraction and weak preannotation (works immediately).
2. Engineer-reviewed YOLOv5 training and inference (requires corrected labels from multiple complete drawings).

The included 24 Thermax boxes are not a trained-CNN result. Do not skip the review gate."""),
    md("## 1 · Upload the platform ZIP and source DXF"),
    code("""from google.colab import files
from pathlib import Path
import os, shutil, sys, zipfile, json

uploaded = files.upload()
zip_candidates = [Path('/content') / name for name in uploaded if name.lower().endswith('.zip')]
dxf_candidates = [Path('/content') / name for name in uploaded if name.lower().endswith('.dxf')]
if not zip_candidates or not dxf_candidates:
    raise ValueError('Upload the delivered platform ZIP and one .dxf file in this cell.')

package_zip, dxf_path = zip_candidates[0], dxf_candidates[0]
extract_root = Path('/content/Thermax_PID_Package')
if extract_root.exists(): shutil.rmtree(extract_root)
extract_root.mkdir(parents=True)
with zipfile.ZipFile(package_zip) as archive: archive.extractall(extract_root)
project_candidates = [p.parent for p in extract_root.rglob('app.py') if (p.parent/'requirements.txt').is_file()]
if not project_candidates: raise FileNotFoundError('app.py and requirements.txt were not found in the ZIP')
project = project_candidates[0]
sys.path.insert(0, str(project))
print('Project:', project)
print('DXF:', dxf_path)
"""),
    md("## 2 · Install into this Colab kernel"),
    code("""!apt-get -qq update
!apt-get -qq install -y tesseract-ocr
%pip install -q -r {project}/requirements-vision.txt
import torch, pytesseract
print('Python:', sys.version.split()[0])
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE — enable T4 GPU before training')
print('Tesseract:', pytesseract.get_tesseract_version())
"""),
    md("## 3 · Extract native DXF evidence"),
    code("""from ingest_dxf import run as ingest_dxf
from render_dxf import run as render_dxf
from asset_hierarchy import build_from_dxf
import pandas as pd

data_dir = project/'data'
summary = ingest_dxf(dxf_path, data_dir)
render_info = render_dxf(dxf_path, data_dir)
assets = pd.read_csv(data_dir/'asset_master.csv')
relationships = pd.read_csv(data_dir/'relationships.csv')
hierarchy, hierarchy_nodes = build_from_dxf(assets, relationships)
hierarchy.to_csv(data_dir/'asset_hierarchy.csv', index=False)
hierarchy_nodes.to_csv(data_dir/'hierarchy_nodes.csv', index=False)
print(json.dumps(summary, indent=2))
"""),
    md("## 4 · View the actual P&ID render and candidate tables"),
    code("""from IPython.display import display, Image
display(Image(filename=str(data_dir/'vector_drawing.png'), width=1400))
display(hierarchy[['asset_id','tag','asset_class','overall_confidence','cnn_symbol_confidence','candidate_connection_count','review_status']])
display(relationships)
print('Important: cnn_symbol_confidence is empty until approved YOLOv5 weights are run.')
"""),
    md("## 5 · Create weak preannotations for CVAT review"),
    code("""import subprocess
preannotations = project/'cnn_preannotations_colab'
subprocess.run([sys.executable, str(project/'prepare_cnn_dataset.py'), str(data_dir), str(preannotations)], check=True)
display(Image(filename=str(preannotations/'preannotation_preview.png'), width=1400))
print((preannotations/'README_REVIEW_REQUIRED.md').read_text())
"""),
    code("""preannotation_zip = Path('/content/thermax_cvat_preannotations.zip')
shutil.make_archive(str(preannotation_zip.with_suffix('')), 'zip', preannotations)
files.download(str(preannotation_zip))
"""),
    md("""## STOP · Correct the labels before training

Import the preannotation image/label into CVAT or Roboflow. Correct every class and box, add missed symbols and hard negative/background crops, and repeat for multiple representative P&ID drawings. Export YOLO format with complete-drawing train/validation/test separation.

Put `dataset_review.json` beside `data.yaml`. It must be truthfully completed and approved. The validator rejects DRAFT status, fewer than three drawings, missing test split, missing negatives, unconfirmed licences and duplicate image bytes across splits."""),
    md("## 6 · Upload the reviewed YOLOv5 dataset ZIP"),
    code("""reviewed = files.upload()
dataset_zip = next((Path('/content')/name for name in reviewed if name.lower().endswith('.zip')), None)
if dataset_zip is None: raise ValueError('Upload the reviewed YOLO dataset ZIP')
dataset_root = Path('/content/reviewed_pid_dataset')
if dataset_root.exists(): shutil.rmtree(dataset_root)
dataset_root.mkdir()
with zipfile.ZipFile(dataset_zip) as archive: archive.extractall(dataset_root)
yaml_candidates = list(dataset_root.rglob('data.yaml')) + list(dataset_root.rglob('data.yml'))
if not yaml_candidates: raise FileNotFoundError('data.yaml not found in dataset ZIP')
data_yaml = yaml_candidates[0]
print('Dataset config:', data_yaml)
"""),
    md("## 7 · Enforce the dataset quality gate"),
    code("""from validate_yolov5_dataset import validate_dataset
validation = validate_dataset(data_yaml)
print(json.dumps(validation, indent=2))
"""),
    md("## 8 · Clone official YOLOv5 and install its pinned dependencies"),
    code("""yolov5_repo = Path('/content/yolov5')
if not yolov5_repo.exists():
    !git clone --depth 1 https://github.com/ultralytics/yolov5.git /content/yolov5
%pip install -q -r /content/yolov5/requirements.txt
"""),
    md("## 9 · Train YOLOv5 on reviewed P&ID symbols"),
    code("""run_root = Path('/content/yolov5_runs')
!python {project}/train_yolov5.py {data_yaml} --yolov5-repo /content/yolov5 --weights yolov5s.pt --epochs 100 --img 1280 --batch-size 8 --device 0 --project {run_root} --name thermax_pid_symbols
best_weights = run_root/'thermax_pid_symbols/weights/best.pt'
if not best_weights.is_file(): raise FileNotFoundError(best_weights)
print('Best weights:', best_weights)
"""),
    md("## 10 · Evaluate on the untouched test drawings"),
    code("""test_run = Path('/content/yolov5_test')
!python /content/yolov5/val.py --data {data_yaml} --weights {best_weights} --img 1280 --task test --project {test_run} --name held_out --exist-ok
for name in ['confusion_matrix.png','PR_curve.png','P_curve.png','R_curve.png','F1_curve.png']:
    path = test_run/'held_out'/name
    if path.exists(): display(Image(filename=str(path), width=900))
print('Do not deploy unless per-class errors and false positives pass the agreed engineering acceptance criteria.')
"""),
    md("## 11 · Run trained YOLOv5 + Tesseract on a new P&ID"),
    code("""inference_upload = files.upload()
inference_image = next((Path('/content')/name for name in inference_upload if Path(name).suffix.lower() in {'.png','.jpg','.jpeg'}), None)
if inference_image is None:
    print('No image uploaded; using the DXF render.')
    inference_image = data_dir/'vector_drawing.png'
from pid_vision import run as run_vision
vision_dir = Path('/content/pid_vision_results')
vision_summary = run_vision(inference_image, best_weights, vision_dir, conf=0.15, yolov5_repo=yolov5_repo)
print(json.dumps(vision_summary, indent=2))
display(Image(filename=str(vision_dir/'annotated_pid.png'), width=1400))
display(pd.read_csv(vision_dir/'detection_results.csv'))
display(pd.read_csv(vision_dir/'flaw_report.csv'))
display(pd.read_csv(vision_dir/'asset_hierarchy.csv'))
"""),
    md("## 12 · Build the searchable high-level graph and Neo4j files"),
    code("""from dexpi_graphrag import build_asset_graph, condensed_graph, export_graph, local_answer
detections = pd.read_csv(vision_dir/'detection_results.csv')
print('Vision results are separate from the original DXF candidate graph until topology is reviewed.')
pid_graph = condensed_graph(__import__('dexpi_graphrag').load_current_graph(data_dir), include_lexical=True)
graph_dir = Path('/content/pid_graph_results')
print(export_graph(pid_graph, graph_dir))
for question in ['How many reactors are present?','What is connected to R-300?','What data are needed for methanol conversion?']:
    print()
    print('Q:', question)
    print('A:', local_answer(question, pid_graph)['answer'])
"""),
    md("## 13 · Optional Microsoft Azure reference repositories"),
    code("""# This downloads the upstream samples; it does not deploy Azure resources.
!python {project}/setup_reference_repos.py --destination /content/external
print('Read:', project/'docs/AZURE_REFERENCE_IMPLEMENTATION.md')
"""),
    md("""## 14 · Package all Colab-generated results

The Streamlit app is deployed separately: push the delivered project contents to a private GitHub repository and select `app.py` in Streamlit Community Cloud, or deploy inside the IT-approved private environment. Colab is a temporary training workbench, not a persistent application server."""),
    code("""result_root = Path('/content/Thermax_PID_Run_Results')
if result_root.exists(): shutil.rmtree(result_root)
result_root.mkdir()
for source, name in [(data_dir,'dxf_data'),(preannotations,'preannotations'),(vision_dir,'vision_results'),(graph_dir,'graph_results'),(run_root/'thermax_pid_symbols','training_run'),(test_run/'held_out','test_run')]:
    if Path(source).exists(): shutil.copytree(source, result_root/name)
shutil.copy2(best_weights, result_root/'best.pt')
result_zip = Path('/content/Thermax_PID_Run_Results.zip')
shutil.make_archive(str(result_zip.with_suffix('')), 'zip', result_root)
print(result_zip, result_zip.stat().st_size, 'bytes')
files.download(str(result_zip))
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "Thermax_PID_YOLOv5_Tesseract_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


target = Path(__file__).resolve().parent / "notebooks" / "Thermax_PID_YOLOv5_Tesseract_Colab.ipynb"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(target)
