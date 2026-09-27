# Thermax Single Tube Reactor P&ID · digital-twin platform POC

Input: user-supplied `Thermax Single Tube Reactor System_PID (004).dxf` (flattened vector DXF). This package includes extracted results, marked weak-preannotation preview, confidence/flaw reports, asset hierarchy, searchable graph, Streamlit dashboard, REST/Q&A API, YOLOv5 + Tesseract workflow, Microsoft Azure reference integration, and optional DEXPI/pyDEXPI/Neo4j/Graph-RAG modules. The original 22 MB DXF is **not** copied into this ZIP; upload it separately to rerun ingestion.

## Results obtained from the actual DXF

| Item | Count / state |
| --- | --- |
| DXF entities / text records | 13,985 / 848 |
| DXF tags promoted to process asset candidates | 24 across 10 tag-derived classes |
| Exploded CAD segments | 53,958, currently unclassified drawing primitives |
| CAD INSERT blocks | 49, all outside the process drawing bounds, likely legend/library symbols |
| Candidate asset-to-asset connections | 16, direction UNKNOWN, engineering review required |
| Exported graph | 35 nodes, 64 relationships (24 SHOWN_ON, 24 TYPE_OF, 16 candidate links) |
| Weak preannotation / review flags | 24 boxes / 48 flags; not CNN-validated |
| Condensed searchable graph | 24 asset nodes, 16 domain candidates, 14 lexical class links |
| Historian readings / trained predictive model / approved YOLOv5 weights | 0 / none / none |
| DEXPI status | Proposed mapping only; the supplied DXF is not a DEXPI Proteus XML |

## CNN text/symbol workflow matching the reference screenshot

`prepare_cnn_dataset.py data cnn_preannotations` creates weak review boxes from DXF tags and local geometry plus `preannotation_preview.png`. Correct them in CVAT and add all missing/symbol-only objects across representative drawings. `validate_yolov5_dataset.py` blocks incomplete review, missing test/negative data and split leakage. `train_yolov5.py` calls the official YOLOv5 repository. `pid_vision.py` then performs tiled YOLOv5 detection, horizontal/vertical Tesseract OCR, one-to-one tag association, calibrated evidence fields, visible flaw checks, marked output and asset hierarchy generation.

The dashboard's **Vision AI inspection** page accepts an approved `best.pt` and a P&ID image. It does not use a generic object detector as if it knew P&ID symbols.

## Launch locally (Python 3.11 or 3.12)

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app initializes `runtime/pid.sqlite3` from bundled `data/` and logs subsequent imports/decisions in the local SQLite file. `runtime/` is ignored in Git and is ephemeral on most free hosted Streamlit services; mount durable IT-approved storage for sustained multiuser use. For a public Streamlit demo, omit plant historian files and credentials.

Optional API (separate terminal):

```bash
# Set PID_API_KEY in your server's secure environment first.
uvicorn api:app --host 127.0.0.1 --port 8000
```

`http://127.0.0.1:8000/docs` shows the REST contract. `/health` is open; all data routes require `X-API-Key`.

Private container preview:

```bash
docker compose up --build
```

Open `http://localhost:8501`. The named `pid_runtime` volume preserves the local SQLite store. For optional vision or Graph-RAG images, set the corresponding Docker build arguments and review the larger dependency/licence footprint.

For engineering approvals, set a separate server secret `PID_REVIEW_TOKEN` and enter that token in the Streamlit sidebar. For an approved external LLM, configure `PID_LLM_ENDPOINT` (an approved HTTPS chat-completions endpoint), `PID_LLM_API_KEY` and `PID_LLM_MODEL`. Data would leave this app when the user clicks the model button. It is disabled by default.

## Deploy on Streamlit Community Cloud

Push the entire extracted folder contents (`app.py`, support Python files, `requirements.txt`, `data/`, `docs/` and `notebooks/`) to the **root** of a GitHub repo. In [share.streamlit.io](https://share.streamlit.io), choose repository, branch `main`, entrypoint `app.py`. Configure secrets/server env carefully. A private repo and access policy require confirmation with Thermax IT because drawings are proprietary. Cloud URL is created by Streamlit after deployment; this package by itself does not create a live URL. The optional FastAPI process needs a separate server/container deployment and persistent database.

## Notebook

Upload `notebooks/Thermax_PID_YOLOv5_Tesseract_Colab.ipynb` to Colab and use a Python 3/T4 GPU runtime. It runs extraction, weak-label export, the mandatory annotation stop, YOLOv5 validation/training/test, Tesseract association, hierarchy/graph generation and result packaging. Colab is a temporary workbench, not a production application server.

## Intelligent formats and integrations

- `data/asset_hierarchy.csv` and the companion Excel workbook are searchable asset registers with source confidence and review state.
- `data/graphrag/high_level_graph.json` / `.graphml` preserve candidate process and lexical relations.
- `neo4j_nodes.csv`, `neo4j_edges.csv` and `neo4j_import.cypher` load the high-level graph into Neo4j without claiming the DXF is DEXPI.
- A genuine DEXPI 1.3 Proteus XML can be loaded through `load_dexpi_proteus`; pyDEXPI produces plant, process and conceptual graphs.
- `POST /qa` exposes deterministic, evidence-backed questions; optional Gemini Graph-RAG is disabled until server-side credentials and IT approval exist.
- `azure_pid_client.py` calls a separately deployed Microsoft reference service and preserves manual correction between symbol, text, graph and persistence stages.

## Documentation

- `docs/STEP_BY_STEP.md`: complete operator sequence, data lineage, model gates and connection rules.
- `docs/ENGINEERING_REQUEST.md`: exact P&ID, BOM, historian, safety, security and integration information to request.
- `docs/DATA_CONTRACTS.md`: normalized schemas, namespace pattern and API examples.
- `docs/AZURE_REFERENCE_IMPLEMENTATION.md`: how to use the two requested Microsoft repositories.
- `docs/INTELLIGENT_FORMATS.md`: search, links, DEXPI/Neo4j, plant-system joins and governance.

## Model honesty

No CNN or LLM has learned from this P&ID. The DXF text anchor and tag-prefix rules generate candidate classes and evidence scores, while spatial proximity generates candidate links. Train YOLOv5 only after engineers label representative drawings and approve held-out per-class metrics. A static P&ID cannot supply live temperature, pressure, flow, conversion or equipment condition. The forecaster activates only on real approved observations; HAZOP answers are review prompts, never conclusions.

Review upstream licences before enterprise distribution. The Microsoft samples are referenced rather than copied. YOLOv5 and pyDEXPI have their own open-source obligations; Thermax legal/IT should approve the chosen deployment model and pinned versions.
