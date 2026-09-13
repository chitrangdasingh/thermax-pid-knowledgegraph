# Thermax P&ID Knowledge Graph PoC

Deployment-ready Streamlit application generated from the old
`Thermax Single Tube Reactor System_PID(2).dxf` extraction.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository, for example `thermax-pid-knowledgegraph`.
2. Upload the complete contents of this folder to the repository root.
3. Sign in at https://share.streamlit.io with the GitHub account that owns it.
4. Choose **Create app**, select the repository/branch, and set main file to `app.py`.
5. Deploy and choose an available subdomain, such as
   `thermax-pid-knowledgegraph.streamlit.app`.

The final URL is assigned by Streamlit and cannot be reserved without account
access. Do not commit confidential raw P&IDs to a public repository; this PoC
contains derived demo data and a rendered drawing, which must still be approved
for external hosting by Thermax.

## Current evidence status

- DXF extraction, asset candidates, hierarchy and query interface: implemented.
- Connections: preliminary and marked for review.
- CNN detections: not claimed until reviewed labels and trained weights exist.
- Historian, maintenance and KPI data: schema-ready, no synthetic claims.

