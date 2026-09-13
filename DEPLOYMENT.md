# Streamlit Community Cloud deployment

1. Review `data/pid_render.png` and CSV files for Thermax external-sharing approval.
2. Create a GitHub repository and upload this folder's contents to its root.
3. Open https://share.streamlit.io and sign in with the repository-owning GitHub account.
4. Select **Create app** and choose the repository and branch.
5. Set the main file path to `app.py` and deploy.
6. Choose an available URL slug. Streamlit assigns the final `.streamlit.app` URL.

No secrets are required for this read-only demonstration. Do not commit raw
DWG/DXF files, trained proprietary weights, historian credentials, or
`.streamlit/secrets.toml`.
