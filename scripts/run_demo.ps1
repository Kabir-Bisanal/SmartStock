$ErrorActionPreference = "Stop"

Write-Host "Checking the SmartStock environment..."
python -m smartstock.utils.environment_check

Write-Host "Checking SmartStock demo artifacts..."
python -m smartstock.utils.health_check --skip-hashes

$env:SMARTSTOCK_DATA_MODE = "csv"
Write-Host "Starting the read-only CSV demo at http://localhost:8501"
python -m streamlit run app/streamlit_app.py
