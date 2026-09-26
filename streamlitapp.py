"""Entrypoint alias for streamlit_app.py."""
import runpy
from pathlib import Path

app_path = Path(__file__).parent / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
