# ai_project_CV_2025 — Multi-Piece YOLO (Dataset + Training)

## 0. Prerequisites

- Python 3.9+ installed
- Git repo already cloned
- You are in the **repository root**
- A virtual environment is recommended
- **Ultralytics** must be installed so the `yolo` CLI is available in the **same venv** as the Python used to run the script

### Install Python packages (once)

From repo root:

```bash
# (optional) create and activate virtual env
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# install dependencies (preferred)
pip install -r requirements.txt

# if you don’t have requirements.txt:
# pip install ultralytics opencv-python pillow numpy
