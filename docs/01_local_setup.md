# 01 Local Setup

Use Terminal for environment setup and VS Code for reading/editing project files.
Start with the DeepLabCut core package so the pipeline can run locally. Install the GUI later if the large PySide6 download is reliable.

## 1. Open The Project Folder

In Terminal:

```bash
cd "/Users/lizhiheng/Desktop/research/生物"
```

In VS Code:

```bash
code "/Users/lizhiheng/Desktop/research/生物"
```

If `code` is not available, open VS Code manually and choose:

```text
File -> Open Folder -> /Users/lizhiheng/Desktop/research/生物
```

## 2. Install Conda Or Miniforge

DeepLabCut should not be installed into the system Python 3.14 environment.
Create a separate Python 3.12 environment.

Recommended: install Miniforge for macOS from:

```text
https://conda-forge.org/download/
```

After installation, close Terminal and open a new Terminal window.

Check:

```bash
conda --version
```

## 3. Create The DeepLabCut Environment

From this project folder:

```bash
conda env create -f environment.yml
conda activate DEEPLABCUT
```

If the environment already exists:

```bash
conda activate DEEPLABCUT
conda env update -f environment.yml --prune
```

## 4. Install DeepLabCut Core

Install the stable core package first:

```bash
python -m pip install --no-cache-dir --progress-bar off deeplabcut==3.0.1
```

The GUI package is larger because it downloads napari and PySide6. Add it later only when needed:

```bash
python -m pip install --no-cache-dir --progress-bar off "deeplabcut[gui]==3.0.1"
```

## 5. Test DeepLabCut

```bash
python -c "import deeplabcut; print(deeplabcut.__version__)"
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__)"
```

Or run the project check script:

```bash
bash scripts/check_setup.sh
```

If the GUI package is installed, this should open the DeepLabCut GUI:

```bash
python -m deeplabcut
```

## 6. Current Local Status

On this machine, Miniforge is installed at:

```text
/Users/lizhiheng/miniforge3
```

The `DEEPLABCUT` environment currently imports:

```text
deeplabcut 3.0.1
torch 2.13.0
torchvision 0.28.0
```

This is CPU-only on the current install, so training can be slow. It is still enough for setup, project creation, labeling preparation, and small pipeline tests.

## 7. Do Not Move These Files

Keep these local files in the project folder:

```text
face.mp4
body.mp4
```

They are ignored by git because they are raw data and too large for normal GitHub commits.

Next: read `docs/02_eye_project.md`.
