# 01 Local Setup

Use Terminal for environment setup and VS Code for reading/editing project files.
Use the DeepLabCut GUI for frame extraction, labeling, training, and video inspection at the beginning.

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

## 4. Test DeepLabCut

```bash
python -c "import deeplabcut; print(deeplabcut.__version__)"
python -m deeplabcut
```

The second command should open the DeepLabCut GUI.

## 5. Do Not Move These Files

Keep these local files in the project folder:

```text
face.mp4
body.mp4
```

They are ignored by git because they are raw data and too large for normal GitHub commits.

Next: read `docs/02_eye_project.md`.
