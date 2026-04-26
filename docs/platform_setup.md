# Platform Setup

## Local Windows Setup

This repository now includes a local setup script that uses `uv` to create a
Python 3.11 virtual environment and install the inference dependencies.

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_local.ps1
```

Then activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the bundled smoke test:

```powershell
python demo.py
```

Run one bundled sample:

```powershell
python infer.py --sample karnataka_nc_g1_r0010
```

Run the Streamlit demo:

```powershell
streamlit run app.py
```

## Kaggle Setup

Practical flow for Kaggle notebooks:

1. Clone the GitHub repo or upload a zip of the repo as a Kaggle dataset.
2. Create a new Kaggle notebook.
3. Attach the repo files in the notebook UI.
4. Install the missing Python packages.
5. Run `demo.py`, `infer.py`, or `app.py` from `/kaggle/working`.

Notebook cells:

```python
!cp -r /kaggle/input/YOUR_DATASET_SLUG/* /kaggle/working/repo
%cd /kaggle/working/repo
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
!python demo.py
```

Single-sample run on Kaggle:

```python
!python infer.py --sample karnataka_nc_g1_r0010
```

Notes:
- The packaged inference path loads the bundled checkpoint locally by default.
- If `app.py` complains about a missing dependency, install `requirements.txt`
  inside the notebook first.
- If you ever need remote model downloads, pass `--allow-backbone-download`
  and enable internet intentionally.

## Colab Setup

Practical flow for Colab:

1. Put this repo on GitHub or upload a zip to Google Drive.
2. Open a new Colab notebook.
3. Either clone the repo from GitHub or unzip it from Drive into `/content`.
4. Install the Python dependencies.
5. Run `demo.py` or `infer.py`.

Option A: clone from GitHub

```python
!git clone https://github.com/YOUR_USER/YOUR_REPO.git
%cd YOUR_REPO
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
!python demo.py
```

Option B: load from Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!cp /content/drive/MyDrive/path/to/bangers.zip /content/
!unzip -q /content/bangers.zip -d /content/repo
%cd /content/repo
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
!python demo.py
```

Single-sample run on Colab:

```python
!python infer.py --sample karnataka_nc_g1_r0010
```

Custom sample pair:

```python
!python infer.py --s2 /content/path/to/foo_s2.tif --corridor /content/path/to/foo_corridor.tif
```
