# Notebook Setup

This repository is intended to be run only in Kaggle or Colab notebooks with a
T4 GPU.

## Kaggle

1. Create a new Kaggle notebook.
2. Set the accelerator to `T4 GPU`.
3. Attach this repository as a Kaggle dataset or clone it into the notebook.
4. Install the Python dependencies.
5. Run `demo.py` or `infer.py`.

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

Custom sample pair on Kaggle:

```python
!python infer.py --s2 /kaggle/working/path/to/foo_s2.tif --corridor /kaggle/working/path/to/foo_corridor.tif
```

## Colab

1. Create a new Colab notebook.
2. Set the runtime type to `T4 GPU`.
3. Clone the repo from GitHub or unzip it from Google Drive.
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

Custom sample pair on Colab:

```python
!python infer.py --s2 /content/path/to/foo_s2.tif --corridor /content/path/to/foo_corridor.tif
```
