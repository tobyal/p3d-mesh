# P3D-Mesh: Patch-Wise Pseudo-Surface Prior Guided Mesh Reconstruction from Sparse Point Clouds

## Installation

Create an environment and install the main dependencies:

```bash
conda create -n your_env python=3.9
conda activate your_env
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install lightning open3d==0.16 trimesh pyhocon mcubes tqdm einops
```

In addition, you also need to install some C++ libraries:

- CGAL
- boost
- libgmp-dev

Before starting, you need to compile several C++/CUDA extensions.

For example:

```bash
cd models/Chamfer3D
python setup.py install

cd ../pointops
python setup.py install

cd models/cpplib
conda activate your_env
python setup.py build_ext --inplace
```

## Stage 1: PSP Inference

### Data preparation

- PSP pretrained weights/Test inputs: [Google Drive](https://drive.google.com/drive/folders/1vst3YBF-WEIGMG_T5_fPLj5iyk0OOapX?usp=drive_link)

1. Download the pretrained PSP checkpoint and put it into:

```text
pretrain/psp/
```

2. Put the sparse input point clouds (2048) into:

```text
dataset/test/<point_cloud_dataset_name>/
```

3. Open `configs/psp.yaml` and modify the fields under `exp`:

```yaml
exp:
  data_name: <point_cloud_dataset_name>
  exp_name: <experiment_name>
```

- `data_name`: your dataset folder name under `dataset/test/`
- `exp_name`: your experiment name

### Run

```bash
python test_psp.py
```

## Stage 2: SDF Training and Mesh Extraction

### Data preparation

Before running:

```bash
python train_gsdf.py
```

please do the following:

1. Open `configs/gsdf.yaml`
2. Modify:

```yaml
exp:
  data_name: <point_cloud_dataset_name>
  exp_name: <experiment_name>
```

- `data_name`: the point cloud dataset name
- `exp_name`: the experiment name

### Run

```bash
python train_gsdf.py
```

### Output

The results will be saved in `output/`.

## Acknowledgements

Our code is based on the following GitHub repositories:

1. [RepKPU](https://github.com/EasyRy/RepKPU/tree/main)
2. [LightweightMR](https://github.com/CharizardChenZhang/LightweightMR/tree/main)
