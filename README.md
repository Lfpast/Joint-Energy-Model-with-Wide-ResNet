# Joint Energy Model (JEM) with Wide ResNet for CIFAR-10

This folder contains a modular Python package that implements a Joint Energy Model (JEM) using a Wide Residual Network (WRN) for classification and energy-based generation on the CIFAR-10 dataset. 

Package Layout (fishbone view)
```
root
  ├───── data/
  │  └──── preprocess.py           # data loading & preprocessing
  ├───── models/
  │  ├──── wrn.py                   # WRN and ResBlock definitions
  │  ├──── wrn_re.py                # WRNRE and ResBlockRE (with Dropout)
  │  ├──── model-20.weights.h5
  │  ├──── JEM-4.weights.h5
  │  └──── __init__.py
  ├───── train/
  │  ├──── train_part1.py           # classification training (Part 1)
  │  ├──── train_part2.py           # JEM training (Part 2)
  │  └──── __init__.py
  ├───── eval/
  │  ├──── eval.py                  # evaluation & misclassified visualization
  │  └──── __init__.py
  ├───── generation/
  │  ├──── energy_sampling.py       # energy, sampling_step, SampleBuffer
  │  └──── __init__.py
  ├───── scripts/
  │  ├──── download_data.py         # download/extract CIFAR-10 into data/
  │  ├──── run_part1.py             # run Part 1 (training + show misclassified)
  │  ├──── run_part2.py             # run Part 2 (JEM training)
  │  └──── visualize.py             # demos and visualization routines
  ├─── __init__.py
  ├─── config.py
  ├─── requirements.txt
  ├─── run_demo.py
  ├─── README.md
  └─── .gitignore
```

Usage (example, run from the workspace root):
1. Install dependencies: `pip install -r requirements.txt` (Install a GPU-enabled TensorFlow for the best performance.)
2. Download data: `python -m scripts.download_data` (this will place the CIFAR-10 dataset into `data/`)
3. Run Part 1 training: `python -m scripts.run_part1`
4. Run Part 2 training: `python -m scripts.run_part2`
5. Visualization: `python -m scripts.visualize`

Note: The package contains a data downloader at `scripts/download_data.py` (also used by the scripts) which will save `cifar-10-batches-py` under `data/`.
All model weight files produced by training will be stored in `models/`.

Notes
