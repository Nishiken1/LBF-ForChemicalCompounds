# Efficient Compound Search Using Learned Bloom Filter (LBF)

## Overview
This repository provides the source code and experimental data related to our research paper titled **"Efficient Compound Search Using Learned Bloom Filter."**

We propose a highly efficient chemical compound search method that integrates a Learned Bloom Filter (LBF) and minGRU to significantly reduce memory usage while maintaining high accuracy and rapid inference speed.

## Key Contributions
- Integration of minGRU into Learned Bloom Filter (LBF).
- Improved memory efficiency compared to conventional Bloom Filters (BF) and the original LBF.
- Enhanced inference performance, particularly suitable for large-scale compound databases.

## Repository Structure
```plaintext
.
├── datasets/              # Datasets used in experiments (PCBA, PFAS)
├── figures/               # Graphs and figures generated from experiments
├── results/               # Experimental results and evaluation outputs
├── savedmodel/            # Saved model checkpoints and trained parameters
├── backup_filter.py       # Implementation of the backup Bloom Filter
├── bloom_filter_utils.py  # Utility functions for Bloom Filter operations
├── pre_filter.py          # Implementation of the minGRU-based pre-filter model
├── run_model.py           # Script for running inference using trained LBF models
├── train.py               # Training script for minGRU-based LBF
├── requirements.txt       # Python dependencies required for the project
└── README.md              # Project overview, setup, and instructions
```

## Datasets
The experiments utilized the following datasets:

- **[PCBA](https://moleculenet.org/datasets-1)**: PubChem BioAssays from [MoleculeNet](https://moleculenet.org/).
- **PFAS**: Custom-created dataset from the [PubChem](https://pubchem.ncbi.nlm.nih.gov/) compounds database.

Datasets are located in the `data/` directory or can be generated using provided scripts.

## Requirements

### Experimental Environment
All experiments were performed under the following computing environment:

- **OS**: Ubuntu 22.04 LTS
- **CPU**: Intel Xeon w5-3423 @ 4.200 GHz (12 cores / 24 threads)
- **Memory**: 125 GiB
- **GPU**: NVIDIA RTX A1000 (VRAM: 10 GB)
- **GCC Version**: 11.4.0
- **Python Version**: 3.11.5
- **PyTorch Version**: 2.4.1+cu121

### Hyperparameter Settings
The following hyperparameters were used during model training:

- **Epochs**: 10
- **Learning Rate**: \(1 \times 10^{-4}\)
- **Batch Size**: 1024
- **Hidden Layer Sizes**: \(\{8, 16, 64, 128, 256\}\)
- **Maximum Sequence Length**: 50

### Python Libraries
- Python >= 3.8 (Recommended: 3.11.5)
- PyTorch >= 2.4.1
- RDKit
- NumPy, Pandas, Matplotlib
```
 -r requirements.txt
```

## Usage
### Training
To train the LBF model with minGRU, run:
```
python scripts/train.py --dataset [PCBA/PFAS] --hidden_size [8/16/64/128/256]
```
### Evaluation
To evaluate trained models, use:
```
python scripts/evaluate.py --model_path [PATH_TO_MODEL] --dataset [PCBA/PFAS]
```

## Results
The experimental results demonstrate that our proposed minGRU-based LBF significantly reduces memory usage and maintains comparable inference speed compared to traditional GRU-based methods. Detailed results and analyses are available in the figures/ directory and in the associated publication.

## Citation
If you use this work in your research, please cite our paper:

```
@article{nishida2025lbf,
  title={Efficient Chemical Compounds Search Using Learned Bloom Filter},
  author={Ken Nishida and Katsuhiko Hayashi and Hidetaka Kamigaito and Hiroyuki Shindo},
  journal={Journal Name},
  year={2025}
}
```

## License

## Contact
For questions or feedback, please open an issue or contact [kenmancf@gmail.com].
