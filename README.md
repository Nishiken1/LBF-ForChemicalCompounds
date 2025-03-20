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
├── data/                 # Datasets used in experiments (PCBA, PFAS)
├── figures/              # Graphs and figures from experiments
├── models/               # Trained model files
├── scripts/              # Code for training, evaluating, and reproducing results
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt      # Required Python libraries
└── README.md
```

## Datasets
The experiments utilized the following datasets:

- PCBA: PubChem BioAssays from MoleculeNet.
- PFAS: Custom-created dataset from the PubChem compounds database.

Datasets are located in the data/ directory or can be generated using provided scripts.

## Requirements
- Python >= 3.8
- PyTorch
- RDKit
- NumPy, Pandas, Matplotlib
- Install required packages via:

```
pip install -r requirements.txt
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
@article{nishida2024lbf,
  title={Efficient Compound Search Using Learned Bloom Filter,
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2025}
}
```

## License

## Contact
For questions or feedback, please open an issue or contact [kenmancf@gmail.com].
