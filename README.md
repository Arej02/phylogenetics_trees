### Project Title:

Inferring Birth–Death–Sampling Model Parameters from Simulated Phylogenetic Trees Using Neural Networks

### Summary:

This project uses simulated phylogenetic trees and neural networks to learn and predict key birth–death–sampling parameters governing evolutionary processes.

### Overview:

Phylogenetic trees encode rich information about evolutionary dynamics such as speciation, extinction, and sampling. Traditionally, inferring the underlying parameters of birth–death–sampling (BDS) models relies on likelihood-based or Bayesian methods, which can be computationally expensive and sensitive to model assumptions.

In this project, we explore a machine-learning-based alternative. We simulate large numbers of phylogenetic trees (5000) under controlled BDS processes, extract summary statistics, and train a neural network regressor to predict the true generating parameters directly from these summaries.

The project focuses on learning five parameters:
- Initial birth rate (λ₁)
- Death rate (μ)
- Sampling rate (ψ)
- Post-change birth rate (λ₂)
- Rate-shift time (t₁)

### Project Structure

```plaintext
phylogenetics/
├── docs/
│   └── parameters.md
├── notebooks/
│   ├── data_engineering.ipynb
│   ├── feature_engineering.ipynb
│   └── plot_height.ipynb
├── output/
│   └── metadata.csv
│   └── preprocessed_parameters.csv
│   └── X.npy
│   └── y.npy
├── results/
│   ├── plots/
│   │   └── loss_curve.png
│   └── report.txt
├── scripts/
│   ├── main.py
│   └── simulation.yaml
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```


### Problem Statement:

Given a phylogenetic tree generated under a piecewise birth–death–sampling process, can we accurately and efficiently infer the underlying model parameters using a neural network?

Specifically, we aim to:

- Learn a stable mapping from tree-derived features to BDS parameters
- Evaluate whether neural networks can recover parameters with reasonable accuracy compared to traditional methods

### Dataset:

The dataset is fully simulated using a configurable YAML-based simulation pipeline from python package called `Phylogenie`.

Key characteristics:
- Thousands of phylogenetic trees generated per run
- Variable number of tips per tree
- Piecewise-constant birth rates with a single rate shift at time t₁
- Explicit sampling process (ψ)

Each tree is transformed into a fixed-length feature vector (X), paired with its true generating parameters (y).

Stored files:
- output/X.npy – Feature matrix
- output/y.npy – Target parameter matrix (5 parameters)

### Tools and Technologies:

- Python – Core programming language
- NumPy / Pandas – Data handling and preprocessing
- PyTorch – Neural network implementation and training
- scikit-learn – Scaling, metrics, and data splitting
- Matplotlib – Training diagnostics and visualizations
- Phylogenie – Simulation configuration

### Methods:

1. Tree Simulation
- Trees are simulated under a canonical birth–death–sampling parameterization
- Acceptance criteria ensure sufficient tips before and after the rate shift

2. Feature Engineering
- Trees are converted into each tree into two vectors, one for split times and sampling times

3. Data Preprocessing
- Input features and targets are standardized
- Data is split into training and test sets

4. Neural Network Model
- Multi-layer perceptron (MLP) with ReLU activations
- Multi-output regression (5 parameters)
- Early stopping based on validation loss

### Key Insights:

- Neural networks can learn meaningful relationships between tree structure and BDS parameters
- Proper scaling of both inputs and targets is critical for stable training
- Some parameters (e.g., λ₂ and t₁) are more difficult to infer than others, reflecting identifiability limits

### How to run this project

1. Clone the repository
``` bash
git clone https://github.com/Arej02/phylogenetics_trees.git
cd phylogenetics_trees
```

2. Create Virtual Environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Generate simualted trees:
```bash
phylogenie scripts/simulate.yaml
```

5. Train the neural network:
```bash
python scripts/main.py
```

6. View results
- Metrics and plots saved in `results/`

### Results and Conclusion:

The trained multi-layer perceptron (MLP) achieves the following performance on the held-out test set (20% split, n=5000):

- Average R²: ≈ 0.5396
- Mean Absolute Error (MAE) per parameter (original scale):
  - λ₁ (lambda1): ≈ 0.11
  - μ (mu): ≈ 0.11
  - ψ (psi): ≈ 0.07
  - λ₂ (lambda2): ≈ 0.18
  - t₁ (t1): ≈ 0.61

Compared to initial experiments (R² ≈ 0.38, t₁ MAE ≈ 1.07), the final model shows clear improvement through:
- log1p transformation of all the target variables for stability, and mild improvement on the skewed parameters
- Architecture: MLP →hidden = [384,192,96], dropout=0.15, AdamW, lr=1e-4, weight_decay=1e-3
- better hyperparameter choices and larger batch size (256)

The model performs particularly well on ψ, t₁, and λ₂ , while μ remains the most challenging parameter due to its highly skewed distribution with many near-zero values.

Overall, the neural network provides a fast and reasonably accurate approximation of birth–death–sampling parameters from tree summary statistics — demonstrating the potential of ML as a lightweight alternative to traditional likelihood-based inference — although certain parameters exhibit limited identifiability from the chosen features alone.

### Future Work:

- Incorporate uncertainty estimation (Bayesian neural networks)
- Compare against likelihood-based and Bayesian baselines
- Extend to multiple rate shifts or state-dependent models
- Improve feature engineering using full tree topology