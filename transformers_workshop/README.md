## Transformers Workshop - September 23rd, 2025

Part 1 of the workshop, the lecture, is now on [YouTube](https://youtu.be/TXPSDArW5MY):
https://youtu.be/TXPSDArW5MY

You can find the slides [here](https://docs.google.com/presentation/d/1kQoffVAUM716_ets9Eqkqh8KQNHYwwDv4YhIJqPHUaU/edit?usp=sharing):
https://docs.google.com/presentation/d/1kQoffVAUM716_ets9Eqkqh8KQNHYwwDv4YhIJqPHUaU/edit?usp=sharing

**Total Duration:** 6 hours

### Structure
- 2 hours lecturing with slides
- 60 minutes setup and general discussion
- 30-60 minutes lunch break
- 2 hours hands-on coding (Jupyter Notebooks)

### Morning Session: Lecture (2 hours)
#### Part 1: Linear Algebra for Machine Learning and Neural Networks (30 minutes)
- Scalars, vectors, and matrices: notation and interpretation
- Dot products and similarity metrics (cosine similarity, Euclidean distance)
- Representing input data in vector spaces
- Linear regression:
  - Least squares and optimization
  - Ridge, Lasso, and Elastic Net regularization
  - Connection between regularization and neural network weight behavior

#### Part 2: From Linear Models to Multilayer Perceptrons (30 minutes)
- Linear layers and layer composition
- Projecting data through vector spaces with weight matrices
- Activation functions: ReLU, GELU, Tanh
- Optimization methods:
  - Gradient descent
  - Mini-batch SGD
  - Adam, AdamW
- Loss surfaces and the role of optimization in training

#### Part 3: Internals of the Transformer Architecture (30 minutes)
- Overview of the transformer model
- Token embeddings and special tokens (CLS, padding)
- Self-attention mechanism:
  - Queries, keys, and values
  - Scaled dot-product attention
  - Multi-head attention
- MLP / feedforward network
- Intuition and interpretations:
  - Information mixing
  - Retrieval
  - Projections
- Other layers:
  - Residual connections, Layer Normalization, Dropout
  - Positional encodings
  - Output heads for classification and sequence modeling

#### Part 4: NLP and PLMs (30 minutes)
- Modeling words and images with transformers
- Modeling biological sequences
- Pretraining types
- Finetuning and transfer learning
- Recent findings on pretraining

### Afternoon Session: Setup and Interactive Labs
#### Environment Setup and Open Discussion (60 minutes)
- Installation of Python 3.11, VSCode, and virtual environments
- Installation of PyTorch and relevant packages
- Loading and verifying Jupyter notebooks
- Open Q&A session on lecture material
- Group discussion topics:
  - Interpretability of attention mechanisms
  - Model size versus dataset requirements
  - Limitations of transformers in small-data settings


### Lunch Break (60 minutes)

### Hands-On Jupyter Notebooks (2 hours)
#### Notebook 1: Intuition for ML
- Basics of linear regression
- Average prototype classifier
- PyTorch 

#### Notebook 2: Building a Minimal Transformer Encoder Block (30 minutes)
- Basic attention implementations
- MLPs
- Transformer block

#### Notebook 3: Comparing Transformers to Other Architectures (30 minutes)
- MNIST with MLP vs. CNN vs. Transformer - Who will win?

#### Notebook 4: Challenge: Secondary Structure Prediction of Proteins (30 minutes)
- Use transformer networks to predict the secondary structure of proteins

### Workshop Wrap-Up
- Summary of major concepts
- Further reading and recommended resources
- Guidance on applying transformers to custom tasks
- Final Q&A
