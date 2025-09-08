## Transformers Workshop - September 23rd, 2025

Full resources coming soon!

RSVP [here](https://docs.google.com/forms/d/14smYLDO3aV03hVxDcVeilne-Uqch8ywKA_CdngCtClo)

Location Fintech 516 - [Directions](https://www.google.com/maps?gs_lcrp=EgZjaHJvbWUyCggAEEUYFhgeGDkyCggBEAAYChgWGB4yCAgCEAAYFhgeMggIAxAAGBYYHjIICAQQABgWGB4yCAgFEAAYFhgeMggIBhAAGBYYHjIGCAcQRRg80gEINDMyNmowajeoAgCwAgA&um=1&ie=UTF-8&fb=1&gl=us&sa=X&geocode=KffXNNduq8eJMfD2uB_J3qQy&daddr=591+Collaboration+Way,+Newark,+DE+19713)

Zoom [link](https://udel.zoom.us/j/92582067018?jst=2)

**Total Duration:** 7 hours

### Structure
- 3 hours lecturing with slides
- 30-60 minutes lunch break
- 60 minutes setup and general discussion
- 2 hours hands-on coding (Jupyter Notebooks)

### Morning Session: Lecture (3 hours)
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

#### Part 3: Internals of the Transformer Architecture (1 hour)
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

#### Part 4: NLP and PLMs (1 hour)
- Modeling words and images with transformers
- Modeling biological sequences
- Pretraining types
- Finetuning and transfer learning
- Recent findings on pretraining

### Lunch Break (30-60 minutes)

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
  - More to come…

### Hands-On Jupyter Notebooks (2 hours)
#### Notebook 1: Implementing Attention from Scratch (30 minutes)
- Manual implementation of attention using PyTorch
- Looking at more efficient implementations
- Basic MLP sections
- Full transformer block
- Comparing transformers to other network types

#### Notebook 2: Building a Minimal Transformer Encoder Block (30 minutes)
- Construction of a simplified transformer block
- Integration of LayerNorm, Multi-head Attention, Feedforward Layers
- Implementation of residual connections
- Forward pass on synthetic data
- Analysis of output hidden states

#### Notebook 3: Comparing Transformers to Other Architectures (30 minutes)
- MNIST data with:
  - Basic ViT
  - ConvNet
  - MLP

#### Notebook 4: Working with Biological Sequences and Transformers (30 minutes)
- Protify
- Probing networks
- Protein and other sequence considerations

### Workshop Wrap-Up
- Summary of major concepts
- Further reading and recommended resources
- Guidance on applying transformers to custom tasks
- Final Q&A