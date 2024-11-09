## Light-MILPopt: Solving Large-scale Mixed Integer Linear Programs with Lightweight Optimizer and Small-scale Training Dataset

### Contact

We welcome academic and business collaborations with funding support. For more details, please contact us via email at xuhua@tsinghua.edu.cn.

### Overview

This release contains the core processes of the **Light-MILPopt** framework, as described in the paper *Light-MILPopt: Solving Large-scale Mixed Integer Linear Programs with Lightweight Optimizer and Small-scale Training Dataset*. The provided code implements the main components of the approach, covering data generation, training, and inference. Interfaces are provided for flexibility, allowing easy adaptation to different contexts.

Here is a brief overview of the provided files. More detailed documentation is available within each file:

- **Code/data_generation.py**: Generates standard mixed integer linear programming (MILP) problems for training and testing datasets.
- **Code/data_solution.py**: Processes training data by generating optimal solutions for MILP problems, which are used as training labels.
- **Code/EGAT_layers.py**: Defines the layer structure of the EGAT (Enhanced Graph Attention Network) model.
- **Code/EGAT_models.py**: Defines the overall structure and architecture of the EGAT model.
- **Code/train.py**: Contains the training process for the EGAT model.
- **Code/test.py**: Carries out the testing process to evaluate the performance of the model on test datasets.

### Requirements

The required environment is specified in the provided YAML file (e.g., `Code/Light_MILPopt.yml`).

### Usage

The code should be run in the following order to perform training and testing:

1. **data_generation.py**: Generate training and testing datasets.
2. **data_solution.py**: Generate optimal solutions for training datasets.
3. **EGAT_layers.py**: Define the layers of the EGAT model.
4. **EGAT_models.py**: Define the overall EGAT model architecture.
5. **train.py**: Train the EGAT model using the generated datasets.
6. **test.py**: Run the trained model on test datasets to obtain optimized results.

### Citing this work

If you use the code or methodology from this repository, please cite the following paper:

```bibtex
@inproceedings{ye2023light,
  title={Light-MILPopt: Solving Large-scale Mixed Integer Linear Programs with Lightweight Optimizer and Small-scale Training Dataset},
  author={Ye, Huigen and Xu, Hua and Wang, Hongyan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}