# embryo-analyzer

## AI-Powered Embryo Stage Analyzer

This project uses a deep learning model to classify the developmental stage of human embryos from microscope images.

## Setup

1.  Clone the repository:
    `git clone <your-repo-url>`
2.  Navigate to the project directory:
    `cd embryo-analyzer`
3.  Create and activate the Conda environment:
    `conda env create -f environment.yml`
    `conda activate embryo_env`
4.  Install dependencies:
    `pip install -r requirements.txt`

## Usage

### Training a new model
To train the model, run the main training script:
`python src/train.py`

### Making a prediction
To classify a new image, use the prediction script:
`python src/predict.py --image_path /path/to/your/image.tif`


## Project Architecture & Guiding Principles

This document outlines the software architecture and core principles for the Embryo Analyzer project. All future development should adhere to these standards to ensure the project remains modular, maintainable, and reproducible.

1. Core Philosophy
The project is built on three foundational ideas:
Data-Centric: The quality and organization of our data are paramount. The model is only as good as the data it's trained on.
Modular: Each part of the system has a single, well-defined responsibility. This allows for independent development, testing, and improvement of each component.
Reproducible: The entire workflow, from environment setup to model training, must be easily reproducible by any developer on any supported operating system (Windows, macOS, Linux).

2. Component Breakdown
The project is organized into a clear directory structure, with each component serving a specific purpose.
/data/: The source of truth for all data.
raw/: Contains the original, unmodified datasets as downloaded from their source.
processed/: Contains the cleaned, organized, and pre-processed data ready to be consumed by the training scripts. The data here is typically structured in class-specific folders (e.g., 2-cell/, 4-cell/).
/notebooks/: The "research lab" of the project.
Purpose: For experimentation, data exploration, and model prototyping only.
Rule: Notebooks are for scratch work. Any code that proves useful must be refactored into a reusable function or class and moved into the /src directory.
/src/: The "production engine" of the project.
Purpose: Contains all the reusable, production-quality Python scripts.
config.yaml: A central file for all hyperparameters and settings (e.g., learning rate, image size, batch size). No magic numbers should be hardcoded in the scripts.
data_loader.py: Responsible for loading images from the /data/processed directory, applying augmentations, and preparing data batches for the model.
model.py: Defines the neural network architecture (e.g., a ResNet50 model adapted for our specific task).
train.py: The main script that orchestrates the entire training process. It initializes the model and data loader, runs the training loop, and saves the final model weights.
predict.py: A script for performing inference. It loads a pre-trained model from /models and classifies a single new image.
/models/: The output of the training process.
Purpose: Stores the final, trained model weight files (e.g., .pth files for PyTorch). These are the valuable assets that our application uses.

3. The Data & Model Lifecycle
The end-to-end workflow follows a clear, linear path:
Data Ingestion: A new raw dataset is downloaded and placed in /data/raw. A script or notebook is used to process it into the required class-folder structure in /data/processed.
Exploration & Prototyping: A new notebook is created in /notebooks to explore the new data and experiment with different model architectures or data augmentations.
Training: The src/train.py script is executed. It reads settings from config.yaml, uses data_loader.py to feed in data, trains the architecture defined in model.py, and saves the final weights to the /models directory.
Inference: The src/predict.py script is used to load a saved model from /models and make a prediction on a new, unseen image.

4. Guiding Principles for Development
Configuration over Hardcoding: Never place values like learning rates, image dimensions, or file paths directly into the Python code. Add them to config.yaml and load them in the script.
Scripts for Production, Notebooks for Exploration: If you write a function in a notebook that you use more than once, it belongs in a .py file in the /src directory.
Type Hinting and Docstrings: All functions and classes in the /src directory should have clear type hints and docstrings explaining their purpose, arguments, and return values. This makes the code self-documenting.

```Python
# Example of a well-documented function
def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """
    Resizes an input image to a specified size.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        size (tuple[int, int]): The target size (width, height).

    Returns:
        np.ndarray: The resized image.
    """
    return cv2.resize(image, size)
```

Hardware Agnostic Code: All PyTorch code must be hardware-agnostic by using a device-agnostic pattern (device = get_device(), model.to(device)) to ensure it runs seamlessly on CPUs, NVIDIA GPUs, and Apple Silicon.