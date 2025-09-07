# Embryo Analyzer: Project Development Plan

This document outlines the phased development plan for the Embryo Analyzer project. It serves as a roadmap from initial data exploration to a fully functional inference script.

---

### Phase 0: Setup and Data Exploration (Completed)

The goal of this phase was to establish a reproducible development environment and perform an initial sanity check on our data.

**Deliverables:**
- [x] Set up a cross-platform Conda environment (`environment.yml`).
- [x] Define the project architecture and principles (`README.md`).
- [x] Establish a clean, modular repository structure (`/src`, `/data`, `/notebooks`).
- [x] Download and organize the initial dataset (`Embryo 2.0`).
- [x] Successfully load and visualize a single embryo image in a Jupyter Notebook (`01_data_exploration.ipynb`).

---

### Phase 1: The Data Pipeline (Current Phase)

The goal of this phase is to move from loading a single image to creating a robust, reusable data pipeline that can efficiently feed batches of data to our model. This is the most critical infrastructure for the entire project.

**Deliverables:**
- [ ] **Create `src/data_loader.py`:**
    - [ ] Implement a custom PyTorch `Dataset` class named `EmbryoDataset`.
    - [ ] The `Dataset` must be able to find all image paths and their corresponding labels (stage names) from the `data/processed` directory.
    - [ ] The `Dataset` must apply necessary image transformations (e.g., resizing, converting to a PyTorch tensor, normalization).
- [ ] **Create a `create_dataloaders` function:**
    - [ ] This function will instantiate our `EmbryoDataset` for both the `train` and `test` splits.
    - [ ] It will wrap these datasets in PyTorch `DataLoader` objects, which handle batching, shuffling, and multi-threaded data loading.
- [ ] **Test the pipeline in a new notebook (`02_pipeline_testing.ipynb`):**
    - [ ] Import the functions from `src/data_loader.py`.
    - [ ] Create the dataloaders and fetch one batch of data.
    - [ ] Visualize a few images from the batch along with their labels to confirm the entire pipeline is working correctly.

---

### Phase 2: Model Architecture and Training Script

With the data pipeline in place, we can now define our model and create the script that will train it.

**Deliverables:**
- [ ] **Create `src/model.py`:**
    - [ ] Define a function that returns a pre-trained ResNet50 model from `torchvision`.
    - [ ] Modify the final classification layer of the ResNet model to match the number of our embryo stages (e.g., 5 classes).
- [ ] **Create `src/config.yaml`:**
    - [ ] Define all key hyperparameters: `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`, `IMAGE_SIZE`.
- [ ] **Create `src/train.py`:**
    - [ ] Load settings from `config.yaml`.
    - [ ] Call `create_dataloaders()` from `data_loader.py`.
    - [ ] Initialize the model from `model.py` and move it to the correct device (GPU/CPU).
    - [ ] Define the loss function (e.g., `CrossEntropyLoss`) and the optimizer (e.g., `Adam`).
    - [ ] Implement the main training loop, which iterates through epochs and batches.
    - [ ] Implement a validation loop to check the model's performance on the test set after each epoch.
    - [ ] Save the best performing model's weights to the `/models` directory.

---

### Phase 3: Experimentation and Evaluation

With the training script complete, the focus shifts to running experiments and analyzing the results to improve performance.

**Deliverables:**
- [ ] **Run the first full training:** Execute `python src/train.py`.
- [ ] **Create `03_results_analysis.ipynb` notebook:**
    - [ ] Load the training history (loss and accuracy curves) and plot them.
    - [ ] Load the best model from `/models`.
    - [ ] Evaluate the model on the test set and generate a **confusion matrix** to see which stages the model is confusing.
    - [ ] Visualize some of the model's incorrect predictions to understand its failure modes.

---

### Phase 4: Inference

The final step is to create a simple, clean script that uses our trained model to make predictions on new, unseen images.

**Deliverables:**
- [ ] **Create `src/predict.py`:**
    - [ ] The script should accept a path to a single image as a command-line argument.
    - [ ] It will load the best trained model from the `/models` directory.
    - [ ] It will load and apply the same transformations to the input image that were used during training.
    - [ ] It will run the model and print the predicted stage (class name) and the model's confidence score.
