# Embryo Analyzer: Project Development Plan

This document outlines the phased development plan for the Embryo Analyzer project. It serves as a roadmap from initial data exploration to a fully functional inference script.

---

### Strategy Update: Addressing Domain Shift

A key analysis has revealed a significant difference in quality and characteristics ("domain shift") between the public "Embryo 2.0" dataset and our target inference images. To ensure the final model is accurate on our specific data, this plan has been updated to include a **two-stage transfer learning and fine-tuning strategy**.

1.  **Pre-training:** We will first train a "base model" on the large, high-quality public dataset to learn the general features of embryo morphology.
2.  **Fine-tuning:** We will then take this base model and continue its training on a smaller, dedicated dataset of our own images. This will adapt the model to the specific nuances (lighting, noise, resolution) of our target domain.

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

### Phase 1: The Data Pipeline (Completed)

The goal of this phase was to create a robust, reusable data pipeline that can efficiently feed batches of data to our model.

**Deliverables:**
- [x] **Create `src/data_loader.py`:**
    - [x] Implemented a custom PyTorch `EmbryoDataset` class.
    - [x] Implemented a `create_dataloaders` function for train/test splits.
- [x] **Test the pipeline (`02_pipeline_testing.ipynb`):**
    - [x] Confirmed the pipeline can correctly load and batch data.

---

### Phase 1.5: Curation of Fine-Tuning Dataset (Next Critical Step)

This is a critical phase to address the domain shift. The goal is to create a small, high-quality dataset of our own images.

**Deliverables:**
- [ ] **Collect and Label Images:**
    - [ ] Gather a representative sample of our own low-quality embryo images (target: 50-100 images per class).
    - [ ] Accurately label each image with its correct developmental stage.
- [ ] **Organize the Dataset:**
    - [ ] Create a new directory, e.g., `/data/processed/my_data/`.
    - [ ] Structure the collected images into the same `stage/split/image.png` format as the original dataset (e.g., `my_data/2cell/train/...`).

---

### Phase 2: Model Architecture & Pre-training (Completed ✓)

The goal was to define our model architecture and create a complete training pipeline for the embryo classification task.

**Deliverables:**
- [x] **Create `src/model.py`:**
    - [x] Define a function that returns a pre-trained ResNet50 model from `torchvision`.
    - [x] Modify the final classification layer to match the number of our embryo stages (5 classes).
    - [x] Implement proper layer freezing for transfer learning.
- [x] **Create `src/config.yaml`:**
    - [x] Define comprehensive hyperparameters: `LEARNING_RATE`, `BATCH_SIZE`, `NUM_EPOCHS`, `IMAGE_SIZE`.
    - [x] Include optimizer settings, scheduler configuration, and training parameters.
- [x] **Create `src/train.py`:**
    - [x] Implement complete training and validation loops with progress tracking.
    - [x] Add model checkpointing and best model saving functionality.
    - [x] Include comprehensive logging and configuration management.
- [x] **Successfully complete full training pipeline:**
    - [x] Completed 20-epoch full training on Embryo 2.0 dataset.
    - [x] Generated best_model_full_training.pth (94MB) - production-ready model.
    - [x] Created checkpoints for all 20 epochs for recovery/inspection.
    - [x] Achieved baseline performance on high-quality embryo dataset.

---

### Phase 3: Fine-Tuning the Model

The goal is to adapt the pre-trained base model to our specific, lower-quality images using the dataset created in Phase 1.5.

**Deliverables:**
- [ ] **Update `src/train.py` (or create `src/fine_tune.py`):**
    - [ ] Add functionality to load the weights from a pre-existing model (e.g., `/models/base_model.pth`).
    - [ ] Add the ability to "freeze" the early layers of the model so that only the later layers are trained.
- [ ] **Update `src/config.yaml`:**
    - [ ] Add a new section for fine-tuning hyperparameters, including a much **lower learning rate**.
- [ ] **Run the Fine-Tuning Process:**
    - [ ] Execute the training script, configured to use the `my_data` directory and load the `base_model.pth`.
    - [ ] Save the final, best performing model to `/models/fine_tuned_model.pth`.

---

### Phase 4: Evaluation of the Final Model

With the final, fine-tuned model, we will perform a rigorous evaluation to understand its performance and limitations.

**Deliverables:**
- [ ] **Create `03_results_analysis.ipynb` notebook:**
    - [ ] Load the training history for both pre-training and fine-tuning and plot the curves.
    - [ ] Load the final `fine_tuned_model.pth` from `/models`.
    - [ ] Evaluate the model on the **test split of our own data** and generate a **confusion matrix**.
    - [ ] Visualize some of the model's incorrect predictions to understand its failure modes.

---

### Phase 5: Inference

The final step is to create a clean script that uses our fully fine-tuned model to make predictions on new, unseen images.

**Deliverables:**
- [ ] **Create `src/predict.py`:**
    - [ ] The script should accept a path to a single image as a command-line argument.
    - [ ] It must load the **`fine_tuned_model.pth`** from the `/models` directory.
    - [ ] It will load and apply the same transformations to the input image that were used during training.
    - [ ] It will run the model and print the predicted stage (class name) and the model's confidence score.
