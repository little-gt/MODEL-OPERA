# OPERA: Optical Phenomena Recognition and Analysis

**OPERA** is an end-to-end deep learning system designed to analyze complex optical images and deconstruct them into their fundamental physical components. By leveraging a powerful synthetic data engine and a Vision Transformer (ViT) based architecture, OPERA can identify underlying phenomena like diffraction, interference, and aberrations, and predict their key physical parameters.

## 1. System Architecture

The project is designed with a highly modular and extensible architecture, separating the core concerns of data generation, model training, and final analysis.

```
OPERA/
├── cache/                     # Stores model checkpoints for resuming training.
├── configs/                   # All YAML configuration files for experiments.
│   ├── data_generation_v1.yaml
│   └── training_config_v1.yaml
├── data/                      # Output directory for generated datasets.
├── opera_analysis/            # Modules for post-training analysis.
│   └── predictor.py           # Encapsulates model loading and inference logic.
├── opera_data/                # The synthetic data generation engine.
│   ├── phenomena/             # Library of individual, parameterizable optical phenomena.
│   │   ├── diffraction.py
│   │   ├── interference.py
│   │   └── aberrations.py
│   ├── composition.py         # Combines basic phenomena into complex scenes.
│   ├── noise_and_distortions.py # Applies realistic noise and sensor effects.
│   ├── label_schema.py        # Defines and validates the ground-truth label structure.
│   └── generator.py           # The main orchestrator for generating datasets.
├── opera_model/               # All modules related to the machine learning model.
│   ├── architecture/          # Core model architecture files.
│   │   ├── vit_encoder.py     # Vision Transformer image encoder.
│   │   ├── text_decoder.py    # Autoregressive Transformer text decoder.
│   │   └── complete_model.py  # Combines encoder and decoder into an end-to-end model.
│   ├── dataset.py             # PyTorch Dataset for loading and preprocessing data.
│   ├── tokenizer.py           # Converts text labels to and from token IDs.
│   └── engine.py              # The core Trainer class for the training/validation loop.
├── results/                   # Output for tokenizers and other training artifacts.
└── scripts/                   # Executable scripts to run the project workflow.
    ├── 1_generate_dataset.py  # Script to generate a new dataset.
    ├── 2_train_model.py       # Script to train the model on a dataset.
    └── 3_analyze_image.py     # Script to analyze a new image with a trained model.
```

## 2. Usage and Code Examples

Follow these steps to set up the environment and run the complete project workflow.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd OPERA
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Create a `requirements.txt` file** with the following content:
    ```
    torch
    torchvision
    numpy
    opencv-python-headless
    matplotlib
    PyYAML
    pydantic
    tqdm
    Pillow
    scipy
    ```

4.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Workflow

The project is designed to be run in three distinct steps.

#### Step 1: Generate the Dataset

First, we generate a synthetic dataset of optical images and their corresponding JSON labels.

1.  **Edit the configuration:** Open `configs/data_generation_v1.yaml` and adjust parameters like `num_samples` and `image_resolution` to your needs. For a serious training run, `num_samples` should be at least 10,000.
2.  **Run the script:**
    ```bash
    python scripts/1_generate_dataset.py --config configs/data_generation_v1.yaml
    ```
    This will create the dataset in the `data/synthetic_dataset_v1` directory and supports resuming if interrupted.

#### Step 2: Train the Model

Next, we train our model on the newly generated data.

1.  **Edit the configuration:** Open `configs/training_config_v1.yaml`. You can set the `batch_size`, `num_epochs`, `learning_rate`, and `model_size`.
2.  **Run the training script:**
    ```bash
    python scripts/2_train_model.py --config configs/training_config_v1.yaml
    ```
    This will:
    *   Build a tokenizer from your dataset labels on the first run.
    *   Split the data into training and validation sets.
    *   Instantiate the model.
    *   Begin training. Checkpoints will be saved in `cache/checkpoints/`, allowing you to stop and resume training at any time by simply re-running the command.

#### Step 3: Analyze a New Image

After training, use the best-performing model to analyze a new image.

1.  **Choose an image:** Pick any image, for example, one from your generated dataset like `data/synthetic_dataset_v1/images/some_image_id.png`.
2.  **Run the analysis script:**
    ```bash
    # Replace the --image argument with the path to your image
    python scripts/3_analyze_image.py --image data/synthetic_dataset_v1/images/syn_1116160.png --config configs/training_config_v1.yaml
    ```
    The script will load the best model from the specified training run and print a JSON object describing the predicted physical components of the input image.

## 3. Other Technical Details

### Data Engine

*   **Physics-Based Phenomena:** The data engine simulates phenomena based on established physical models:
    *   **Diffraction:** Fraunhofer diffraction for single-slits and circular apertures (Airy disk) using the Sinc and Bessel functions.
    *   **Interference:** Young's double-slit interference, modeled as a product of interference and diffraction terms.
    *   **Aberrations:** Complex wavefront errors are modeled using **Zernike Polynomials**, allowing for precise simulation of defocus, astigmatism, coma, and spherical aberration.
*   **Composition Engine:** Can combine phenomena in two ways:
    *   **Sequential:** Simulates light passing through multiple elements in a series (e.g., `Aberration(Diffraction(source))`).
    *   **Superposition:** Simulates the non-coherent sum of multiple light patterns.
*   **Realistic Noise Model:** A chain of effects is applied to simulate a real sensor pipeline: geometric distortions, Poisson shot noise, Gaussian read noise, and finally ADC quantization and saturation.
*   **Validated Labels:** Ground-truth labels are validated against a strict schema using `Pydantic`, ensuring data integrity.

### Model Architecture

*   **Image-to-Sequence (Image-to-Text):** The problem is framed as a translation task where an image is "translated" into a JSON string that describes it.
*   **ViT Encoder:** A standard Vision Transformer (ViT) processes the input image. It divides the image into patches, embeds them, and uses a stack of self-attention layers to build a rich feature representation.
*   **Transformer Decoder:** A standard autoregressive Transformer decoder receives the image features from the encoder and generates the output text token by token, using masked self-attention and cross-attention.

### Extensibility

The modular design makes the system easy to extend:
*   **To add a new physical phenomenon:** Simply create a new function in a file within the `opera_data/phenomena/` directory and add it to the `_function_map` in `opera_data/generator.py`.
*   **To add a new model size or architecture:** Add a new configuration to the `MODEL_CONFIGS` dictionary in `scripts/2_train_model.py` and reference it in your YAML config file.

---
