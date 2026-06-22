# Digital Rock (preprocessing)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

**Author:** Honggeun Jo (Inha University)  
**Last Updated:** March 24, 2026

Welcome to the **Digital Rock** repository! Developed by researchers at **Inha University**, this project provides a comprehensive toolkit for digital rock physics workflows. It leverages Machine Learning and Neural Networks to process micro-CT rock images, perform segmentation, extract pore network models (PNM), and more.

## 🌟 Key Features

* **Image Preprocessing**: Tools for preparing and standardizing raw digital rock images for downstream analysis.
* **Neural Network Applications**:
  * **De-artifacting & De-noising**: Enhance image quality by removing scanning artifacts and noise using deep learning.
  * **Semantic Segmentation**: Advanced U-Net and variant models (`main_unet.ipynb`) for accurate multiphase segmentation (e.g., pore space, matrix, minerals).
* **Pore Network Modeling (PNM)**: Extract topological and geometric properties of the pore space to simulate fluid flow and petrophysical properties.

## 📂 Repository Structure

The repository is organized as follows to separate core scripts, neural network architectures, and interactive workflows:

```text
digital_rock/
│
├── networks/           # Contains neural network architectures (e.g., U-Net models)
├── scripts/            # Helper scripts for preprocessing, data loading, and PNM extraction
├── main.ipynb          # Standard pipeline and baseline models
├── main_unet.ipynb     # Dedicated notebook for U-Net based segmentation and denoising
├── main_v2.ipynb       # Advanced/updated experimental pipelines
├── LICENSE             # MIT License
└── README.md           # Project documentation
```
## ⚙️ Installation & Requirements

To run the notebooks and scripts in this repository, you will need a Python environment with Jupyter Notebook installed. It is highly recommended to use a virtual environment (like conda or venv).

1. Clone the repository:
   ```bash
   git clone https://github.com/geomodeller/digital_rock.git
   cd digital_rock
   ```
1. Install dependencies:
While a specific requirements.txt is not provided, the typical dependencies for these notebooks include:
   ```bash
   pip install numpy pandas matplotlib scikit-image jupyter
   pip install torch torchvision   # If using PyTorch for the Neural Networks
   # OR
   pip install tensorflow          # If using TensorFlow/Keras
   ```
(Note: Depending on the specific Pore Network Modeling tools used in the scripts, you may also need libraries like OpenPNM or porespy.)

## 🚀 Usage
The primary way to interact with this project is through the provided Jupyter Notebooks.
Open one of the main notebooks depending on your task:
 * main.ipynb: Use this for standard digital rock processing workflows.
 * main_unet.ipynb: Open this to train, validate, or infer using the U-Net architecture for image segmentation and noise reduction.
 * main_v2.ipynb: Check out the latest experimental features and updated workflows.

## 🏛️ Acknowledgements
This repository is maintained and developed by the geomodeling research team at Inha University. If you use this code in your research, please consider reaching out or properly attributing this repository.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
   
