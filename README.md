# Garbage Classification App

A Streamlit-based web application for classifying garbage images using a pre-trained Keras model and showcasing the associated Jupyter notebook. The app allows users to upload images and predict the garbage category (e.g., paper, plastic, metal) with confidence scores, while also providing a rendered view of the notebook used to develop the model.

## Overview

- **Model**: A convolutional neural network (CNN) based on the Xception architecture, trained using transfer learning for garbage classification (12 categories).
- **Frontend**: Built with Streamlit for an interactive user interface.
- **Notebook**: The Jupyter notebook (`garbage-classification-transfer-learning.ipynb`) is rendered as HTML for transparency and reproducibility.


## Features

- Upload and classify garbage images (JPEG, JPG, PNG).
- Display predicted class and confidence percentage.
- Showcase the original Jupyter notebook as an interactive HTML page.
- Support for 320x320 pixel RGB images, preprocessed using Xception input norms.

## Prerequisites

- **Python**: 3.7–3.9 (recommended: 3.9 for compatibility with TensorFlow 2.17.0 or 2.3.0).
- **pip**: For installing Python packages.
- **Git** (optional): For version control if using a repository.

## Installation

### 1. Clone or Download the Repository
If using Git:
```bash
git clone https://github.com/your-username/garbage-classification-app.git
cd garbage-classification-app
