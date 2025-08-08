# Handwritten Digit Generation with Gaussian Mixture Models

## 📌 Project Description
This project demonstrates how to model and generate handwritten digits from the `digits` dataset in `scikit-learn` using Gaussian Mixture Models (GMM). Each digit (0-9) is modeled separately

## Project Goal

The goal of this project is to explore probabilistic modeling of image data by training a separate Gaussian Mixture Model for each handwritten digit. By doing so, the project aims to generate new, synthetic digit images that resemble the original data, showcasing the generative capabilities of GMMs for image synthesis.

## 🧠 Technical Concepts Demonstrated
- **Unsupervised Learning**: Modeling data distributions without labels
- **Probabilistic Modeling**: Using GMMs to capture digit variations
- **Data Generation**: Creating new samples from learned distributions
- **Dimensionality Handling**: Working with 64-dimensional pixel space (8×8 images)

## 🛠️ Installation & Usage

### Requirements
- Python 3.8+
- NumPy
- matplotlib
- scikit-learn

Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
