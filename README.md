# Medical AI X-ray Pneumonia Detection

A Medical AI demo for chest X-ray pneumonia detection using deep learning, Streamlit, and explainable Grad-CAM heatmaps.

## Overview

This project demonstrates a computer vision workflow for medical image analysis.  
The system is designed to classify chest X-ray images as either **Normal** or **Pneumonia** and provide visual explanation using Grad-CAM heatmaps.

The goal is to present a practical AI demo for healthcare imaging workflows, focusing on:

- Chest X-ray classification
- Explainable AI with Grad-CAM
- Interactive Streamlit deployment
- Clean clinical-style result presentation

## Features

- Upload chest X-ray images
- Predict Normal or Pneumonia
- Show confidence score
- Generate Grad-CAM heatmap
- Display original image and explanation side by side
- Medical disclaimer for responsible AI use

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Streamlit

## Project Structure

```text
medical-ai-xray/
│
├── app.py
├── train.py
├── predict.py
├── gradcam.py
├── requirements.txt
├── README.md
│
├── model/
├── sample_images/
└── assets/
