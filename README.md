# Comparative Analysis of Models for Object Detection and Character Recognition

## Project Overview
This project explores various machine learning models to automate the process of detecting vehicle license plates and recognizing characters on them. The primary focus is on 2-wheeler vehicles to assist law enforcement by automating helmet rule enforcement through image processing techniques.

## Key Features
- **Object Detection**: Using models like OpenCV and YOLO for detecting vehicle license plates.
- **Character Recognition**: Utilizing OCR and DenseNet for extracting and recognizing characters from license plates.
- **Comparative Analysis**: Evaluating the performance of different models to determine the most efficient and accurate approach.
- **Helmet Detection**: Implementing models to detect helmet usage among two-wheeler riders.

## Models and Technologies
- **CNN**: For image processing and object detection.
- **YOLO (You Only Look Once)**: Real-time object detection.
- **DenseNet**: For character recognition from detected license plates.
- **OpenCV**: Employed for image manipulation and enhancing model inputs.
- **Pytesseract**: Optical Character Recognition to convert images to text.

## Dataset
The models are trained and tested on a dataset sourced from Kaggle, specifically designed for license plate detection and character recognition:
- **Helmet Detection**: YOLOv3 trained with images marking helmets.
- **License Plate Detection**: Focused on images of 2-wheelers, adapted from a dataset originally containing 4-wheelers.

## Repository Structure
- **src/**: Contains all source code for running the models.
- **data/**: Training and test datasets along with preprocessing scripts.
- **models/**: Trained models and configuration files for quick deployment.
- **docs/**: Documentation on model performance and setup instructions.
- **notebooks/**: Jupyter notebooks for interactive model training and evaluation.

## Acknowledgments
- Dr. Vaibhav Kumar for his guidance and supervision.
- Master Dhawal Patil for technical support throughout the project.
- Kaggle for providing the dataset used for training the models.

## Setup and Installation
Ensure you have Python 3.8+ installed, then clone this repository and install required packages:
```bash
git clone https://github.com/paithanemehul/Comparative-Analysis-of-MS-Models-for-Helmet-and-License-Plate-Recognition.git
cd <repository-name>
pip install -r requirements.txt
To run the object detection and character recognition models:
python src/detect.py


