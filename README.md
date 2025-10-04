# Activity_Recognization_CPP
<p align="center">
  <img src="act_recogn.png" alt="Project Picture" width="500"/>
</p>


## Introduction
We present a project focused on recognizing rat activity in a Conditioned Place Preference (CPP) experiment using state-of-the-art machine learning and deep learning techniques.
This repository provides a complete pipeline, including dataset preparation, model training, evaluation, and deployment, to accurately classify and analyze behavioral patterns in experimental settings.

## :ledger: Index
- [Introduction](#introduction)
- [Dataset](#beginner-dataset)
- [Models](#beginner-models)
- [Repository Structure](#file_folder-repository-structure)
- [Installation](#electric_plug-installation)
- [Usage](#zap-usage)
- [License](#lock-license)


## :beginner: Dataset
for training pose-model we use videos of rat moving in CPP place in 3 parts.

[pose-dataset](https://drive.google.com/file/d/1lkLGwOrxNDFnX1eTDNyvd3s9VOWBSEsV/view?usp=drive_link)

[LSTM-dataset](https://drive.google.com/file/d/1H9p2U5dO5ae4EQlEwmk1eBxVLU4AaoKE/view?usp=drive_linkhttps://drive.google.com/file/d/1H9p2U5dO5ae4EQlEwmk1eBxVLU4AaoKE/view?usp=drive_link)

## :beginner: Models

We trained YOLO-Pose v11n and YOLO-Pose v11s models to construct the skeletal representation of the rat, and developed an LSTM model (from scratch) to recognize and classify its activities.

### links 
[yolo-pose_v11n](https://drive.google.com/file/d/1hxonzR52R1a-swXUbRtG55ONlc-JFRHz/view?usp=drive_link)

[yolo-pose_v11s](https://drive.google.com/file/d/1fzFr6NZxyUwEEBFYM2D3iK7wc5t1pLka/view?usp=drive_link)

[LSTM](https://drive.google.com/file/d/1ZWJBci8MMzIu1qGywGLjzdm-6Ar39s9s/view?usp=drive_link)

## :file_folder: Repository Structure
This repository contains two main folders for dataset preparation and LSTM model training:

### LSTM_dataset_prepar/

This folder is responsible for preparing the dataset (.csv file) used to train the LSTM.
Extract rat keypoints and store them in a CSV file along with activity labels.
Compute 18 additional features from the keypoints.
Normalize feature values and balance the label distribution to ensure fair training.

### LSTM_activity_recognization_feature_ext/

This folder handles the training and evaluation of the LSTM model.
Load the prepared dataset (CSV) from the dataset/ directory.
Configure training parameters in main.py and run training.
After training, update arguments in activity_recognition_lstm_model.py to test the trained model.
Provides previews and saved videos of the model’s activity recognition performance.

```bash
Activity_Recognization_CPP/
├── examples/           # Colab notebooks for experiments & demos
├── models/             # Saved models / checkpoints
├── LSTM_activity_recognization_feature_ext/
└── LSTM_dataset_prepar/
```

## :electric_plug: Installation

Use [conda](https://docs.conda.io/en/latest/)
 to create and manage the project environment, and [pip](https://pip.pypa.io/en/stable/)
 to install additional dependencies such as foobar.

```bash
# Create a new conda environment
conda create -n activity_env python=3.10

# Activate the environment
conda activate activity_env

# Install required packages with pip
pip install opencv-python
pip install torch
pip install numpy pandas matplotlib scikit-learn
```

## :zap: Usage
After installing dependencies, you can:

**Prepare dataset** → Run python files in LSTM_dataset_prepar/ to generate the CSV with features.

**Train model** → Configure arguments in main.py (inside LSTM_activity_recognization_feature_ext/) and train the LSTM.

**Test & visualize results** → Update arguments in activity_recognition_lstm_model.py to preview and save testing results.

## :lock: License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).
