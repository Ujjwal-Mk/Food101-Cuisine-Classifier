# Food101-Cuisine Classifier

## Overview

This repository contains the implementation of a food image analysis project aimed at classifying dishes based on their visual cues. The project utilizes an interdisciplinary approach, combining computer vision, machine learning, and food science to accurately classify dishes into various cuisine categories.

## Problem Statement

Classifying the cuisine of a dish solely from its image poses several challenges, including visual ambiguity, limited diverse data, and the difficulty of incorporating new cuisines without retraining models. This project addresses these limitations by developing a robust and scalable food image analysis system.

## Literature Review

The project draws insights from various research papers, including:

- "Food-101 – Mining Discriminative Components with Random Forests"
- "Combining Weakly and Webly Supervised Learning for Classifying Food Images"
- "Food Image Classification with Convolutional Neural Networks"
- "Wide-Slice Residual Networks for Food Recognition"

These papers provide valuable methodologies, datasets, and metrics for training and evaluating food image classification models.

## Proposed Solution

The proposed solution advocates for a Neural Network-centric approach, leveraging the TensorFlow library. The implementation involves a modified dataset of the Food-101 dataset, wherein input data instances are categorized into cuisine classes. This categorization aids in classifying food images accurately.

The anticipated model surpasses a minimum accuracy threshold of 80% using a Convolutional Neural Network architecture. This architecture can be custom-designed or implemented through Transfer Learning followed by Fine Tuning.


## Repository Contents

- **backend.py**: Contains the backend logic for the food image classifier.
- **cuisine.ipynb**: Jupyter Notebook containing code for analyzing cuisine data.
- **cuisine_79_6.h5**: Pre-trained model for classifying food images with an accuracy of 79.6%.
- **food101.ipynb**: Jupyter Notebook containing code for preprocessing the Food-101 dataset.
- **frontend.py**: Implements the frontend interface for interacting with the food image classifier.
- **helper_functions.py**: Utility functions used across different parts of the project.
- **model.png**: Visualization of the neural network architecture used for image classification.
- **model_84_9.h5**: Pre-trained model for classifying food images with an accuracy of 84.9%.
- **rnd.ipynb**: Jupyter Notebook containing random experiments and explorations.

## Contributor

Special thanks to [@VarshaRohidekar](https://github.com/VarshaRohidekar) for their contributions to this project.

## Contributing

Contributions to this project are welcome. If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## Acknowledgments

- The authors of the referenced research papers for their valuable contributions to the field of food image classification.
- The TensorFlow community for providing a powerful framework for building and training neural networks.
