# Machine Learning - Date Fruit Recognition

This project focuses on the development of a machine learning model for the multi-class classification of date fruits. Utilizing TensorFlow and Keras, we've constructed a Sequential neural network tailored for this task.

## Dataset

The dataset used in this project is the "Date Fruit Datasets" available on Kaggle. It consists of images of various types of date fruits, suitable for multi-class classification tasks.

- **Accessing the Dataset:** To access and download the dataset, visit [Date Fruit Datasets on Kaggle](https://www.kaggle.com/datasets/muratkokludataset/date-fruit-datasets).

### Data Preprocessing
The dataset undergoes several preprocessing steps to ensure optimal model performance:
- **Feature Scaling:** Min-Max scaling is applied to normalize the feature values.
- **Label Encoding:** Target variables are encoded to facilitate model training.
- **Standardization:** The StandardScaler from Scikit-learn is used to standardize features, achieving zero mean and unit variance.

### Model Implementation
The neural network model comprises:
- **Model Architecture:** A Sequential model with five dense layers, including dropout regularization to prevent overfitting.
- **Activation Function:** ReLU is used in hidden layers, with a Softmax activation in the output layer for multi-class classification.
- **Optimizer:** Stochastic Gradient Descent (SGD) with Batch Normalization, selected for its efficiency and performance, showing a test accuracy of 91.85%.

## Usage

To replicate or further develop this project, ensure you have TensorFlow, Keras, and Scikit-learn installed in your Python environment. The model can be trained and evaluated using the provided scripts, with customization options available for model architecture and training parameters.

## Contributing

Contributions to improve the model or extend its functionality are welcome. Please submit pull requests for review.

## License

[See the LICENSE file for details.](https://creativecommons.org/publicdomain/zero/1.0/)

## Acknowledgments

- TensorFlow and Keras Teams for providing the deep learning framework.
- Scikit-learn Contributors for preprocessing tools.
