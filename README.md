# High Accuracy Battery SOC Estimation Models Comparative Analysis

## About the Project

This study was conducted as an Electrical Engineering (ELK) Graduation Project. The project involves a comparative analysis of high-accuracy battery State of Charge (SoC) estimation models. Within the scope of the project, Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Artificial Neural Network (ANN) models were used.

## Project Team

- **Turan Büyükkamacı** - Project Owner
- **Mehmet Onur Gülbahçe** - Advisor
- **Arda Akyıldız** - Graduate Student Assistant

## Contents

- [Data Preprocessing](#data-preprocessing)
- [Model Descriptions](#model-descriptions)
  - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
  - [Recurrent Neural Network (RNN)](#recurrent-neural-network-rnn)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
- [Results and Analysis](#results-and-analysis)
- [Technologies Used](#technologies-used)
- [Installation](#installation)

## Data Preprocessing

The dataset used in this project includes battery data for different temperature and C-rate values. The data was processed using the [process_experimental_data.py](https://github.com/brosaplanella/TEC-reduced-model/blob/main/tec_reduced_model/process_experimental_data.py) file. Data preprocessing steps include standardization and conversion to sequences.

## Model Descriptions

### Artificial Neural Network (ANN)

The ANN model estimates battery SoC using features in the dataset. The model is designed as a three-layer artificial neural network. For detailed explanations and code, you can review the [Battery - ANN] file.

### Recurrent Neural Network (RNN)

The RNN model estimates battery SoC using time series data. The model is trained with sequence data. For detailed explanations and code, you can review the [Battery - RNN] file.

### Convolutional Neural Network (CNN)

The CNN model estimates battery SoC using convolutional layers. The model takes sequence data as input and predicts the SoC value as output. For detailed explanations and code, you can review the [Battery - CNN] file.

### Long Short-Term Memory (LSTM)

The LSTM model is a type of RNN designed to learn long-term dependencies. This model is used for battery SoC estimation. For detailed explanations and code, you can review the [Battery - LSTM] file.

## Results and Analysis

The performance of the models was evaluated using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). For detailed analysis and comparisons, you can review the results section.

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- Optuna
- Matplotlib
- Seaborn

## Installation

To run the project on your computer, open this folder with VSCode or your preferred IDE.

## Contact

For questions or feedback about this project, you can email [buyukkamaci18@itu.edu.tr].

