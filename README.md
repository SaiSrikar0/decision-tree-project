# End-to-End Decision Tree Classifier Application

A comprehensive Streamlit application demonstrating the complete machine learning pipeline for Decision Tree classification, using the Iris dataset as an example.

## Overview

This project implements a full machine learning workflow including:
- **Data Ingestion**: Download or upload datasets
- **Exploratory Data Analysis (EDA)**: Visualize and understand your data
- **Data Cleaning**: Handle missing values with various strategies
- **Data Preprocessing**: Feature scaling and encoding
- **Model Training**: Decision Tree classifier with configurable hyperparameters
- **Model Evaluation**: Performance metrics and visualizations

## Features

- **Interactive UI**: Built with Streamlit for easy interaction
- **Flexible Data Input**: Download the Iris dataset or upload your own CSV file
- **Configurable Decision Tree Parameters**:
  - Criterion selection (gini, entropy, log_loss)
  - Splitter strategy (best, random)
  - Max depth control (1-20)
  - Min samples split and leaf configuration
- **Comprehensive Metrics**: Confusion matrix, classification report, and accuracy
- **Tree Visualization**: Visual representation of the trained decision tree
- **Automated Logging**: Timestamp-based logging for monitoring execution

## Project Structure

```
decision tree project/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    ├── raw/                  # Raw datasets
    │   └── iris.csv
    └── cleaned/              # Processed datasets
        └── cleaned_iris_*.csv
```

## Installation

1. **Clone or download the project**:
   ```bash
   cd "decision tree project"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Workflow Steps

1. **Step 1 - Data Ingestion**: 
   - Choose to download the Iris dataset or upload your own CSV file

2. **Step 2 - EDA**:
   - View dataset statistics (shape, missing values)
   - Visualize correlations with a heatmap

3. **Step 3 - Data Cleaning**:
   - Handle missing values using Mean, Median, or Drop Rows strategy

4. **Step 4 - Model Training**:
   - Configure Decision Tree hyperparameters via the sidebar
   - Train the model on your data

5. **Step 5 - Model Evaluation**:
   - View confusion matrix
   - Analyze classification metrics (precision, recall, F1-score)
   - Check overall accuracy
   - Visualize the decision tree structure

## Dependencies

- **streamlit**: Interactive web framework for building ML applications
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models, preprocessing, and metrics
- **matplotlib**: Plotting library for visualizations
- **seaborn**: Statistical data visualization
- **requests**: HTTP library for downloading datasets

## Configuration

### Decision Tree Hyperparameters (Sidebar)

- **Criterion**: Impurity measurement function (gini, entropy, log_loss)
- **Splitter**: Strategy for splitting nodes (best, random)
- **Max Depth**: Maximum depth of the tree (1-20)
- **Min Samples Split**: Minimum samples required to split a node (2-20)
- **Min Samples Leaf**: Minimum samples required at leaf node (1-20)

## Output

The application generates:
- Cleaned datasets saved in `data/cleaned/` with timestamps
- Visualizations (correlation heatmap, confusion matrix, decision tree)
- Classification metrics and model performance reports
- Timestamped console logs for debugging

## Notes

- The application uses session state to track data cleaning progress
- All operations are logged with timestamps for debugging and monitoring
- Data files are automatically organized in the `data/` directory structure
- The decision tree can be visualized directly in the application

## Author

AI/ML Data Science Project - Week 12

---

For questions or improvements, please refer to the logged messages in the console output.
