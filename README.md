# Crime Prediction and Analysis Project

## Overview

This project performs comprehensive analysis and prediction of crime patterns across different states and union territories in India using machine learning techniques. The analysis covers crime data from 2001 to 2012, providing insights into crime trends, state-wise comparisons, and predictive modeling for crime classification.

## Dataset

The project uses the "District-wise Crime Data" dataset containing:
- **35 States/Union Territories** across India
- **12 years** of data (2001-2012)
- **420 total records** (35 states × 12 years)
- **32 features** including various crime categories

### Crime Categories Analyzed

1. **Violent Crimes**: Murder, Attempt to Murder, Culpable Homicide, Rape
2. **Property Crimes**: Robbery, Burglary, Theft, Auto Theft
3. **Organized Crime**: Dacoity, Kidnapping & Abduction
4. **Social Crimes**: Riots, Arson, Hurt/Grievous Hurt
5. **Economic Crimes**: Criminal Breach of Trust, Cheating, Counterfeiting
6. **Crimes Against Women**: Dowry Deaths, Assault on Women, Insult to Modesty
7. **Other IPC Crimes**: Various other criminal offenses

## Features

### 1. Data Preprocessing
- **Missing Value Handling**: Imputation using mean strategy
- **Categorical Encoding**: One-hot encoding for state/UT names
- **Feature Scaling**: StandardScaler for numerical features
- **Data Transformation**: Conversion to appropriate formats for ML models

### 2. Exploratory Data Analysis (EDA)

#### State-wise Crime Analysis
- Total crime comparison across all states/UTs
- Identification of high-crime and low-crime regions
- Visual representation using bar charts

#### Crime Type Analysis
- Distribution of different crime categories
- Identification of most prevalent crime types
- Comparative analysis of crime rates

#### Temporal Analysis
- Year-wise crime trends for each state
- Time series visualization of crime patterns
- Identification of increasing/decreasing crime trends

#### Geographic Distribution
- Pie chart visualization of crime distribution by state
- Percentage contribution of each state to total crimes
- Regional crime pattern analysis

### 3. Clustering Analysis

#### K-Means Clustering
- **Elbow Method**: Optimal cluster determination (2 clusters identified)
- **State Classification**: Safe vs Unsafe state categorization
- **Visual Representation**: Bar charts showing cluster distribution

### 4. Machine Learning Models

The project implements and compares multiple classification models:

#### 1. Artificial Neural Network (ANN)
- **Architecture**: Sequential model with dense layers
- **Layers**: Input → Dense(6, ReLU) → Dense(1, Sigmoid)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Training**: 100 epochs, batch size 32

#### 2. Support Vector Machine (SVM)
- **Kernel SVM**: RBF kernel implementation
- **Linear SVM**: Linear kernel for comparison
- **Performance**: High accuracy with good generalization

#### 3. K-Nearest Neighbors (K-NN)
- **Parameters**: 5 neighbors, Minkowski distance (p=2)
- **Performance**: Effective for crime pattern classification

#### 4. Logistic Regression
- **Implementation**: Standard logistic regression
- **Performance**: Good baseline model performance

#### 5. Random Forest
- **Parameters**: 10 estimators
- **Performance**: Robust ensemble method

#### 6. Decision Tree
- **Implementation**: Standard decision tree classifier
- **Performance**: Interpretable model with good accuracy

#### 7. Support Vector Regression (SVR)
- **Kernel**: RBF kernel
- **Application**: Regression-based crime prediction

### 5. Model Evaluation

#### Performance Metrics
- **Confusion Matrix**: True vs Predicted classifications
- **Accuracy Score**: Overall model performance
- **Comparative Analysis**: Model performance comparison

#### Results Summary
- **Best Performing Models**: SVM (Linear), Logistic Regression, Random Forest
- **Accuracy**: High accuracy across multiple models
- **Consistency**: Consistent performance across different algorithms

## Key Findings

### 1. Crime Distribution
- **High Crime States**: Maharashtra, Madhya Pradesh, Tamil Nadu, Uttar Pradesh
- **Low Crime States**: Lakshadweep, Daman & Diu, Sikkim, A & N Islands
- **Regional Patterns**: Clear geographic clustering of crime rates

### 2. Crime Trends
- **Temporal Patterns**: Varying trends across different states
- **Year-wise Analysis**: Some states showing increasing crime rates
- **Stability**: Certain states maintaining consistent crime levels

### 3. Model Insights
- **Classification Accuracy**: High accuracy in distinguishing safe/unsafe states
- **Feature Importance**: Crime categories effectively used for classification
- **Predictive Power**: Models successfully predict crime risk categories

## Technical Implementation

### Libraries Used
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn, TensorFlow
- **Visualization**: Matplotlib, Seaborn
- **Statistical Analysis**: Various statistical methods

### Data Pipeline
1. **Data Loading**: CSV file import and validation
2. **Preprocessing**: Cleaning, encoding, scaling
3. **Feature Engineering**: Transformation and selection
4. **Model Training**: Multiple algorithm implementation
5. **Evaluation**: Performance assessment and comparison
6. **Visualization**: Results presentation and analysis

## Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### Running the Analysis
1. Ensure the dataset file is in the project directory
2. Open the Jupyter notebook: `CrimePredictionAndAnalysis.ipynb`
3. Run cells sequentially for complete analysis
4. View generated visualizations and model results

### Output Files
- **Visualizations**: Various charts and graphs
- **Model Results**: Performance metrics and predictions
- **Processed Data**: Transformed datasets for analysis


### Law Enforcement
- **Resource Allocation**: Identify high-crime areas for increased policing
- **Prevention Strategies**: Target specific crime types based on patterns
- **Policy Making**: Data-driven approach to crime prevention

### Research and Analysis
- **Academic Research**: Criminology and social science studies
- **Policy Research**: Government and NGO analysis
- **Comparative Studies**: Cross-regional crime pattern analysis

### Predictive Policing
- **Risk Assessment**: Identify areas at risk of increased crime
- **Trend Prediction**: Forecast future crime patterns
- **Strategic Planning**: Long-term law enforcement planning



**Note**: This analysis is based on historical crime data and should be used responsibly. The predictions and classifications are statistical estimates and should not be the sole basis for policy decisions.
