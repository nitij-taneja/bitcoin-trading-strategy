# Bitcoin Price Prediction and Trading Strategy

## Overview
This project is focused on analyzing Bitcoin price movements using machine learning and statistical methods. It involves exploratory data analysis, feature engineering, and trading strategy implementation based on technical indicators and time-series analysis.

## Project Structure
The repository contains the following files:

- **`code.ipynb`**: Jupyter Notebook containing all data preprocessing, analysis, and model implementation.
- **`risk_management_rules.pdf`**: Guidelines and rules for risk management in trading.
- **`backtesting_results.csv`**: Results from backtesting the trading strategy.
- **`development_process.pptx`**: PowerPoint presentation summarizing the project, methodology, and conclusions.
- **`project_summary.pdf`**: PDF document outlining the key details and findings of the project.

## Data
The dataset consists of Bitcoin price data collected from multiple timeframes. The files used for analysis include:

- `btc_1h.csv`
- `btc_2h.csv`
- `btc_3m.csv`
- `btc_4h.csv`
- `btc_5m.csv`
- `btc_6h.csv`
- `btc_15m.csv`
- `btc_30m.csv`

Each dataset contains historical price information with columns such as:
- `datetime`: Timestamp of the recorded data
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## Methodology
### 1. Data Preprocessing
- Combined multiple CSV files into a single dataset.
- Converted `datetime` column to a proper timestamp format.
- Filtered data between **2018-01-01** and **2022-01-31**.
- Resampled data to specified time intervals.
- Performed exploratory data analysis (EDA).

### 2. Exploratory Data Analysis (EDA)
- **Statistical Summary**: Calculated basic descriptive statistics.
- **Data Visualization**:
  - Time-series plots for `close`, `high`, `low`, and `open` prices.
  - Heatmap of correlation between features.
  - Distribution of daily returns.
  - Rolling mean and standard deviation for trend analysis.
  - Autocorrelation (ACF) and Partial Autocorrelation (PACF) plots.

### 3. Feature Engineering
- **Principal Component Analysis (PCA)**: Dimensionality reduction for key financial indicators.
- **Moving Averages**:
  - **Golden Cross**: Short-term moving average crossing above the long-term moving average (Buy Signal).
  - **Death Cross**: Short-term moving average crossing below the long-term moving average (Sell Signal).
- **Autocorrelation-based Features**:
  - Computed autocorrelation length of `close` price.
  - Added mean reversion signals based on autocorrelation patterns.

### 4. Machine Learning Models
Implemented various classification models for predicting market movements:
- **Random Forest Classifier**
- **Logistic Regression**
- **Support Vector Classifier (SVC)**
- **Gradient Boosting Classifier**
- **Hyperparameter Tuning** using `GridSearchCV`
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC Score

### 5. Trading Strategy & Backtesting
- Implemented a simple **momentum-based trading strategy**.
- Backtested the strategy to evaluate its effectiveness.
- Compared model predictions with actual market movements.

## Results & Conclusion
- Identified key technical indicators that influence Bitcoin price movements.
- Demonstrated the effectiveness of machine learning models in predicting short-term trends.
- Highlighted risk factors and performance metrics in **risk_management_rules.pdf**.
- Summarized key findings in **development_process.pptx**.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bitcoin-trading-strategy.git
   ```
2.  **Install required dependencies:**

    ```
    pip install -r requirements.txt
    ```

3.  **Open the Jupyter Notebook:**

    ```
    jupyter notebook code.ipynb
    ```

4.  Run the cells sequentially to reproduce the results.   

## Dependencies  
The project requires the following Python libraries:  

- `pandas`  
- `numpy`  
- `matplotlib`  
- `seaborn`  
- `scikit-learn`  
- `statsmodels`  

You can install them using:  

```bash 
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```
## Acknowledgments
This project was developed as part of a hackathon challenge. Special thanks to the mentors and organizers for their guidance.
   
