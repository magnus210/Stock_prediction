# Stock Price Prediction with LSTM & ARIMA

> A learning project exploring time series forecasting using deep learning (LSTM) and statistical methods (ARIMA) for stock price prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow)

## About This Project

This is a **personal learning project** built out of curiosity to understand:
- How **LSTM (Long Short-Term Memory)** networks work for sequence prediction
- How **ARIMA (AutoRegressive Integrated Moving Average)** models handle time series data
- The differences between deep learning and statistical approaches to forecasting

> **Note**: This project is still under active development. I'm continuously experimenting and learning locally.

## What I'm Learning

### LSTM (Long Short-Term Memory)
- Recurrent Neural Networks for sequential data
- How memory cells retain information over long sequences
- Sequence-to-sequence prediction for time series
- Data preprocessing with sliding windows (sequence_length = 60)

### ARIMA (AutoRegressive Integrated Moving Average)
- Statistical forecasting methods
- Stationarity and differencing
- Parameter selection (p, d, q)
- Short-term vs long-term predictions

### Technical Indicators
- **SMA** (Simple Moving Average): 20-day and 50-day
- **EMA** (Exponential Moving Average): 12-day and 26-day
- Understanding how traders use these indicators

## Current Implementation

| Component | Status | Description |
|-----------|--------|-------------|
| Data Collection | ✅ Done | Yahoo Finance API (yfinance) |
| Data Preprocessing | ✅ Done | MinMaxScaler, sequence creation |
| Technical Indicators | ✅ Done | SMA_20, SMA_50, EMA_12, EMA_26 |
| ARIMA Model | ✅ Done | Next-day prediction |
| LSTM Model | ✅ Done | 20-day forecast |
| Visualization | ✅ Done | Historical + predictions plot |
| Model Evaluation | 🚧 WIP | RMSE, MAE metrics |
| Hyperparameter Tuning | 🚧 WIP | Grid search for optimal params |

## Tech Stack

| Library | Purpose |
|---------|----------|
| **yfinance** | Fetch historical stock data |
| **pandas** | Data manipulation |
| **numpy** | Numerical operations |
| **TensorFlow/Keras** | LSTM neural network |
| **statsmodels** | ARIMA implementation |
| **scikit-learn** | Data scaling (MinMaxScaler) |
| **matplotlib** | Visualizations |

## Project Structure

```
Stock_prediction/
├── Stock_Prediction.ipynb   # Main notebook with all experiments
└── README.md                # Project documentation
```

## Current Results

### Data Used
- **Stock**: Apple (AAPL)
- **Period**: 1999-01-01 to 2024-11-13
- **Features**: Close price + technical indicators

### Model Outputs
- **ARIMA**: Next-day prediction ~$224.35
- **LSTM**: 20-day forward predictions
- **Training**: 10 epochs, sequence length of 60 days

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/magnus210/Stock_prediction.git
   cd Stock_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install yfinance pandas numpy tensorflow statsmodels scikit-learn matplotlib
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook Stock_Prediction.ipynb
   ```

4. **Run all cells** to see predictions

## Key Learnings So Far

### What I've Discovered

1. **LSTM vs ARIMA Trade-offs**
   - LSTM: Better at capturing complex patterns, needs more data
   - ARIMA: Faster, more interpretable, good for short-term

2. **Data Preprocessing is Crucial**
   - Scaling data to [0,1] significantly improves LSTM performance
   - Sequence length affects what patterns the model can learn

3. **Stock Prediction Limitations**
   - Past performance doesn't guarantee future results
   - External factors (news, earnings) aren't captured
   - This is for learning, not financial advice!

## Planned Improvements

- [ ] Add model evaluation metrics (RMSE, MAE, MAPE)
- [ ] Implement walk-forward validation
- [ ] Experiment with different LSTM architectures (stacked, bidirectional)
- [ ] Add more features (volume, sentiment analysis)
- [ ] Try other models (GRU, Transformer)
- [ ] Create interactive predictions with Streamlit
- [ ] Compare multiple stocks

## Disclaimer

⚠️ **This project is for educational purposes only.**

Stock price prediction is inherently uncertain. This project is meant to learn about time series forecasting and deep learning concepts, NOT to provide financial advice. Never make investment decisions based on model predictions alone.

## Resources I'm Using

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [ARIMA Model Guide](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Time Series Forecasting with TensorFlow](https://www.tensorflow.org/tutorials/structured_data/time_series)

---

*This is a learning journey. Feedback and suggestions are welcome!*
