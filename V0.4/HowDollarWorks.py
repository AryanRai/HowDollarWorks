import os
import logging
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.vector_ar.var_model import VAR
import seaborn as sns
import warnings
from typing import Optional, Dict, Any, List
warnings.filterwarnings('ignore')

# For interactive graphs using Plotly
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional: import pmdarima for auto ARIMA tuning
try:
    from pmdarima import auto_arima
except ImportError:
    logging.error("pmdarima not installed. Run 'pip install pmdarima' to use ARIMA auto-tuning.")
    auto_arima = None

# Optional: import fredapi for FRED economic data
try:
    from fredapi import Fred
except ImportError:
    logging.error("fredapi not installed. Run 'pip install fredapi' to use FRED data.")
    Fred = None

class CurrencyAnalyzer:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 fred_api_key: Optional[str] = None, 
                 use_real_data: bool = False,
                 polynomial_degree: int = 3,
                 arima_order: Optional[tuple] = None,
                 var_lags: int = 2) -> None:
        """
        Initialize the Currency Analyzer.
        
        :param api_key: Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY env variable)
        :param fred_api_key: FRED API key (or set FRED_API_KEY env variable)
        :param use_real_data: Use real APIs if True; otherwise, generate dummy data.
        :param polynomial_degree: Degree for polynomial regression.
        :param arima_order: Optional ARIMA order; if None, auto_arima will be used.
        :param var_lags: Number of lags for the VAR model.
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self.use_real_data = use_real_data
        self.polynomial_degree = polynomial_degree
        self.arima_order = arima_order  # If None, use auto_arima
        self.var_lags = var_lags
        self.base_url = "https://www.alphavantage.co/query"
        self.historical_df: Optional[pd.DataFrame] = None
        self.aus_economic_data: Optional[pd.DataFrame] = None
        self.predictions: Optional[pd.DataFrame] = None
    
    def fetch_historical_data(self, from_symbol: str = "INR", to_symbol: str = "AUD", 
                             start_year: int = 2015, end_year: int = 2024) -> pd.DataFrame:
        """
        Fetch historical exchange rate data. If use_real_data is False, generate synthetic data.
        """
        if self.use_real_data and self.api_key:
            # Real API call logic would go here
            logging.info(f"Fetching real historical data from {from_symbol} to {to_symbol}...")
            # Actual implementation omitted for brevity
        else:
            # Generate synthetic data
            logging.info(f"Generating synthetic historical data from {from_symbol} to {to_symbol}...")
            start_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 12, 31)
            date_range = pd.date_range(start=start_date, end=end_date, freq='M')
            
            # Seed for reproducibility
            np.random.seed(42)
            
            # Base trend with seasonal component and some noise
            trend = np.linspace(0.018, 0.022, len(date_range))  # Slight upward trend
            seasonal = 0.002 * np.sin(np.arange(len(date_range)) * (2 * np.pi / 12))  # Annual cycle
            noise = np.random.normal(0, 0.0005, len(date_range))  # Random fluctuations
            
            # Generate rates for INR/AUD (how many AUD you get for 1 INR)
            inr_aud_rates = trend + seasonal + noise
            
            # Calculate inverse for AUD/INR (how many INR you need for 1 AUD)
            aud_inr_rates = 1 / inr_aud_rates
            
            # Create DataFrame
            df = pd.DataFrame({
                f"{from_symbol}/{to_symbol}": inr_aud_rates,
                f"{to_symbol}/{from_symbol}": aud_inr_rates
            }, index=date_range)
            
            self.historical_df = df
            return df
    
    def fetch_economic_indicators(self) -> pd.DataFrame:
        """
        Fetch economic indicators for Australia from FRED or generate synthetic data.
        """
        if self.use_real_data and self.fred_api_key and Fred is not None:
            # Real API call logic for FRED
            logging.info("Fetching real Australian economic indicators...")
            # Implementation omitted for brevity
        else:
            # Generate synthetic economic data
            logging.info("Generating synthetic Australian economic indicators...")
            if self.historical_df is None:
                raise ValueError("Historical data must be loaded first.")
                
            date_range = self.historical_df.index
            
            # Generate synthetic indicators
            np.random.seed(43)  # Different seed than historical data
            
            # GDP growth (quarterly, annualized)
            gdp_growth = 2.5 + 1.5 * np.sin(np.arange(len(date_range)) * (2 * np.pi / 16)) + np.random.normal(0, 0.5, len(date_range))
            
            # Inflation rate (monthly, annualized)
            inflation = 2.0 + 1.0 * np.sin(np.arange(len(date_range)) * (2 * np.pi / 24) + 2) + np.random.normal(0, 0.3, len(date_range))
            
            # Unemployment rate
            unemployment = 5.0 + 1.0 * np.sin(np.arange(len(date_range)) * (2 * np.pi / 20) + 4) + np.random.normal(0, 0.2, len(date_range))
            
            # Interest rate
            interest_rate = 3.0 + 1.5 * np.sin(np.arange(len(date_range)) * (2 * np.pi / 30) + 1) + np.random.normal(0, 0.15, len(date_range))
            
            # Create DataFrame
            eco_df = pd.DataFrame({
                'GDP_Growth': gdp_growth,
                'Inflation': inflation,
                'Unemployment': unemployment,
                'Interest_Rate': interest_rate
            }, index=date_range)
            
            self.aus_economic_data = eco_df
            return eco_df
    
    def decompose_seasonality(self) -> Any:
        """
        Decompose the time series into trend, seasonal, and residual components.
        """
        if self.historical_df is None:
            raise ValueError("Historical data not loaded.")
        
        # Using the INR/AUD column for decomposition
        result = seasonal_decompose(self.historical_df["INR/AUD"], model='additive', period=12)
        return result
    
    def predict_future_rates(self, prediction_months: int = 12) -> pd.DataFrame:
        """
        Predict future exchange rates using linear and polynomial regression models.
        Also adjusts for seasonality.
        """
        if self.historical_df is None:
            raise ValueError("Historical data not loaded.")
        
        df = self.historical_df.copy().reset_index()
        df['numeric_date'] = np.arange(len(df))
        X = df[['numeric_date']]
        y = df["INR/AUD"]
        
        # Linear regression
        lr_model = LinearRegression().fit(X, y)
        
        # Polynomial regression
        poly = PolynomialFeatures(degree=self.polynomial_degree)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression().fit(X_poly, y)
        
        # Generate future dates and numeric values
        last_date = df['index'].max()
        future_dates = [last_date + timedelta(days=30 * i) for i in range(1, prediction_months + 1)]
        future_numeric = np.arange(len(df), len(df) + prediction_months).reshape(-1, 1)
        
        # Make predictions
        lr_preds = lr_model.predict(future_numeric)
        poly_preds = poly_model.predict(poly.transform(future_numeric))
        
        # Add seasonal adjustments
        seasonal_decomp = self.decompose_seasonality()
        seasonal_pattern = seasonal_decomp.seasonal.values
        # Extract the most recent 12 months of seasonal pattern
        if len(seasonal_pattern) >= 12:
            seasonal_pattern = seasonal_pattern[-12:]
        
        # Apply seasonal pattern to future predictions
        seasonal_adjustments = np.array([seasonal_pattern[i % len(seasonal_pattern)] for i in range(len(df), len(df) + prediction_months)])
        seasonal_poly_preds = poly_preds + seasonal_adjustments
        
        # Create prediction DataFrame
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Linear_Prediction': lr_preds,
            'Polynomial_Prediction': poly_preds,
            'Seasonal_Polynomial_Prediction': seasonal_poly_preds
        })
        
        # Calculate inverse rates (AUD/INR)
        pred_df['Linear_AUD/INR'] = 1 / pred_df['Linear_Prediction']
        pred_df['Polynomial_AUD/INR'] = 1 / pred_df['Polynomial_Prediction']
        pred_df['Seasonal_Polynomial_AUD/INR'] = 1 / pred_df['Seasonal_Polynomial_Prediction']
        
        self.predictions = pred_df
        return pred_df

    def plan_money_transfer(self, total_inr: float, target_date: datetime, chunked: bool = True) -> Dict[str, Any]:
        """
        Plan the money transfer strategy given a total amount (in INR) to be transferred by a target date.
        
        The method compares two options:
          - Lump-sum: Transfer the entire amount on the best predicted month (highest AUD/INR rate) before the target date.
          - Chunked: Split the total amount over several recommended months with favorable rates.
        
        It also shows interactive graphs with predicted exchange rates and lists popular payment methods.
        
        :param total_inr: Total amount in INR to be transferred.
        :param target_date: Deadline by which funds must be transferred.
        :param chunked: If True, propose a split (chunked) transfer plan.
        :return: A dictionary with recommendations and interactive Plotly figures.
        """
        if self.predictions is None:
            logging.info("No predictions available. Generating predictions...")
            self.predict_future_rates()
            
        # Filter predictions that occur before or on the target date.
        pred_df = self.predictions[self.predictions['Date'] <= target_date]
        if pred_df.empty:
            raise ValueError("No predicted data available before the target date. Increase prediction horizon.")
        
        # For transfers, we want to maximize AUD/INR (i.e., get more AUD per INR)
        # Use the "Seasonal_Polynomial_AUD/INR" column.
        best_idx = pred_df['Seasonal_Polynomial_AUD/INR'].idxmax()
        lump_sum_plan = pred_df.loc[best_idx]
        lump_sum_aud = total_inr * lump_sum_plan['Seasonal_Polynomial_AUD/INR']
        
        plan = {"lump_sum": {
                    "transfer_date": lump_sum_plan['Date'],
                    "predicted_rate": lump_sum_plan['Seasonal_Polynomial_AUD/INR'],
                    "expected_aud": lump_sum_aud
                }
              }
        
        # If chunked transfers are desired, select several best months before the target date.
        if chunked:
            # For simplicity, select the top 3 months with highest predicted rates.
            best_months = pred_df.sort_values('Seasonal_Polynomial_AUD/INR', ascending=False).head(3)
            chunked_plan = []
            # Split total INR evenly among these months.
            inr_chunk = total_inr / len(best_months)
            total_aud = 0
            for idx, row in best_months.iterrows():
                aud_received = inr_chunk * row['Seasonal_Polynomial_AUD/INR']
                total_aud += aud_received
                chunked_plan.append({
                    "transfer_date": row['Date'],
                    "predicted_rate": row['Seasonal_Polynomial_AUD/INR'],
                    "inr_amount": inr_chunk,
                    "expected_aud": aud_received
                })
            plan["chunked"] = {
                "transfers": chunked_plan,
                "total_expected_aud": total_aud
            }
        
        # Interactive graph: Plot predicted AUD/INR rates until target_date.
        fig_rate = px.line(pred_df, x='Date', y='Seasonal_Polynomial_AUD/INR',
                           title="Predicted AUD/INR Rates until Target Date",
                           labels={"Seasonal_Polynomial_AUD/INR": "Predicted AUD/INR Rate"})
        fig_rate.add_scatter(x=[lump_sum_plan['Date']], y=[lump_sum_plan['Seasonal_Polynomial_AUD/INR']],
                             mode='markers', marker=dict(color='red', size=12),
                             name="Best Lump Sum Date")
        if chunked:
            for transfer in plan["chunked"]["transfers"]:
                fig_rate.add_scatter(x=[transfer["transfer_date"]], y=[transfer["predicted_rate"]],
                                     mode='markers', marker=dict(color='green', size=10),
                                     name="Chunk Transfer Date")
        
        # Payment methods (a static list for demonstration)
        payment_methods = [
            {"Method": "Wise", "Fee_%": 0.5, "Notes": "Low fees, near mid-market rate"},
            {"Method": "Remitly", "Fee_%": 1.0, "Notes": "Competitive for larger sums"},
            {"Method": "Western Union", "Fee_%": 2.0, "Notes": "Widespread, higher fees"},
            {"Method": "Bank Transfer", "Fee_%": 1.5, "Notes": "Convenient but may have hidden costs"}
        ]
        df_methods = pd.DataFrame(payment_methods)
        fig_methods = go.Figure(data=[go.Table(
            header=dict(values=list(df_methods.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[df_methods[k] for k in df_methods.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        fig_methods.update_layout(title="Popular Payment Methods & Fees")
        
        plan["fig_rate"] = fig_rate  # interactive rate forecast graph
        plan["fig_methods"] = fig_methods  # interactive payment methods table
        
        return plan
    
    def get_transfer_recommendations(self) -> Dict[str, Any]:
        """
        Generate transfer recommendations based on predicted rates.
        """
        # Example recommendation logic
        if self.predictions is None:
            self.predict_future_rates()
            
        # Find months with best rates
        best_month = self.predictions.loc[self.predictions['Seasonal_Polynomial_AUD/INR'].idxmax()]
        
        recommendations = {
            "best_month": best_month['Date'].strftime('%B %Y'),
            "predicted_rate": best_month['Seasonal_Polynomial_AUD/INR'],
            "general_advice": [
                "Consider splitting large transfers across multiple favorable months",
                "Compare fees across different transfer services",
                "Monitor economic indicators that might affect exchange rates"
            ]
        }
        
        return recommendations

    def run_complete_analysis(self, from_symbol: str = "INR", to_symbol: str = "AUD", 
                              start_year: int = 2015, end_year: int = 2024) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline including:
        - Fetching historical data
        - Fetching economic indicators
        - Generating predictions with multiple methods
        - Providing transfer recommendations
        """
        self.historical_df = self.fetch_historical_data(from_symbol, to_symbol, start_year, end_year)
        self.fetch_economic_indicators()
        self.predict_future_rates()
        # (Additional prediction methods omitted for brevity)
        recommendations = self.get_transfer_recommendations()
        return {
            "historical_data": self.historical_df,
            "economic_data": self.aus_economic_data,
            "predictions": self.predictions,
            "recommendations": recommendations
        }


# Example usage for planning money transfer
def main() -> None:
    analyzer = CurrencyAnalyzer(use_real_data=False)
    analyzer.run_complete_analysis()
    
    # Suppose your dad plans to transfer 1,000,000 INR by December 31, 2025 for college funds.
    total_inr = 1_000_000
    target_date = datetime(2025, 12, 31)
    
    plan = analyzer.plan_money_transfer(total_inr=total_inr, target_date=target_date, chunked=True)
    
    # Display recommendations:
    best_lump = plan["lump_sum"]
    logging.info("Lump Sum Transfer Recommendation:")
    logging.info(f" - Transfer Date: {best_lump['transfer_date'].strftime('%B %Y')}")
    logging.info(f" - Predicted Rate: {best_lump['predicted_rate']:.4f} AUD per INR")
    logging.info(f" - Expected AUD Received: {best_lump['expected_aud']:.2f} AUD")
    
    if "chunked" in plan:
        logging.info("Chunked Transfer Recommendation:")
        for transfer in plan["chunked"]["transfers"]:
            logging.info(f" - Transfer on {transfer['transfer_date'].strftime('%B %Y')}: "
                         f"{transfer['inr_amount']:.0f} INR at {transfer['predicted_rate']:.4f} AUD/INR "
                         f"= {transfer['expected_aud']:.2f} AUD")
        logging.info(f"Total Expected AUD Received: {plan['chunked']['total_expected_aud']:.2f} AUD")
    
    # Show interactive graphs in a browser (or in a Jupyter Notebook cell)
    plan["fig_rate"].show()
    plan["fig_methods"].show()

if __name__ == "__main__":
    main()