import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CurrencyAnalyzer:
    def __init__(self, api_key=None):
        """Initialize the Currency Analyzer with API key"""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.historical_df = None
        self.aus_economic_data = None
        self.predictions = None
        
    def fetch_historical_data(self, from_symbol="INR", to_symbol="AUD", start_year=2015, end_year=2024):
        """Fetch historical currency exchange rate data"""
        print(f"Fetching historical {from_symbol} to {to_symbol} exchange rate data...")
        
        if self.api_key is None:
            return self._generate_dummy_data(start_year, end_year)
        
        try:
            params = {
                "function": "FX_MONTHLY",
                "from_symbol": from_symbol,
                "to_symbol": to_symbol,
                "apikey": self.api_key,
                "datatype": "json"
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series FX (Monthly)" in data:
                df = pd.DataFrame.from_dict(data["Time Series FX (Monthly)"], orient="index")
                df = df[["4. close"]].rename(columns={"4. close": f"{from_symbol}/{to_symbol}"})
                df.index = pd.to_datetime(df.index)
                df[f"{from_symbol}/{to_symbol}"] = df[f"{from_symbol}/{to_symbol}"].astype(float)
                # For INR to AUD, we need the reciprocal (as more INR per AUD is worse for sending money)
                if from_symbol == "INR" and to_symbol == "AUD":
                    df[f"AUD/{from_symbol}"] = 1 / df[f"{from_symbol}/{to_symbol}"]
                return df[(df.index.year >= start_year) & (df.index.year <= end_year)]
            else:
                print(f"Error fetching data: {data}")
                return self._generate_dummy_data(start_year, end_year)
        except Exception as e:
            print(f"Exception occurred: {e}")
            return self._generate_dummy_data(start_year, end_year)
    
    def _generate_dummy_data(self, start_year=2015, end_year=2024):
        """Generate dummy data for demonstration when API fails"""
        print("Using synthetic data for demonstration purposes.")
        dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="M")
        
        # Create realistic INR/AUD data with seasonal patterns
        trend = np.linspace(0.018, 0.0205, len(dates))
        seasonal = 0.001 * np.sin(np.pi * 2 * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 0.0003, len(dates))
        exchange_rates = trend + seasonal + noise
        
        df = pd.DataFrame({
            "INR/AUD": exchange_rates,
            "AUD/INR": 1 / exchange_rates
        }, index=dates)
        
        return df
    
    def fetch_economic_indicators(self):
        """
        Fetch Australian economic indicators (dummy data) including additional indicators:
        - Mining & Education exports (existing)
        - GDP Growth, Inflation, Interest Rates, Unemployment, Trade Balance,
          Commodity Price Index, and Global Economic Sentiment.
        """
        dates = self.historical_df.index
        n = len(dates)
        
        # Existing indicators:
        # Mining exports - seasonal trends and growth
        mining_trend = np.linspace(20, 30, n)  # Billions AUD
        mining_seasonal = 2 * np.sin(np.pi * 2 * np.arange(n) / 12)
        mining_noise = np.random.normal(0, 0.5, n)
        mining_exports = mining_trend + mining_seasonal + mining_noise
        
        # Education exports - affected by academic calendar
        edu_base = np.linspace(3, 5, n)  # Billions AUD
        month_indices = np.array([d.month for d in dates])
        edu_seasonal = np.zeros(n)
        edu_seasonal[month_indices == 2] += 0.8
        edu_seasonal[month_indices == 3] += 0.6
        edu_seasonal[month_indices == 7] += 0.7
        edu_seasonal[month_indices == 8] += 0.5
        edu_noise = np.random.normal(0, 0.2, n)
        edu_exports = edu_base + edu_seasonal + edu_noise
        
        # Additional indicators:
        # GDP Growth Rate (%)
        gdp_growth = np.linspace(2.0, 2.5, n)  # percent
        gdp_growth_noise = np.random.normal(0, 0.2, n)
        gdp_growth = gdp_growth + gdp_growth_noise
        
        # Inflation Rate (CPI % change)
        inflation = np.linspace(1.5, 2.5, n)
        inflation_noise = np.random.normal(0, 0.1, n)
        inflation = inflation + inflation_noise
        
        # Interest Rates (RBA rate %)
        interest_rate = np.linspace(1.5, 2.5, n)
        interest_rate_noise = np.random.normal(0, 0.1, n)
        interest_rate = interest_rate + interest_rate_noise
        
        # Unemployment Rate (%)
        unemployment = np.linspace(4.0, 5.5, n)
        unemployment_noise = np.random.normal(0, 0.2, n)
        unemployment = unemployment + unemployment_noise
        
        # Trade Balance (Billions AUD, surplus or deficit)
        trade_balance = np.linspace(-2, 2, n)
        trade_balance_noise = np.random.normal(0, 0.5, n)
        trade_balance = trade_balance + trade_balance_noise
        
        # Commodity Price Index (dummy index, baseline around 100)
        commodity_index = np.linspace(100, 110, n)
        commodity_noise = np.random.normal(0, 2, n)
        commodity_index = commodity_index + commodity_noise
        
        # Global Economic Sentiment (index, higher means more positive)
        sentiment = np.linspace(45, 55, n)
        sentiment_noise = np.random.normal(0, 1, n)
        sentiment = sentiment + sentiment_noise
        
        # Create dataframe with all indicators
        eco_df = pd.DataFrame({
            "Mining_Exports_Billion_AUD": mining_exports,
            "Education_Exports_Billion_AUD": edu_exports,
            "GDP_Growth_Percent": gdp_growth,
            "Inflation_Rate_Percent": inflation,
            "Interest_Rate_Percent": interest_rate,
            "Unemployment_Rate_Percent": unemployment,
            "Trade_Balance_Billion_AUD": trade_balance,
            "Commodity_Price_Index": commodity_index,
            "Global_Economic_Sentiment": sentiment
        }, index=dates)
        
        self.aus_economic_data = eco_df
        return eco_df
    
    def analyze_correlation(self):
        """Analyze correlation between exchange rates and economic indicators"""
        if self.historical_df is None or self.aus_economic_data is None:
            print("Please load data first")
            return None
            
        analysis_df = pd.concat([self.historical_df, self.aus_economic_data], axis=1)
        corr_matrix = analysis_df.corr()
        return corr_matrix
    
    def decompose_seasonality(self, column="INR/AUD", period=12):
        """Decompose time series to identify seasonal patterns"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        decomposition = seasonal_decompose(self.historical_df[column], model='additive', period=period)
        return decomposition
    
    def predict_future_rates(self, prediction_months=12, method="polynomial"):
        """Predict future exchange rates using linear and polynomial regression models"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        df = self.historical_df.copy()
        df.reset_index(inplace=True)
        df['numeric_date'] = np.arange(len(df))
        
        X = df[['numeric_date']]
        y = df["INR/AUD"]
        
        predictions = {}
        # Linear regression
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # Polynomial regression
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        
        # Generate future dates
        last_date = df['index'].max()
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, prediction_months+1)]
        future_numeric_dates = np.arange(len(df), len(df) + prediction_months)
        
        lr_predictions = lr_model.predict(future_numeric_dates.reshape(-1, 1))
        poly_predictions = poly_model.predict(poly.transform(future_numeric_dates.reshape(-1, 1)))
        
        # Add seasonality adjustments from historical patterns
        seasonal_patterns = self.decompose_seasonality().seasonal.values
        seasonal_indices = [(i % 12) for i in range(len(df), len(df) + prediction_months)]
        seasonal_adjustments = np.array([seasonal_patterns[i] for i in seasonal_indices])
        seasonal_poly_predictions = poly_predictions + seasonal_adjustments
        
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Linear_Prediction': lr_predictions,
            'Polynomial_Prediction': poly_predictions,
            'Seasonal_Polynomial_Prediction': seasonal_poly_predictions
        })
        
        # Calculate AUD/INR (for money transfers)
        pred_df['Linear_AUD/INR'] = 1 / pred_df['Linear_Prediction']
        pred_df['Polynomial_AUD/INR'] = 1 / pred_df['Polynomial_Prediction']
        pred_df['Seasonal_Polynomial_AUD/INR'] = 1 / pred_df['Seasonal_Polynomial_Prediction']
        
        self.predictions = pred_df
        return pred_df

    def predict_future_rates_arima(self, prediction_months=12):
        """Predict future exchange rates using an ARIMA model"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        
        # Fit an ARIMA model (order may be adjusted for better fit)
        model = ARIMA(self.historical_df["INR/AUD"], order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=prediction_months)
        
        last_date = self.historical_df.index.max()
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, prediction_months+1)]
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "ARIMA_Prediction": forecast
        })
        return forecast_df

    def predict_with_economic_indicators(self, prediction_months=12):
        """
        Predict future exchange rates using a VAR model that incorporates both
        exchange rates and additional economic indicators.
        """
        if self.historical_df is None or self.aus_economic_data is None:
            print("Please load all data first")
            return None
        
        # Combine exchange rate and economic data
        df = pd.concat([self.historical_df["INR/AUD"], self.aus_economic_data], axis=1).dropna()
        model = VAR(df)
        results = model.fit(maxlags=2, ic='aic')
        forecast = results.forecast(df.values[-results.k_ar:], steps=prediction_months)
        
        future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, prediction_months+1)]
        forecast_df = pd.DataFrame(forecast, columns=df.columns, index=future_dates)
        return forecast_df
    
    def identify_optimal_transfer_months(self):
        """Identify optimal months for transferring money based on historical data"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        df = self.historical_df.copy()
        df['Month'] = df.index.month
        monthly_avg = df.groupby('Month').mean()
        monthly_avg['Month_Name'] = [datetime(2000, m, 1).strftime('%B') for m in monthly_avg.index]
        return monthly_avg.sort_values('AUD/INR', ascending=False)
    
    def plot_historical_rates(self, show_trend=True):
        """Plot historical exchange rates with an optional trend line"""
        if self.historical_df is None:
            print("Please load historical data first")
            return
        plt.figure(figsize=(14, 7))
        plt.plot(self.historical_df.index, self.historical_df["INR/AUD"], label="Historical INR/AUD", color="blue")
        if show_trend:
            z = np.polyfit(np.arange(len(self.historical_df)), self.historical_df["INR/AUD"], 1)
            p = np.poly1d(z)
            plt.plot(self.historical_df.index, p(np.arange(len(self.historical_df))), "r--", label="Trend")
        plt.title("INR to AUD Exchange Rate Historical Data")
        plt.xlabel("Date")
        plt.ylabel("INR/AUD Exchange Rate")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_predictions(self):
        """Plot historical data and predictions"""
        if self.historical_df is None or self.predictions is None:
            print("Please load data and generate predictions first")
            return
        plt.figure(figsize=(14, 7))
        plt.plot(self.historical_df.index, self.historical_df["INR/AUD"], label="Historical INR/AUD", color="blue")
        plt.plot(self.predictions['Date'], self.predictions['Seasonal_Polynomial_Prediction'], 
                 label="Predicted INR/AUD (with seasonality)", linestyle="--", color="green")
        optimal_months = self.identify_optimal_transfer_months()
        top_months = optimal_months.head(3).index.tolist()
        for i, row in self.predictions.iterrows():
            if row['Date'].month in top_months:
                plt.scatter(row['Date'], row['Seasonal_Polynomial_Prediction'], color='red', s=100, zorder=5)
                plt.annotate("Good time to transfer", (row['Date'], row['Seasonal_Polynomial_Prediction']),
                             xytext=(10, -30), textcoords="offset points",
                             arrowprops=dict(arrowstyle="->", color='red'))
        plt.title("INR to AUD Exchange Rate: Historical and Predicted with Optimal Transfer Times")
        plt.xlabel("Date")
        plt.ylabel("INR/AUD Exchange Rate (Lower is better for transfers from India)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_monthly_pattern(self):
        """Plot monthly patterns to identify seasonal trends"""
        if self.historical_df is None:
            print("Please load historical data first")
            return
        df = self.historical_df.copy()
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        monthly_avg = df.groupby('Month')['AUD/INR'].mean()
        months = [datetime(2000, m, 1).strftime('%b') for m in monthly_avg.index]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(months, monthly_avg.values, color='skyblue')
        threshold = monthly_avg.quantile(0.75)
        for i, v in enumerate(monthly_avg.values):
            if v >= threshold:
                bars[i].set_color('green')
                plt.text(i, v + 0.001, 'Good', ha='center', va='bottom', fontweight='bold', color='green')
            elif v <= monthly_avg.quantile(0.25):
                bars[i].set_color('red')
                plt.text(i, v + 0.001, 'Avoid', ha='center', va='bottom', fontweight='bold', color='red')
        plt.title("Average AUD/INR Rate by Month (Higher is Better for Transfers from India)")
        plt.xlabel("Month")
        plt.ylabel("Average AUD/INR")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_correlation_with_economic_factors(self):
        """Plot correlation between exchange rates and economic factors"""
        if self.historical_df is None or self.aus_economic_data is None:
            print("Please load all data first")
            return
        analysis_df = pd.concat([self.historical_df, self.aus_economic_data], axis=1)
        corr_matrix = analysis_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation between Exchange Rates and Economic Factors")
        plt.tight_layout()
        return plt.gcf()
    
    def get_transfer_recommendations(self):
        """Generate recommendations for money transfers based on historical and predicted data"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        optimal_months = self.identify_optimal_transfer_months()
        if self.predictions is not None:
            predicted_best = self.predictions.sort_values('Seasonal_Polynomial_AUD/INR', ascending=False)
            predicted_worst = self.predictions.sort_values('Seasonal_Polynomial_AUD/INR', ascending=True)
        else:
            predicted_best = None
            predicted_worst = None
        recommendations = {
            "historical_best_months": optimal_months.head(3)['Month_Name'].tolist(),
            "historical_worst_months": optimal_months.tail(3)['Month_Name'].tolist(),
        }
        if predicted_best is not None:
            recommendations["predicted_best_date"] = predicted_best.iloc[0]['Date'].strftime('%B %Y')
            recommendations["predicted_best_rate"] = predicted_best.iloc[0]['Seasonal_Polynomial_AUD/INR']
            recommendations["predicted_worst_date"] = predicted_worst.iloc[0]['Date'].strftime('%B %Y')
            recommendations["predicted_worst_rate"] = predicted_worst.iloc[0]['Seasonal_Polynomial_AUD/INR']
            best_rate = predicted_best.iloc[0]['Seasonal_Polynomial_AUD/INR']
            worst_rate = predicted_worst.iloc[0]['Seasonal_Polynomial_AUD/INR']
            savings_percent = ((best_rate - worst_rate) / worst_rate) * 100
            recommendations["potential_savings_percent"] = savings_percent
        return recommendations

    def run_complete_analysis(self, from_symbol="INR", to_symbol="AUD", start_year=2015, end_year=2024):
        """Run the complete analysis pipeline including fetching data, economic indicators, predictions, and recommendations"""
        # 1. Fetch historical data
        self.historical_df = self.fetch_historical_data(from_symbol, to_symbol, start_year, end_year)
        # 2. Fetch economic indicators (including additional factors)
        self.fetch_economic_indicators()
        # 3. Generate predictions using polynomial model
        self.predict_future_rates()
        # (Optional) Generate ARIMA-based forecast
        arima_forecast = self.predict_future_rates_arima()
        # (Optional) Generate VAR forecast using economic indicators
        var_forecast = self.predict_with_economic_indicators()
        # 4. Generate transfer recommendations
        recommendations = self.get_transfer_recommendations()
        
        return {
            "historical_data": self.historical_df,
            "economic_data": self.aus_economic_data,
            "predictions": self.predictions,
            "arima_forecast": arima_forecast,
            "var_forecast": var_forecast,
            "recommendations": recommendations
        }

# Example usage
def main():
    # Create analyzer instance with your API key (or use None to generate dummy data)
    analyzer = CurrencyAnalyzer(api_key="2ONTZBSMSIYA85NQ")
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Visualizations
    print("\n===== VISUALIZING EXCHANGE RATE PATTERNS =====")
    analyzer.plot_historical_rates()
    plt.savefig("historical_rates.png")
    plt.close()
    
    analyzer.plot_predictions()
    plt.savefig("predictions.png")
    plt.close()
    
    analyzer.plot_monthly_pattern()
    plt.savefig("monthly_patterns.png")
    plt.close()
    
    analyzer.plot_correlation_with_economic_factors()
    plt.savefig("correlations.png")
    plt.close()
    
    # Print recommendations
    print("\n===== MONEY TRANSFER RECOMMENDATIONS =====")
    rec = results["recommendations"]
    print(f"Based on historical data (2015-2024):")
    print(f"- Best months to transfer money from India to Australia: {', '.join(rec['historical_best_months'])}")
    print(f"- Months to avoid for transfers: {', '.join(rec['historical_worst_months'])}")
    print(f"\nPredictions for upcoming transfers:")
    print(f"- Best time to transfer in the next 12 months: {rec['predicted_best_date']} (Rate: {rec['predicted_best_rate']:.4f} AUD per INR)")
    print(f"- Worst time to transfer: {rec['predicted_worst_date']} (Rate: {rec['predicted_worst_rate']:.4f} AUD per INR)")
    print(f"- Potential savings by timing your transfer optimally: {rec['potential_savings_percent']:.2f}%")
    
    print("\nFor a family transferring 1,000,000 INR for college funds:")
    best_aud = 1000000 * rec['predicted_best_rate']
    worst_aud = 1000000 * rec['predicted_worst_rate']
    diff_aud = best_aud - worst_aud
    print(f"- Transferring at the best time: {best_aud:.2f} AUD")
    print(f"- Transferring at the worst time: {worst_aud:.2f} AUD")
    print(f"- Difference: {diff_aud:.2f} AUD saved by optimal timing")
    
    print("\n===== RELATIONSHIP WITH AUSTRALIAN ECONOMY =====")
    corr = analyzer.analyze_correlation()
    mining_corr = corr.loc["INR/AUD", "Mining_Exports_Billion_AUD"]
    edu_corr = corr.loc["INR/AUD", "Education_Exports_Billion_AUD"]
    
    print(f"Correlation between exchange rate and mining exports: {mining_corr:.2f}")
    print(f"Correlation between exchange rate and education sector: {edu_corr:.2f}")
    
    if abs(mining_corr) > abs(edu_corr):
        print("\nThe mining sector appears to have a stronger relationship with the exchange rate.")
        if mining_corr > 0:
            print("As mining exports increase, the INR/AUD rate tends to increase (less favorable for transfers).")
        else:
            print("As mining exports increase, the INR/AUD rate tends to decrease (more favorable for transfers).")
    else:
        print("\nThe education sector appears to have a stronger relationship with the exchange rate.")
        if edu_corr > 0:
            print("As education exports increase, the INR/AUD rate tends to increase (less favorable for transfers).")
        else:
            print("As education exports increase, the INR/AUD rate tends to decrease (more favorable for transfers).")
    
    print("\nRECOMMENDATION SUMMARY:")
    print("1. Consider scheduling major transfers during the historically favorable months.")
    print("2. For your college fund transfers, aim for transfers during the predicted optimal periods.")
    print("3. For large sums, consider splitting the transfer into multiple transactions spread across favorable months.")
    print("4. Monitor Australian economic news and key indicators (GDP, inflation, interest rates, etc.) as they may signal exchange rate changes.")

if __name__ == "__main__":
    API_KEY = "2ONTZBSMSIYA85NQ"
    if API_KEY == "API_KEY":
        print("Please replace 'YOUR_API_KEY_HERE' with a valid Alpha Vantage API key.")
        main()  # Runs with dummy data
    else:
        main()
