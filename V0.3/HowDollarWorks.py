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
    def __init__(self, api_key=None, fred_api_key=None, use_real_data=False):
        """
        Initialize the Currency Analyzer.
        Parameters:
        - api_key: API key for Alpha Vantage (currency data)
        - fred_api_key: API key for FRED (economic data)
        - use_real_data: if True, fetch data from real APIs rather than using dummy/simulated data
        """
        self.api_key = api_key
        self.fred_api_key = fred_api_key
        self.use_real_data = use_real_data
        self.base_url = "https://www.alphavantage.co/query"
        self.historical_df = None
        self.aus_economic_data = None
        self.predictions = None

    def fetch_historical_data(self, from_symbol="INR", to_symbol="AUD", start_year=2015, end_year=2024):
        """Fetch historical currency exchange rate data from Alpha Vantage or use dummy data."""
        print(f"Fetching historical {from_symbol} to {to_symbol} exchange rate data...")
        
        if self.api_key is None or not self.use_real_data:
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
                if from_symbol == "INR" and to_symbol == "AUD":
                    df["AUD/INR"] = 1 / df[f"{from_symbol}/{to_symbol}"]
                return df[(df.index.year >= start_year) & (df.index.year <= end_year)]
            else:
                print(f"Error fetching data: {data}")
                return self._generate_dummy_data(start_year, end_year)
        except Exception as e:
            print(f"Exception occurred: {e}")
            return self._generate_dummy_data(start_year, end_year)
    
    def _generate_dummy_data(self, start_year=2015, end_year=2024):
        """Generate dummy historical exchange rate data when real data is unavailable."""
        print("Using synthetic data for demonstration purposes.")
        dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="M")
        trend = np.linspace(0.018, 0.0205, len(dates))
        seasonal = 0.001 * np.sin(np.pi * 2 * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 0.0003, len(dates))
        exchange_rates = trend + seasonal + noise
        df = pd.DataFrame({
            "INR/AUD": exchange_rates,
            "AUD/INR": 1 / exchange_rates
        }, index=dates)
        return df

    # === New Methods to Fetch Real Economic Data via Other APIs ===

    def fetch_world_bank_indicator(self, indicator, country_code="AUS", start_year=2015, end_year=2024):
        """
        Fetch indicator data from the World Bank API.
        Example indicators:
          - GDP: "NY.GDP.MKTP.CD"
          - Inflation: "FP.CPI.TOTL.ZG"
        """
        url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&date={start_year}:{end_year}"
        try:
            response = requests.get(url)
            data = response.json()
            if data and len(data) > 1:
                records = data[1]
                df = pd.DataFrame(records)
                df = df[['date', 'value']].dropna()
                df['date'] = pd.to_datetime(df['date'], format='%Y')
                df = df.set_index('date').sort_index()
                df.columns = [indicator]
                return df
            else:
                print(f"No data returned for indicator {indicator}.")
                return None
        except Exception as e:
            print(f"Error fetching World Bank data for {indicator}: {e}")
            return None

    def fetch_fred_data(self, series_id, start_date="2015-01-01", end_date="2024-12-31"):
        """
        Fetch economic data from the FRED API.
        Requires the 'fredapi' package (pip install fredapi).
        Example series_id: "UNRATE" for unemployment rate.
        """
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_api_key)
            data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            df = data.to_frame(name=series_id)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            print(f"Error fetching FRED data for {series_id}: {e}")
            return None

    def fetch_economic_indicators(self):
        """
        Fetch Australian economic indicators. If use_real_data is True, fetch from APIs,
        otherwise generate dummy data. Combines several indicators.
        """
        dates = self.historical_df.index
        n = len(dates)
        
        if self.use_real_data:
            # Example: Fetch Australia's GDP and Inflation from the World Bank.
            gdp_df = self.fetch_world_bank_indicator("NY.GDP.MKTP.CD", country_code="AUS")
            inflation_df = self.fetch_world_bank_indicator("FP.CPI.TOTL.ZG", country_code="AUS")
            # For other series, you could also use FRED.
            # unemployment_df = self.fetch_fred_data("UNRATE")
            # For demonstration, merge available indicators on dates.
            # Here we resample the World Bank annual data to monthly via forward-fill.
            if gdp_df is not None:
                gdp_df = gdp_df.resample('M').ffill().reindex(dates, method='ffill')
            else:
                gdp_df = pd.DataFrame(np.linspace(1, 2, n), index=dates, columns=["GDP"])
            if inflation_df is not None:
                inflation_df = inflation_df.resample('M').ffill().reindex(dates, method='ffill')
            else:
                inflation_df = pd.DataFrame(np.linspace(1.5, 2.5, n), index=dates, columns=["Inflation"])
            
            # You can similarly fetch other indicators or use dummy values.
            # For simplicity, we keep the dummy simulations for these:
            mining_exports = np.linspace(20, 30, n) + 2 * np.sin(np.pi * 2 * np.arange(n) / 12) + np.random.normal(0, 0.5, n)
            edu_exports = np.linspace(3, 5, n) + np.random.normal(0, 0.2, n)
            # Combine all indicators into one DataFrame.
            eco_df = pd.DataFrame({
                "Mining_Exports_Billion_AUD": mining_exports,
                "Education_Exports_Billion_AUD": edu_exports,
            }, index=dates)
            eco_df = pd.concat([eco_df, gdp_df, inflation_df], axis=1)
        else:
            # Dummy data (including additional simulated indicators)
            mining_trend = np.linspace(20, 30, n)
            mining_seasonal = 2 * np.sin(np.pi * 2 * np.arange(n) / 12)
            mining_noise = np.random.normal(0, 0.5, n)
            mining_exports = mining_trend + mining_seasonal + mining_noise
            
            edu_base = np.linspace(3, 5, n)
            month_indices = np.array([d.month for d in dates])
            edu_seasonal = np.zeros(n)
            edu_seasonal[month_indices == 2] += 0.8
            edu_seasonal[month_indices == 3] += 0.6
            edu_seasonal[month_indices == 7] += 0.7
            edu_seasonal[month_indices == 8] += 0.5
            edu_noise = np.random.normal(0, 0.2, n)
            edu_exports = edu_base + edu_seasonal + edu_noise
            
            # Additional simulated indicators:
            gdp_growth = np.linspace(2.0, 2.5, n) + np.random.normal(0, 0.2, n)
            inflation = np.linspace(1.5, 2.5, n) + np.random.normal(0, 0.1, n)
            
            eco_df = pd.DataFrame({
                "Mining_Exports_Billion_AUD": mining_exports,
                "Education_Exports_Billion_AUD": edu_exports,
                "GDP_Growth_Percent": gdp_growth,
                "Inflation_Rate_Percent": inflation,
            }, index=dates)
        
        self.aus_economic_data = eco_df
        return eco_df
    
    def analyze_correlation(self):
        """Analyze correlation between exchange rates and economic indicators."""
        if self.historical_df is None or self.aus_economic_data is None:
            print("Please load data first")
            return None
        analysis_df = pd.concat([self.historical_df, self.aus_economic_data], axis=1)
        corr_matrix = analysis_df.corr()
        return corr_matrix
    
    def decompose_seasonality(self, column="INR/AUD", period=12):
        """Decompose time series to identify seasonal patterns."""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        decomposition = seasonal_decompose(self.historical_df[column], model='additive', period=period)
        return decomposition
    
    def predict_future_rates(self, prediction_months=12, method="polynomial"):
        """Predict future exchange rates using linear and polynomial regression models."""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        df = self.historical_df.copy()
        df.reset_index(inplace=True)
        df['numeric_date'] = np.arange(len(df))
        X = df[['numeric_date']]
        y = df["INR/AUD"]
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
        # Seasonal adjustment using historical decomposition
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
        # Calculate AUD/INR for money transfers
        pred_df['Linear_AUD/INR'] = 1 / pred_df['Linear_Prediction']
        pred_df['Polynomial_AUD/INR'] = 1 / pred_df['Polynomial_Prediction']
        pred_df['Seasonal_Polynomial_AUD/INR'] = 1 / pred_df['Seasonal_Polynomial_Prediction']
        self.predictions = pred_df
        return pred_df

    def predict_future_rates_arima(self, prediction_months=12):
        """Predict future exchange rates using an ARIMA model."""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
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
        df = pd.concat([self.historical_df["INR/AUD"], self.aus_economic_data], axis=1).dropna()
        model = VAR(df)
        results = model.fit(maxlags=2, ic='aic')
        forecast = results.forecast(df.values[-results.k_ar:], steps=prediction_months)
        future_dates = [df.index[-1] + pd.DateOffset(months=i) for i in range(1, prediction_months+1)]
        forecast_df = pd.DataFrame(forecast, columns=df.columns, index=future_dates)
        return forecast_df
    
    def identify_optimal_transfer_months(self):
        """Identify optimal months for transferring money based on historical data."""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
        df = self.historical_df.copy()
        df['Month'] = df.index.month
        monthly_avg = df.groupby('Month').mean()
        monthly_avg['Month_Name'] = [datetime(2000, m, 1).strftime('%B') for m in monthly_avg.index]
        return monthly_avg.sort_values('AUD/INR', ascending=False)
    
    def plot_historical_rates(self, show_trend=True):
        """Plot historical exchange rates with an optional trend line."""
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
        """Plot historical data and predictions."""
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
        """Plot monthly patterns to identify seasonal trends."""
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
        """Plot correlation between exchange rates and economic factors."""
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
        """Generate recommendations for money transfers based on historical and predicted data."""
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
        """Run the complete analysis pipeline including data fetching, prediction, and recommendations."""
        self.historical_df = self.fetch_historical_data(from_symbol, to_symbol, start_year, end_year)
        self.fetch_economic_indicators()
        self.predict_future_rates()
        arima_forecast = self.predict_future_rates_arima()
        var_forecast = self.predict_with_economic_indicators()
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
    # Set use_real_data=True and provide valid API keys to fetch real economic data.
    analyzer = CurrencyAnalyzer(api_key="2ONTZBSMSIYA85NQ", fred_api_key="f8cd687d590a0c8523f2ab8f7d9eb61d", use_real_data=True)
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
    main()
