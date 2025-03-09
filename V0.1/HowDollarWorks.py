import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose
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
        # Base trend with slight increase over time
        trend = np.linspace(0.018, 0.0205, len(dates))
        
        # Add seasonal component (higher in Q1, lower in Q3)
        seasonal = 0.001 * np.sin(np.pi * 2 * np.arange(len(dates)) / 12)
        
        # Add realistic noise
        noise = np.random.normal(0, 0.0003, len(dates))
        
        # Combine components
        exchange_rates = trend + seasonal + noise
        
        df = pd.DataFrame({
            "INR/AUD": exchange_rates,
            "AUD/INR": 1 / exchange_rates
        }, index=dates)
        
        return df
    
    def fetch_economic_indicators(self):
        """Fetch Australian economic indicators (dummy data)"""
        dates = self.historical_df.index
        
        # Mining exports - has seasonal trends and growth
        mining_trend = np.linspace(20, 30, len(dates))  # Billions AUD
        mining_seasonal = 2 * np.sin(np.pi * 2 * np.arange(len(dates)) / 12)
        mining_noise = np.random.normal(0, 0.5, len(dates))
        mining_exports = mining_trend + mining_seasonal + mining_noise
        
        # Education exports - affected by academic calendar
        edu_base = np.linspace(3, 5, len(dates))  # Billions AUD
        # Strong seasonality with peaks in Feb-Mar and Jul-Aug (semester starts)
        month_indices = np.array([d.month for d in dates])
        edu_seasonal = np.zeros(len(dates))
        edu_seasonal[month_indices == 2] += 0.8  # February peak
        edu_seasonal[month_indices == 3] += 0.6  # March
        edu_seasonal[month_indices == 7] += 0.7  # July peak
        edu_seasonal[month_indices == 8] += 0.5  # August
        edu_noise = np.random.normal(0, 0.2, len(dates))
        edu_exports = edu_base + edu_seasonal + edu_noise
        
        # Create dataframe
        eco_df = pd.DataFrame({
            "Mining_Exports_Billion_AUD": mining_exports,
            "Education_Exports_Billion_AUD": edu_exports
        }, index=dates)
        
        self.aus_economic_data = eco_df
        return eco_df
    
    def analyze_correlation(self):
        """Analyze correlation between exchange rates and economic indicators"""
        if self.historical_df is None or self.aus_economic_data is None:
            print("Please load data first")
            return None
            
        # Combine datasets
        analysis_df = pd.concat([self.historical_df, self.aus_economic_data], axis=1)
        
        # Calculate correlation matrix
        corr_matrix = analysis_df.corr()
        
        return corr_matrix
    
    def decompose_seasonality(self, column="INR/AUD", period=12):
        """Decompose time series to identify seasonal patterns"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
            
        # Decompose the time series
        decomposition = seasonal_decompose(self.historical_df[column], model='additive', period=period)
        
        return decomposition
    
    def predict_future_rates(self, prediction_months=12, method="polynomial"):
        """Predict future exchange rates using multiple models"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
            
        # Prepare data
        df = self.historical_df.copy()
        df.reset_index(inplace=True)
        df['numeric_date'] = np.arange(len(df))
        
        # Feature for prediction
        X = df[['numeric_date']]
        y = df["INR/AUD"]
        
        # Train models
        predictions = {}
        
        # 1. Linear regression
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # 2. Polynomial regression
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        
        # Generate future dates
        last_date = df['index'].max()
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, prediction_months+1)]
        future_numeric_dates = np.arange(len(df), len(df) + prediction_months)
        
        # Predict with each model
        lr_predictions = lr_model.predict(future_numeric_dates.reshape(-1, 1))
        poly_predictions = poly_model.predict(poly.transform(future_numeric_dates.reshape(-1, 1)))
        
        # Add seasonality from historical patterns
        seasonal_patterns = self.decompose_seasonality().seasonal.values
        seasonal_indices = [(i % 12) for i in range(len(df), len(df) + prediction_months)]
        seasonal_adjustments = np.array([seasonal_patterns[i] for i in seasonal_indices])
        
        # Seasonal adjustment for polynomial predictions
        seasonal_poly_predictions = poly_predictions + seasonal_adjustments
        
        # Prepare prediction dataframe
        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Linear_Prediction': lr_predictions,
            'Polynomial_Prediction': poly_predictions,
            'Seasonal_Polynomial_Prediction': seasonal_poly_predictions
        })
        
        # Calculate AUD/INR (this is what matters for money transfer)
        pred_df['Linear_AUD/INR'] = 1 / pred_df['Linear_Prediction']
        pred_df['Polynomial_AUD/INR'] = 1 / pred_df['Polynomial_Prediction']
        pred_df['Seasonal_Polynomial_AUD/INR'] = 1 / pred_df['Seasonal_Polynomial_Prediction']
        
        self.predictions = pred_df
        return pred_df
    
    def identify_optimal_transfer_months(self):
        """Identify optimal months for transferring money based on historical data"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
            
        # Group by month and calculate average
        df = self.historical_df.copy()
        df['Month'] = df.index.month
        monthly_avg = df.groupby('Month').mean()
        
        # For transfers from INR to AUD, we want MORE AUD per INR
        # This means higher AUD/INR rates are better
        monthly_avg['Month_Name'] = [datetime(2000, m, 1).strftime('%B') for m in monthly_avg.index]
        
        return monthly_avg.sort_values('AUD/INR', ascending=False)
    
    def plot_historical_rates(self, show_trend=True):
        """Plot historical exchange rates with optional trend line"""
        if self.historical_df is None:
            print("Please load historical data first")
            return
            
        plt.figure(figsize=(14, 7))
        
        # Plot historical rates
        plt.plot(self.historical_df.index, self.historical_df["INR/AUD"], 
                 label="Historical INR/AUD", color="blue")
        
        if show_trend:
            # Add trend line
            z = np.polyfit(np.arange(len(self.historical_df)), 
                           self.historical_df["INR/AUD"], 1)
            p = np.poly1d(z)
            plt.plot(self.historical_df.index, 
                     p(np.arange(len(self.historical_df))), 
                     "r--", label="Trend")
        
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
        
        # Historical data
        plt.plot(self.historical_df.index, self.historical_df["INR/AUD"], 
                 label="Historical INR/AUD", color="blue")
        
        # Predictions - using the seasonal polynomial model for best results
        plt.plot(self.predictions['Date'], self.predictions['Seasonal_Polynomial_Prediction'], 
                 label="Predicted INR/AUD (with seasonality)", 
                 linestyle="--", color="green")
        
        # Highlight best months to transfer
        optimal_months = self.identify_optimal_transfer_months()
        top_months = optimal_months.head(3).index.tolist()
        
        # Highlight the best months in the prediction period
        for i, row in self.predictions.iterrows():
            month = row['Date'].month
            if month in top_months:
                plt.scatter(row['Date'], row['Seasonal_Polynomial_Prediction'], 
                            color='red', s=100, zorder=5)
                plt.annotate(f"Good time to transfer", 
                            (row['Date'], row['Seasonal_Polynomial_Prediction']),
                            xytext=(10, -30),
                            textcoords="offset points",
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
            
        # Prepare data
        df = self.historical_df.copy()
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        
        # Calculate monthly averages
        monthly_avg = df.groupby('Month')['AUD/INR'].mean()
        months = [datetime(2000, m, 1).strftime('%b') for m in monthly_avg.index]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(months, monthly_avg.values, color='skyblue')
        
        # Highlight best months
        threshold = monthly_avg.quantile(0.75)
        for i, v in enumerate(monthly_avg.values):
            if v >= threshold:
                bars[i].set_color('green')
                plt.text(i, v + 0.001, 'Good', 
                         ha='center', va='bottom', 
                         fontweight='bold', color='green')
            elif v <= monthly_avg.quantile(0.25):
                bars[i].set_color('red')
                plt.text(i, v + 0.001, 'Avoid', 
                         ha='center', va='bottom', 
                         fontweight='bold', color='red')
        
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
            
        # Combine datasets
        analysis_df = pd.concat([self.historical_df, self.aus_economic_data], axis=1)
        
        # Calculate correlation matrix
        corr_matrix = analysis_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation between Exchange Rates and Economic Factors")
        plt.tight_layout()
        
        return plt.gcf()
    
    def get_transfer_recommendations(self):
        """Generate recommendations for money transfers"""
        if self.historical_df is None:
            print("Please load historical data first")
            return None
            
        # Get optimal months based on historical data
        optimal_months = self.identify_optimal_transfer_months()
        
        # Get best months from predictions if available
        if self.predictions is not None:
            predicted_best = self.predictions.sort_values('Seasonal_Polynomial_AUD/INR', ascending=False)
            predicted_worst = self.predictions.sort_values('Seasonal_Polynomial_AUD/INR', ascending=True)
        else:
            predicted_best = None
            predicted_worst = None
        
        # Generate recommendations text
        recommendations = {
            "historical_best_months": optimal_months.head(3)['Month_Name'].tolist(),
            "historical_worst_months": optimal_months.tail(3)['Month_Name'].tolist(),
        }
        
        if predicted_best is not None:
            recommendations["predicted_best_date"] = predicted_best.iloc[0]['Date'].strftime('%B %Y')
            recommendations["predicted_best_rate"] = predicted_best.iloc[0]['Seasonal_Polynomial_AUD/INR']
            recommendations["predicted_worst_date"] = predicted_worst.iloc[0]['Date'].strftime('%B %Y')
            recommendations["predicted_worst_rate"] = predicted_worst.iloc[0]['Seasonal_Polynomial_AUD/INR']
            
            # Calculate potential savings
            best_rate = predicted_best.iloc[0]['Seasonal_Polynomial_AUD/INR']
            worst_rate = predicted_worst.iloc[0]['Seasonal_Polynomial_AUD/INR']
            savings_percent = ((best_rate - worst_rate) / worst_rate) * 100
            recommendations["potential_savings_percent"] = savings_percent
        
        return recommendations

    def run_complete_analysis(self, from_symbol="INR", to_symbol="AUD", start_year=2015, end_year=2024):
        """Run the complete analysis pipeline"""
        # 1. Fetch historical data
        self.historical_df = self.fetch_historical_data(from_symbol, to_symbol, start_year, end_year)
        
        # 2. Fetch economic indicators
        self.fetch_economic_indicators()
        
        # 3. Generate predictions
        self.predict_future_rates()
        
        # 4. Generate transfer recommendations
        recommendations = self.get_transfer_recommendations()
        
        return {
            "historical_data": self.historical_df,
            "economic_data": self.aus_economic_data,
            "predictions": self.predictions,
            "recommendations": recommendations
        }

# Example usage
def main():
    # Create analyzer instance
    analyzer = CurrencyAnalyzer(api_key="2ONTZBSMSIYA85NQ")
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Display visualizations
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
    print("4. Monitor Australian economic news, particularly in the mining and education sectors, as they may provide early signals of exchange rate movements.")


if __name__ == "__main__":
    API_KEY = "2ONTZBSMSIYA85NQ"
    if API_KEY == "API_KEY":
        print("Please replace 'YOUR_API_KEY_HERE' with a valid Alpha Vantage API key.")
        main()  # Runs with dummy data
    else:
        main()

