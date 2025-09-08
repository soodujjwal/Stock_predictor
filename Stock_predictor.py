# This script predicts the next day's stock closing price using a simple Linear Regression model.

# Import necessary libraries
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Step 1: Define the list of stock tickers
# We'll use a list of popular tech stocks as an example.
stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

for stock_ticker in stock_tickers:
    try:
        print("-" * 50)
        print(f"Processing historical data for {stock_ticker}...")
        stock_data = yf.download(stock_ticker, period="1y")

        if stock_data.empty:
            print(f"No data found for the ticker: {stock_ticker}. Skipping...")
            continue
        else:
            print("Data downloaded successfully.")
            
            # Step 2: Prepare the data
            # We'll create a new column 'Prediction' by shifting the 'Close' prices by one day.
            # The model will learn to predict this shifted value.
            stock_data['Prediction'] = stock_data['Close'].shift(-1)
            
            # Drop the last row as it will have a 'NaN' value in the 'Prediction' column.
            stock_data.dropna(inplace=True)

            # Define the features (X) and the target (y)
            # We'll use the 'Close' price as the feature to predict the next day's price.
            X = np.array(stock_data[['Close']])
            y = np.array(stock_data['Prediction'])
            
            # Step 3: Split the data into training and testing sets
            # This allows us to evaluate the model's performance on unseen data.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 4: Create and train the model
            # We'll use a Linear Regression model, a simple but effective algorithm for this task.
            model = LinearRegression()
            print("Training the model...")
            model.fit(X_train, y_train)
            print("Model training complete.")
            
            # Evaluate the model
            accuracy = model.score(X_test, y_test)
            print(f"Model accuracy on test data: {accuracy:.2f}")

            # Step 5: Make a prediction
            # Get the latest available closing price.
            last_price = np.array(stock_data['Close'].iloc[-1]).reshape(-1, 1)

            # Use the trained model to predict the next day's price.
            predicted_price = model.predict(last_price)
            
            # Print the results
            print(f"The model predicts the next day's closing price for {stock_ticker} will be: ${predicted_price[0]:.2f}")
    
    except Exception as e:
        print(f"An error occurred for {stock_ticker}: {e}")
        print("Please ensure you have the required libraries installed: 'yfinance' and 'scikit-learn'.")
        print("You can install them using the following command:")
        print("pip install yfinance scikit-learn numpy")
