from nsetools import Nse

# Create an instance of Nse
nse = Nse()

# Get the list of all stock symbols traded on the NSE
all_stock_symbols = nse.get_stock_codes()

# Remove the index and non-stock symbols
all_stock_symbols.pop('NIFTY')
all_stock_symbols.pop('BANKNIFTY')

# Fetch historical stock data for a specified period
start_date = '2010-01-01'
end_date = '2022-12-31'

# Function to calculate price performance
def calculate_price_performance(symbol):
    data = nse.get_history(symbol, start=start_date, end=end_date)
    data['Price_Performance'] = data['Close'].pct_change()
    return data['Price_Performance'].mean()

# Calculate price performance for each stock
price_performance = {symbol: calculate_price_performance(symbol) for symbol in all_stock_symbols}

# Define the criteria for selecting multibagger stocks
threshold = 2.0  # Minimum performance of 100% or more to be considered a potential multibagger
minimum_price = 10  # Minimum price per share for small investors

# Filter stocks based on criteria
multibaggers = {symbol: performance for symbol, performance in price_performance.items() if performance >= threshold}
filtered_multibaggers = {symbol: performance for symbol, performance in multibaggers.items() if nse.get_quote(symbol)['lastPrice'] > minimum_price}

# Sort the multibaggers by performance in descending order
sorted_multibaggers = sorted(filtered_multibaggers.items(), key=lambda x: x[1], reverse=True)

# Select the top N stocks for the portfolio
portfolio_size = 5
selected_stocks = sorted_multibaggers[:portfolio_size]

# Print the selected multibagger stocks for the portfolio
print("Future Multibagger Portfolio for Small Investor:")
for stock_symbol, performance in selected_stocks:
    last_price = nse.get_quote(stock_symbol)['lastPrice']
    print(f"{stock_symbol}: {performance} (Price: {last_price})")
