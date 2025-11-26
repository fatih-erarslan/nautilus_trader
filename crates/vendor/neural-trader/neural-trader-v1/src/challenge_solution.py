# Neural Trading Bot Challenge Solution
def trading_bot(price_data):
    """
    Analyze price data and return trading signal
    Input: dict with current_price, rsi, volume, trend
    Output: "BUY", "SELL", or "HOLD"
    """
    rsi = price_data.get("rsi", 50)
    
    # RSI < 30 = oversold (BUY signal)
    if rsi < 30:
        return "BUY"
    
    # RSI > 70 = overbought (SELL signal)
    elif rsi > 70:
        return "SELL"
    
    # RSI between 30-70 = neutral (HOLD)
    else:
        return "HOLD"

# Test the solution with the provided test cases
if __name__ == "__main__":
    # Test case 1: RSI = 25 (should return BUY)
    test1 = {"rsi": 25}
    print(f"Test 1 - RSI: {test1['rsi']} -> Signal: {trading_bot(test1)}")
    
    # Test case 2: RSI = 75 (should return SELL)
    test2 = {"rsi": 75}
    print(f"Test 2 - RSI: {test2['rsi']} -> Signal: {trading_bot(test2)}")
    
    # Test case 3: RSI = 50 (should return HOLD)
    test3 = {"rsi": 50}
    print(f"Test 3 - RSI: {test3['rsi']} -> Signal: {trading_bot(test3)}")