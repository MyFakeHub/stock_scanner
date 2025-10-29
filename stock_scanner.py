import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
import os
import sys
import logging

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/scanner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION FROM ENVIRONMENT =====
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', '')
PRICE_LIMIT = float(os.getenv('PRICE_LIMIT', '10.0'))
SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL', '300'))

# Validate configuration
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.error("❌ TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set!")
    sys.exit(1)

if not ALPHAVANTAGE_API_KEY:
    logger.error("❌ ALPHAVANTAGE_API_KEY must be set!")
    logger.error("Get free API key at: https://www.alphavantage.co/support/#api-key")
    sys.exit(1)

# Stock universe
def load_watchlist():
    """Load watchlist from file if exists, otherwise use default"""
    watchlist_file = '/app/watchlist.txt'
    
    if os.path.exists(watchlist_file):
        try:
            with open(watchlist_file, 'r') as f:
                stocks = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f"📋 Loaded {len(stocks)} stocks from watchlist.txt")
            return stocks
        except Exception as e:
            logger.error(f"Error reading watchlist file: {e}")
    
    # Default watchlist - keep it small for free API limits
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
        "NVDA", "META", "AMD", "NFLX", "DIS"
    ]

WATCHLIST = load_watchlist()

def send_telegram_message(message):
    """Send message to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        return None

def fetch_stock_data(ticker):
    """Fetch stock data from Alpha Vantage"""
    try:
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": ALPHAVANTAGE_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Check for API limit
        if "Note" in data:
            logger.warning(f"API limit reached: {data['Note']}")
            return None
        
        if "Error Message" in data:
            logger.debug(f"Error for {ticker}: {data['Error Message']}")
            return None
        
        if "Time Series (Daily)" not in data:
            logger.debug(f"No data for {ticker}")
            return None
        
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        
        return df
        
    except Exception as e:
        logger.debug(f"Error fetching {ticker}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    width = ((upper - lower) / sma * 100)
    return upper, lower, sma, width

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def analyze_stock(ticker):
    """Analyze a single stock using Pine Script logic"""
    try:
        # Fetch data
        df = fetch_stock_data(ticker)
        
        if df is None or df.empty or len(df) < 50:
            return None
        
        # Get recent data (last 90 days)
        df = df.tail(90)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Current price check
        current_price = close.iloc[-1]
        if current_price >= PRICE_LIMIT:
            return None
        
        # Technical calculations
        rsi = calculate_rsi(close)
        upper_bb, lower_bb, basis, bb_width = calculate_bollinger_bands(close)
        macd_line, signal_line = calculate_macd(close)
        
        # Volume analysis
        vol_avg = volume.rolling(window=20).mean()
        
        # ATR
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean()
        atr_avg = atr.rolling(14).mean()
        
        # Get latest values
        current_rsi = rsi.iloc[-1]
        current_bb_width = bb_width.iloc[-1]
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_volume = volume.iloc[-1]
        current_vol_avg = vol_avg.iloc[-1]
        current_atr = atr.iloc[-1]
        current_atr_avg = atr_avg.iloc[-1]
        
        # Consolidation detection
        tight_bb = current_bb_width < 3
        narrow_range = current_bb_width == bb_width.iloc[-40:].min()
        support_level = low.iloc[-50:].min()
        near_support = close.iloc[-1] <= support_level * 1.10
        low_volume = current_volume < current_vol_avg
        
        consolidation = tight_bb and narrow_range and near_support
        
        # Breakout detection
        resistance_daily = high.iloc[-30:-1].max()
        daily_breakout = close.iloc[-1] > resistance_daily
        
        # Momentum indicators
        rsi_ready = 45 < current_rsi < 55
        macd_cross = (macd_line.iloc[-2] <= signal_line.iloc[-2] and 
                      macd_line.iloc[-1] > signal_line.iloc[-1])
        volume_spike = current_volume > current_vol_avg * 2
        atr_increase = current_atr > current_atr_avg * 1.2
        
        # Signal conditions
        base_setup = consolidation and rsi_ready and low_volume
        daily_signal = base_setup and daily_breakout and macd_cross and volume_spike and atr_increase
        
        if daily_signal:
            return {
                'ticker': ticker,
                'price': current_price,
                'rsi': current_rsi,
                'volume_ratio': current_volume / current_vol_avg,
                'bb_width': current_bb_width,
                'signal_type': 'DAILY BREAKOUT'
            }
        elif base_setup:
            return {
                'ticker': ticker,
                'price': current_price,
                'rsi': current_rsi,
                'volume_ratio': current_volume / current_vol_avg,
                'bb_width': current_bb_width,
                'signal_type': 'CONSOLIDATION SETUP'
            }
        
        return None
        
    except Exception as e:
        logger.debug(f"Error analyzing {ticker}: {e}")
        return None

def scan_stocks():
    """Scan all stocks in watchlist"""
    logger.info("="*60)
    logger.info(f"Starting scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    signals = []
    successful_scans = 0
    failed_scans = 0
    
    for i, ticker in enumerate(WATCHLIST, 1):
        logger.info(f"[{i}/{len(WATCHLIST)}] Analyzing {ticker}...")
        signal = analyze_stock(ticker)
        
        if signal:
            signals.append(signal)
            logger.info(f"✅ {ticker}: {signal['signal_type']}")
            successful_scans += 1
        elif signal is None:
            failed_scans += 1
        
        # Alpha Vantage free tier: 5 requests/minute
        # Wait 15 seconds between requests to be safe
        if i < len(WATCHLIST):
            time.sleep(15)
    
    logger.info(f"Scan complete: {successful_scans} successful, {failed_scans} failed")
    
    # Send Telegram notifications
    if signals:
        message = f"🚀 <b>Breakout Scanner Alert</b>\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for signal in signals:
            message += f"<b>{signal['ticker']}</b> - {signal['signal_type']}\n"
            message += f"💰 Price: ${signal['price']:.2f}\n"
            message += f"📊 RSI: {signal['rsi']:.1f}\n"
            message += f"📈 Vol Ratio: {signal['volume_ratio']:.2f}x\n"
            message += f"🎯 BB Width: {signal['bb_width']:.2f}%\n\n"
        
        send_telegram_message(message)
        logger.info(f"✅ Sent {len(signals)} alert(s) to Telegram")
    else:
        logger.info("❌ No signals detected")
    
    # Update health check file
    with open('/tmp/scanner_running', 'w') as f:
        f.write(str(time.time()))

def main():
    """Main loop"""
    logger.info("🚀 Stock Scanner Started!")
    logger.info(f"📊 Monitoring {len(WATCHLIST)} stocks")
    logger.info(f"💰 Price limit: ${PRICE_LIMIT}")
    logger.info(f"⏰ Scan interval: {SCAN_INTERVAL} seconds")
    logger.info(f"⚠️  Alpha Vantage free tier: 5 requests/min, 500/day")
    
    # Calculate minimum scan time
    min_scan_time = len(WATCHLIST) * 15  # 15 seconds per stock
    logger.info(f"⏱️  Minimum scan time: ~{min_scan_time} seconds")
    
    if SCAN_INTERVAL < min_scan_time:
        logger.warning(f"⚠️  SCAN_INTERVAL too short! Setting to {min_scan_time + 60} seconds")
        global SCAN_INTERVAL
        SCAN_INTERVAL = min_scan_time + 60
    
    # Test Telegram connection
    test_msg = send_telegram_message("🤖 Stock Scanner is now running!")
    if not test_msg:
        logger.error("❌ Failed to send test message to Telegram")
        logger.error("Please check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        sys.exit(1)
    
    logger.info("✅ Telegram connection successful")
    
    while True:
        try:
            scan_stocks()
            logger.info(f"⏳ Next scan in {SCAN_INTERVAL} seconds...")
            time.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            logger.info("🛑 Scanner stopped by user")
            send_telegram_message("🛑 Stock Scanner stopped")
            break
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
