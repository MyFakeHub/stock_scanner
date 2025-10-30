import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
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
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'ctbu4n9r01qoo45pautgctbu4n9r01qoo45pauv0')  # Free tier for news
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
    
    # Default watchlist - focus on volatile stocks under $10
    return [
        "PLTER", "NFLX", "META", "NVDA", "AMD", 
        "CGNX", "NNE", "QCOM", "TM", "SNOW",
        "ABM", "PENG", "QS", "LCID", "LMT"
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

def fetch_company_news(ticker):
    """Fetch recent news for stock (last 7 days)"""
    try:
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        news = response.json()
        
        # Look for important keywords in headlines
        important_keywords = [
            'breakthrough', 'partnership', 'acquisition', 'merger', 'fda', 'approval',
            'contract', 'deal', 'record', 'surge', 'earnings beat', 'upgraded',
            'bullish', 'collaboration', 'innovation', 'patent', 'award'
        ]
        
        strong_news = []
        if isinstance(news, list):
            for item in news[:10]:  # Check last 10 news items
                headline = item.get('headline', '').lower()
                if any(keyword in headline for keyword in important_keywords):
                    strong_news.append({
                        'headline': item.get('headline'),
                        'date': datetime.fromtimestamp(item.get('datetime', 0)).strftime('%Y-%m-%d'),
                        'source': item.get('source', 'Unknown')
                    })
        
        return strong_news if strong_news else None
        
    except Exception as e:
        logger.debug(f"Error fetching news for {ticker}: {e}")
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

def detect_consolidation_pattern(df, weeks=3):
    """
    Detect very narrow range consolidation for several weeks
    النطاق الضيق جدا لعدة أسابيع
    """
    try:
        # Get last N weeks of data
        days = weeks * 5  # Trading days
        recent_data = df.tail(days)
        
        if len(recent_data) < days:
            return False, 0
        
        # Calculate price range
        high_price = recent_data['High'].max()
        low_price = recent_data['Low'].min()
        price_range_pct = ((high_price - low_price) / low_price) * 100
        
        # Very tight consolidation: less than 15% range over the period
        is_tight = price_range_pct < 15
        
        return is_tight, price_range_pct
        
    except Exception as e:
        return False, 0

def detect_volume_pattern(df):
    """
    Detect decreased volume during consolidation + sudden spike
    انخفاض الفوليوم أثناء التماسك + زيادة مفاجئة في الفوليوم
    """
    try:
        volume = df['Volume']
        
        # Average volume over last 20 days
        vol_avg_20 = volume.tail(20).mean()
        
        # Volume during consolidation (days 3-20 ago)
        consolidation_volume = volume.iloc[-20:-2].mean()
        
        # Recent volume (last 2 days)
        recent_volume = volume.tail(2).mean()
        
        # Check: consolidation volume lower than average
        low_volume_consolidation = consolidation_volume < vol_avg_20 * 0.8
        
        # Check: sudden spike in last 1-2 days
        volume_spike = recent_volume > vol_avg_20 * 1.5
        
        volume_ratio = recent_volume / vol_avg_20
        
        return low_volume_consolidation and volume_spike, volume_ratio
        
    except Exception as e:
        return False, 0

def detect_resistance_breakout(df):
    """
    Detect resistance breakout on daily/weekly timeframe
    كسر مقاومة على فريم يومي أو أسبوعي
    """
    try:
        close = df['Close']
        high = df['High']
        
        # Find resistance (highest high in last 50 days, excluding last 5 days)
        resistance = high.iloc[-50:-5].max()
        
        # Current price
        current_price = close.iloc[-1]
        
        # Breakout: current price above resistance
        breakout = current_price > resistance * 1.02  # 2% above resistance
        
        breakout_pct = ((current_price - resistance) / resistance) * 100
        
        return breakout, breakout_pct, resistance
        
    except Exception as e:
        return False, 0, 0

def detect_near_strong_support(df):
    """
    Check if trading near previous strong support (bottom)
    يتداول قرب قاع قوي سابق
    """
    try:
        low = df['Low']
        close = df['Close']
        
        # Find strong support (lowest low in last 90 days)
        support_level = low.tail(90).min()
        
        # Current price
        current_price = close.iloc[-1]
        
        # Near support: within 20% above support
        near_support = current_price <= support_level * 1.20
        
        distance_from_support = ((current_price - support_level) / support_level) * 100
        
        return near_support, distance_from_support, support_level
        
    except Exception as e:
        return False, 0, 0

def calculate_potential_gain(current_price, resistance):
    """
    Calculate potential for 200%+ gain
    إمكانية الارتفاع أكثر من 200%
    """
    # This is speculative, but we can check historical volatility
    # For now, we'll mark stocks that have shown high volatility
    try:
        potential = (resistance * 3 - current_price) / current_price * 100
        return potential > 200
    except:
        return False

def analyze_stock(ticker):
    """
    Analyze stock based on Arabic requirements
    تحليل السهم حسب الشروط المطلوبة
    """
    try:
        # Fetch data
        df = fetch_stock_data(ticker)
        
        if df is None or df.empty or len(df) < 90:
            return None
        
        # Get recent data (last 90 days for analysis)
        df = df.tail(90)
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # ===== CONDITION 1: Price under $10 =====
        current_price = close.iloc[-1]
        if current_price >= PRICE_LIMIT:
            return None
        
        # ===== CONDITION 2: RSI between 45-55 =====
        rsi = calculate_rsi(close)
        current_rsi = rsi.iloc[-1]
        rsi_ok = 45 < current_rsi < 55
        
        if not rsi_ok:
            return None
        
        # ===== CONDITION 3: MACD bullish crossover =====
        macd_line, signal_line = calculate_macd(close)
        macd_cross = (macd_line.iloc[-2] <= signal_line.iloc[-2] and 
                      macd_line.iloc[-1] > signal_line.iloc[-1])
        
        # ===== CONDITION 4: Very narrow consolidation for weeks =====
        is_consolidating, range_pct = detect_consolidation_pattern(df, weeks=3)
        
        # ===== CONDITION 5: Volume pattern (low during consolidation, spike before breakout) =====
        volume_pattern_ok, volume_ratio = detect_volume_pattern(df)
        
        # ===== CONDITION 6: Resistance breakout =====
        breakout, breakout_pct, resistance = detect_resistance_breakout(df)
        
        # ===== CONDITION 7: Near strong support =====
        near_support, support_distance, support_level = detect_near_strong_support(df)
        
        # ===== CONDITION 8: Strong news/catalyst =====
        news = fetch_company_news(ticker)
        has_catalyst = news is not None and len(news) > 0
        
        # ===== SCORING SYSTEM =====
        score = 0
        signals = []
        
        # Critical conditions (must have most of these)
        if rsi_ok:
            score += 2
            signals.append(f"✅ RSI: {current_rsi:.1f} (في النطاق المثالي)")
        
        if macd_cross:
            score += 3
            signals.append("✅ MACD: تقاطع صعودي حديث")
        
        if is_consolidating:
            score += 2
            signals.append(f"✅ التماسك: نطاق ضيق {range_pct:.1f}%")
        
        if volume_pattern_ok:
            score += 3
            signals.append(f"✅ الفوليوم: ارتفاع مفاجئ {volume_ratio:.1f}x")
        
        if breakout:
            score += 3
            signals.append(f"✅ كسر المقاومة: {breakout_pct:.1f}% فوق المقاومة")
        
        if near_support:
            score += 2
            signals.append(f"✅ قرب الدعم: {support_distance:.1f}% من القاع")
        
        if has_catalyst:
            score += 3
            signals.append(f"✅ أخبار قوية: {len(news)} خبر إيجابي")
        
        # Minimum score threshold: 12 out of 18
        if score < 12:
            return None
        
        # Calculate signal strength
        signal_strength = "🔥🔥🔥 قوي جداً" if score >= 16 else "🔥🔥 قوي" if score >= 14 else "🔥 واعد"
        
        return {
            'ticker': ticker,
            'price': current_price,
            'score': score,
            'signal_strength': signal_strength,
            'rsi': current_rsi,
            'volume_ratio': volume_ratio,
            'breakout_pct': breakout_pct,
            'consolidation_range': range_pct,
            'resistance': resistance,
            'support': support_level,
            'signals': signals,
            'news': news[:3] if news else [],  # Top 3 news items
            'signal_type': 'انفجار محتمل - EXPLOSIVE BREAKOUT'
        }
        
    except Exception as e:
        logger.debug(f"Error analyzing {ticker}: {e}")
        return None

def scan_stocks():
    """Scan all stocks in watchlist"""
    logger.info("="*60)
    logger.info(f"بدء المسح - Starting scan at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    signals = []
    successful_scans = 0
    failed_scans = 0
    
    for i, ticker in enumerate(WATCHLIST, 1):
        logger.info(f"[{i}/{len(WATCHLIST)}] تحليل - Analyzing {ticker}...")
        signal = analyze_stock(ticker)
        
        if signal:
            signals.append(signal)
            logger.info(f"✅ {ticker}: {signal['signal_strength']}")
            successful_scans += 1
        else:
            failed_scans += 1
        
        # Alpha Vantage free tier: 5 requests/minute
        if i < len(WATCHLIST):
            time.sleep(15)
    
    logger.info(f"المسح مكتمل - Scan complete: {successful_scans} successful, {failed_scans} failed")
    
    # Send Telegram notifications
    if signals:
        # Sort by score
        signals.sort(key=lambda x: x['score'], reverse=True)
        
        message = f"🚀 <b>تنبيه اختراق قوي - EXPLOSIVE BREAKOUT ALERT</b>\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"🎯 وجدنا {len(signals)} سهم واعد\n\n"
        message += "="*30 + "\n\n"
        
        for signal in signals:
            message += f"<b>💎 {signal['ticker']}</b> - {signal['signal_strength']}\n"
            message += f"💰 السعر: ${signal['price']:.2f}\n"
            message += f"⭐ النقاط: {signal['score']}/18\n"
            message += f"📊 RSI: {signal['rsi']:.1f}\n"
            message += f"📈 الفوليوم: {signal['volume_ratio']:.1f}x\n"
            message += f"🎯 كسر المقاومة: +{signal['breakout_pct']:.1f}%\n"
            message += f"📉 نطاق التماسك: {signal['consolidation_range']:.1f}%\n\n"
            
            message += "<b>الإشارات:</b>\n"
            for sig in signal['signals']:
                message += f"{sig}\n"
            
            if signal['news']:
                message += f"\n<b>📰 الأخبار الأخيرة:</b>\n"
                for news_item in signal['news']:
                    message += f"• {news_item['headline'][:80]}...\n"
            
            message += "\n" + "="*30 + "\n\n"
        
        # Split message if too long
        if len(message) > 4000:
            messages = [message[i:i+4000] for i in range(0, len(message), 4000)]
            for msg in messages:
                send_telegram_message(msg)
                time.sleep(1)
        else:
            send_telegram_message(message)
        
        logger.info(f"✅ تم إرسال {len(signals)} تنبيه - Sent {len(signals)} alert(s)")
    else:
        logger.info("❌ لا توجد إشارات - No signals detected")
    
    # Update health check file
    with open('/tmp/scanner_running', 'w') as f:
        f.write(str(time.time()))

def main():
    """Main loop"""
    logger.info("🚀 ماسح الأسهم بدأ - Stock Scanner Started!")
    logger.info(f"📊 مراقبة - Monitoring {len(WATCHLIST)} stocks")
    logger.info(f"💰 حد السعر - Price limit: ${PRICE_LIMIT}")
    logger.info(f"⏰ فترة المسح - Scan interval: {SCAN_INTERVAL} seconds")
    
    # Calculate minimum scan time
    min_scan_time = len(WATCHLIST) * 15
    logger.info(f"⏱️  Minimum scan time: ~{min_scan_time} seconds")
    
    # Adjust scan interval if needed
    scan_interval = SCAN_INTERVAL
    if scan_interval < min_scan_time:
        scan_interval = min_scan_time + 60
        logger.warning(f"⚠️  SCAN_INTERVAL too short! Using {scan_interval} seconds instead")
    
    # Test Telegram connection
    test_msg = send_telegram_message("🤖 ماسح الأسهم يعمل الآن!\n🤖 Stock Scanner is now running!")
    if not test_msg:
        logger.error("❌ Failed to send test message to Telegram")
        sys.exit(1)
    
    logger.info("✅ Telegram connection successful")
    
    while True:
        try:
            scan_stocks()
            logger.info(f"⏳ المسح القادم خلال - Next scan in {scan_interval} seconds...")
            time.sleep(scan_interval)
        except KeyboardInterrupt:
            logger.info("🛑 توقف الماسح - Scanner stopped")
            send_telegram_message("🛑 توقف ماسح الأسهم - Stock Scanner stopped")
            break
        except Exception as e:
            logger.error(f"❌ خطأ - Error: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
