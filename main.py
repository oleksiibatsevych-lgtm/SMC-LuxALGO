import os
import time
import logging
import asyncpg
import asyncio
import io
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from tenacity import retry, stop_after_attempt, wait_fixed
from apscheduler.schedulers.background import BackgroundScheduler

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

CACHE_TTL = 120
NOTIFICATION_CHAT_ID = os.getenv("NOTIFICATION_CHAT_ID")
DATABASE_URL = os.getenv("DATABASE_URL")
BOT_TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.getenv("PORT", 10000))
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "https://smc-luxalgo.onrender.com")

allowed_env = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = [int(uid.strip()) for uid in allowed_env.split(",") if uid.strip().isdigit()]

USER_COOLDOWNS = {}
COOLDOWN_TIME = 15

PAIR_MAPPING = {
    "CHF/JPY": "CHFJPY=X", "AUD/CAD": "AUDCAD=X", "GBP/AUD": "GBPAUD=X", 
    "EUR/USD": "EURUSD=X", "EUR/CAD": "EURCAD=X", "AUD/USD": "AUDUSD=X", 
    "AUD/CHF": "AUDCHF=X", "CAD/CHF": "CADCHF=X", "EUR/CHF": "EURCHF=X", 
    "GBP/CHF": "GBPCHF=X", "USD/CAD": "USDCAD=X", "GBP/USD": "GBPUSD=X", 
    "GBP/JPY": "GBPJPY=X", "EUR/AUD": "EURAUD=X", "CAD/JPY": "CADJPY=X", 
    "USD/CHF": "USDCHF=X", "EUR/GBP": "EURGBP=X", "USD/JPY": "USDJPY=X", 
    "AUD/JPY": "AUDJPY=X", "EUR/JPY": "EURJPY=X", "GBP/CAD": "GBPCAD=X"
}

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_atr(df, length=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=length).mean()

def calc_adx(df, period=14):
    if len(df) < period * 2:
        return pd.Series(25, index=df.index)
    alpha = 1 / period
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr = calc_atr(df, period)
    tr = tr.replace(0, np.nan).fillna(1e-9)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / tr
    
    sum_di = plus_di + minus_di
    sum_di = sum_di.replace(0, np.nan).fillna(1e-9)
    dx = 100 * np.abs(plus_di - minus_di) / sum_di
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx

def check_volume_spike(df):
    if len(df) < 20:
        return False
    avg_vol = df['Volume'].iloc[-20:-1].mean()
    last_vol = df['Volume'].iloc[-1]
    return last_vol > (avg_vol * 1.5)

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return pd.DataFrame({
        'MACD': macd_line,
        'MACDs': signal_line,
        'MACDh': macd_hist
    })

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def detect_fvg(df):
    if len(df) < 3:
        return "None"
    if df['Low'].iloc[-1] > df['High'].iloc[-3]:
        return "Bullish FVG"
    elif df['High'].iloc[-1] < df['Low'].iloc[-3]:
        return "Bearish FVG"
    return "None"

def detect_order_block(df):
    if len(df) < 5:
        return "None"
    last_body = df['Close'].iloc[-2] - df['Open'].iloc[-2]
    prev_body = df['Close'].iloc[-3] - df['Open'].iloc[-3]
    if last_body > 0 and prev_body < 0:
        return "Bullish OB"
    elif last_body < 0 and prev_body > 0:
        return "Bearish OB"
    return "None"

async def init_db():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS pair_statistics (
                pair TEXT PRIMARY KEY,
                total_signals INTEGER DEFAULT 0,
                calls_count INTEGER DEFAULT 0,
                puts_count INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                wins_count INTEGER DEFAULT 0,
                losses_count INTEGER DEFAULT 0
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS market_cache (
                key TEXT PRIMARY KEY,
                data BYTEA,
                timestamp REAL
            )
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS active_signals (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT,
                pair TEXT,
                direction TEXT,
                entry_price DOUBLE PRECISION,
                expiry_time REAL,
                confidence INTEGER
            )
        ''')
    finally:
        await conn.close()

async def get_cached_data_db(key: str) -> pd.DataFrame:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        row = await conn.fetchrow('SELECT data, timestamp FROM market_cache WHERE key = $1', key)
    finally:
        await conn.close()
    if row:
        data_blob, timestamp = row
        if time.time() - timestamp < CACHE_TTL:
            try:
                return pickle.loads(data_blob)
            except Exception as e:
                logging.error(f"Помилка десеріалізації кешу: {e}")
    return pd.DataFrame()

async def set_cached_data_db(key: str, df: pd.DataFrame):
    if df.empty:
        return
    try:
        data_blob = pickle.dumps(df)
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute('''
                INSERT INTO market_cache (key, data, timestamp)
                VALUES ($1, $2, $3)
                ON CONFLICT (key) DO UPDATE SET data = EXCLUDED.data, timestamp = EXCLUDED.timestamp
            ''', key, data_blob, time.time())
        finally:
            await conn.close()
    except Exception as e:
        logging.error(f"Помилка запису кешу в БД: {e}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _fetch_yf(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return yf.download(ticker, period=period, interval=interval, progress=False)

async def cached_yf_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    key = f"{ticker}_{period}_{interval}"
    df_cached = await get_cached_data_db(key)
    if not df_cached.empty:
        return df_cached
    try:
        df = await asyncio.to_thread(_fetch_yf, ticker, period, interval)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            await set_cached_data_db(key, df)
        return df
    except Exception as e:
        logging.error(f"Помилка завантаження Yahoo Finance для {ticker} ({interval}): {e}")
        return pd.DataFrame()

async def update_statistics(pair: str, direction: str, confidence: int):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        row = await conn.fetchrow('SELECT total_signals, calls_count, puts_count, avg_confidence FROM pair_statistics WHERE pair = $1', pair)
        is_call = "CALL" in direction
        if row is None:
            total, calls, puts, avg_conf = 1, (1 if is_call else 0), (0 if is_call else 1), float(confidence)
        else:
            total = row['total_signals'] + 1
            calls = row['calls_count'] + (1 if is_call else 0)
            puts = row['puts_count'] + (1 if not is_call else 0)
            avg_conf = ((row['avg_confidence'] * row['total_signals']) + confidence) / total
        await conn.execute('''
            INSERT INTO pair_statistics (pair, total_signals, calls_count, puts_count, avg_confidence, wins_count, losses_count)
            VALUES ($1, $2, $3, $4, $5, 0, 0)
            ON CONFLICT (pair) DO UPDATE SET
                total_signals = EXCLUDED.total_signals,
                calls_count = EXCLUDED.calls_count,
                puts_count = EXCLUDED.puts_count,
                avg_confidence = EXCLUDED.avg_confidence
        ''', pair, total, calls, puts, avg_conf)
    finally:
        await conn.close()

async def update_outcome(pair: str, is_win: bool):
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        if is_win:
            await conn.execute('UPDATE pair_statistics SET wins_count = wins_count + 1 WHERE pair = $1', pair)
        else:
            await conn.execute('UPDATE pair_statistics SET losses_count = losses_count + 1 WHERE pair = $1', pair)
    finally:
        await conn.close()

async def get_statistics_text() -> str:
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        rows = await conn.fetch('SELECT pair, total_signals, calls_count, puts_count, avg_confidence, wins_count, losses_count FROM pair_statistics ORDER BY total_signals DESC')
    finally:
        await conn.close()
    if not rows:
        return "📊 <b>Статистика порожня:</b> бот ще не згенерував жодного сигналу."
    text = "📊 <b>Накопичена статистика та Win Rate по парах:</b>\n\n"
    for row in rows:
        pair, total, calls, puts, avg_conf, wins, losses = row['pair'], row['total_signals'], row['calls_count'], row['puts_count'], row['avg_confidence'], row['wins_count'], row['losses_count']
        total_closed = wins + losses
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
        text += (
            f"💱 <b>{pair}</b>\n"
            f"  • Сигналів: <b>{total}</b> | 🟢 {calls} | 🔴 {puts}\n"
            f"  • ✅ {wins} | ❌ {losses} | 🏆 Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"  • Ймовірність: <b>{avg_conf:.1f}%</b>\n"
            f"-----------------------------------\n"
        )
    return text

def check_access(update: Update) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return update.effective_user.id in ALLOWED_USER_IDS

def check_antiflood(user_id: int) -> bool:
    now = time.time()
    last_time = USER_COOLDOWNS.get(user_id, 0)
    if now - last_time < COOLDOWN_TIME:
        return False
    USER_COOLDOWNS[user_id] = now
    return True

def generate_market_chart(df_1m: pd.DataFrame, pair: str, direction: str, entry_price: float, poc_price: float) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 5))
    recent = df_1m.tail(60)
    ax.plot(recent.index, recent['Close'], label='Close Price', color='#1f77b4', linewidth=1.5)
    ax.axhline(y=entry_price, color='#ff7f0e', linestyle='--', label=f'Entry: {entry_price:.5f}')
    ax.axhline(y=poc_price, color='#2ca02c', linestyle=':', label=f'POC: {poc_price:.5f}')
    ax.set_title(f"Multi-TF SMC Analysis: {pair} ({direction})", fontsize=12, fontweight='bold')
    ax.set_xlabel("Time (1m)", fontsize=10)
    ax.set_ylabel("Price", fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=25)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

async def analyze_market(pair: str) -> dict:
    ticker_symbol = PAIR_MAPPING.get(pair)
    try:
        # Оптимізовані 4 таймфрейми для захисту від блокування Yahoo Finance
        df_1m, df_5m, df_1h, df_1d = await asyncio.gather(
            cached_yf_download(ticker_symbol, period="1d", interval="1m"),
            cached_yf_download(ticker_symbol, period="5d", interval="5m"),
            cached_yf_download(ticker_symbol, period="30d", interval="1h"),
            cached_yf_download(ticker_symbol, period="1y", interval="1d")
        )
        
        if df_1m.empty or df_1h.empty or df_1d.empty or len(df_1m) < 30:
            return {"status": False, "reason": "Недостатньо даних котирувань мульти-ТФ"}
        
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in [21, 22, 23]:
            return {"status": False, "reason": "Нічний роловер ринку (низька ліквідність)"}

        df_1m['ATR'] = calc_atr(df_1m, length=14)
        df_1m['RSI'] = calc_rsi(df_1m['Close'], length=14)
        df_1m['ADX'] = calc_adx(df_1m, length=14)
        df_1m = pd.concat([df_1m, calc_macd(df_1m['Close'])], axis=1)
        
        current_atr = df_1m['ATR'].iloc[-1]
        current_rsi = df_1m['RSI'].iloc[-1]
        current_adx = df_1m['ADX'].iloc[-1]
        avg_price = df_1m['Close'].iloc[-1]
        
        if pd.isna(current_atr) or (current_atr / avg_price) < 0.00010:
            return {"status": False, "reason": "Ринок у стані флету (низький ATR)"}
            
        if pd.isna(current_adx) or current_adx < 22:
            return {"status": False, "reason": f"Слабкий тренд або флет (ADX: {current_adx:.1f} < 22)"}
            
        def get_trend(df):
            if len(df) < 50: 
                return "Neutral"
            e20 = calc_ema(df['Close'], 20).iloc[-1]
            e50 = calc_ema(df['Close'], 50).iloc[-1]
            return "Bullish" if e20 > e50 else "Bearish"

        trend_1d = get_trend(df_1d)
        trend_1h = get_trend(df_1h)

        fvg_val = detect_fvg(df_5m)
        ob_val = detect_order_block(df_5m)

        score = 50  
        
        bullish_weights = sum(1 for t in (trend_1d, trend_1h) if t == "Bullish")
        bearish_weights = sum(1 for t in (trend_1d, trend_1h) if t == "Bearish")

        if bullish_weights >= bearish_weights:
            direction = "🟢 CALL (Вгору)"
            score += bullish_weights * 10
            if "Bullish" in fvg_val: score += 15
            if "Bullish" in ob_val: score += 15
            if current_rsi < 65: score += 10
        else:
            direction = "🔴 PUT (Вниз)"
            score += bearish_weights * 10
            if "Bearish" in fvg_val: score += 15
            if "Bearish" in ob_val: score += 15
            if current_rsi > 35: score += 10

        if check_volume_spike(df_1m):
            score += 10

        confidence = int(min(98, max(65, score)))
        if confidence < 78:
            return {"status": False, "reason": f"Низька сумісність індикаторів (Score: {score})"}

        atr_ratio = current_atr / avg_price
        expiry_minutes = 3 if atr_ratio > 0.0008 else (5 if atr_ratio > 0.0004 else 15)
            
        recent_data = df_1m.tail(1000)
        min_p, max_p = recent_data['Close'].min(), recent_data['Close'].max()
        bins = np.linspace(min_p, max_p, 30)
        hist = pd.cut(recent_data['Close'], bins=bins, include_lowest=True)
        vol_profile = recent_data.groupby(hist, observed=False)['Volume'].sum()
        poc_bin = vol_profile.idxmax()
        poc_price = (poc_bin.left + poc_bin.right) / 2
        
        current_close = df_1m['Close'].iloc[-1]

        await update_statistics(pair, direction, confidence)
        return {
            "status": True, "pair": pair, "direction": direction, "confidence": confidence,
            "entry_price": current_close, "expiry": expiry_minutes, "poc": poc_price,
            "ob": ob_val, "fvg": fvg_val, "rsi": current_rsi, "df": df_1m
        }
    except Exception as e:
        logging.error(f"Помилка аналізу для {pair}: {e}")
        return {"status": False, "reason": "Помилка обчислення мульти-ТФ"}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not check_access(update):
        await update.message.reply_text("⛔ У вас немає доступу.")
        return
    keyboard = []
    pairs = list(PAIR_MAPPING.keys())
    for i in range(0, len(pairs), 2):
        keyboard.append([InlineKeyboardButton(pair, callback_data=f"sig_{pair}") for pair in pairs[i:i+2]])
    keyboard.append([InlineKeyboardButton("📊 Аналіз усіх пар", callback_data="signal_all")])
    keyboard.append([InlineKeyboardButton("📈 Статистика", callback_data="show_stats")])
    await update.message.reply_text("👋 Вітаю! Мультитаймфреймовий сканер готовий. Оберіть пару:", reply_markup=InlineKeyboardMarkup(keyboard))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
        return
    data = query.data
    if data == "show_stats":
        response_text = await get_statistics_text()
        keyboard = [[InlineKeyboardButton("🔙 Меню", callback_data="back_menu")]]
        await query.edit_message_text(text=response_text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
        return
    if data.startswith("win_") or data.startswith("loss_"):
        is_win = data.startswith("win_")
        pair = data.split("_", 1)[1]
        await update_outcome(pair, is_win)
        await query.answer("Результат збережено!", show_alert=False)
        return
    if data.startswith("sig_"):
        target = data.split("_", 1)[1]
        if not check_antiflood(user_id):
            await query.answer(f"⏳ Зачекайте {COOLDOWN_TIME} секунд перед наступним запитом!", show_alert=True)
            return
        res = await analyze_market(target)
        if not res["status"]:
            await query.edit_message_text(text=f"⚠️ {res['reason']}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Меню", callback_data="back_menu")]]))
            return
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            await conn.execute('''
                INSERT INTO active_signals (chat_id, pair, direction, entry_price, expiry_time, confidence)
                VALUES ($1, $2, $3, $4, $5, $6)
            ''', query.message.chat_id, target, res["direction"], res["entry_price"], time.time() + (res["expiry"] * 60), res["confidence"])
        finally:
            await conn.close()
        chart_buf = await asyncio.to_thread(generate_market_chart, res["df"], target, res["direction"], res["entry_price"], res["poc"])
        caption = (
            f"💱 Пара: <b>{target}</b>\n"
            f"📈 Напрямок: <b>{res['direction']}</b>\n"
            f"⏳ Експірація: <b>{res['expiry']} хв</b>\n"
            f"🎯 Math Score: <b>{res['confidence']}%</b>\n"
            f"📦 OB: <code>{res['ob']}</code> | FVG: <code>{res['fvg']}</code>\n"
            f"📊 RSI: <code>{res['rsi']:.1f}</code> | POC: <code>{res['poc']:.5f}</code>\n"
        )
        keyboard = [
            [InlineKeyboardButton("✅ Win", callback_data=f"win_{target}"), InlineKeyboardButton("❌ Loss", callback_data=f"loss_{target}")],
            [InlineKeyboardButton("🔙 Меню", callback_data="back_menu")]
        ]
        await query.message.delete()
        await context.bot.send_photo(chat_id=query.message.chat_id, photo=chart_buf, caption=caption, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "signal_all":
        if not check_antiflood(user_id):
            await query.answer(f"⏳ Зачекайте {COOLDOWN_TIME} секунд!", show_alert=True)
            return
        await query.edit_message_text(text="<b>⏳ Глибоке мульти-ТФ сканування ринку...</b>", parse_mode="HTML")
        
        # Послідовне сканування з паузами для уникнення лімітів API
        results = []
        for p in PAIR_MAPPING.keys():
            r = await analyze_market(p)
            results.append(r)
            await asyncio.sleep(0.3)

        res_text = "<b>📊 Звіт мультитаймфрейм сканування:</b>\n\n"
        found = False
        for r in results:
            if r["status"]:
                res_text += f"💱 <b>{r['pair']}</b> | {r['direction']} | {r['confidence']}%\n"
                found = True
        if not found:
            res_text += "<i>Наразі немає активних сигналів із високою точністю.</i>"
            
        keyboard = [[InlineKeyboardButton("🔙 Меню", callback_data="back_menu")]]
        await query.edit_message_text(text=res_text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "back_menu":
        keyboard = []
        pairs = list(PAIR_MAPPING.keys())
        for i in range(0, len(pairs), 2):
            keyboard.append([InlineKeyboardButton(pair, callback_data=f"sig_{pair}") for pair in pairs[i:i+2]])
        keyboard.append([InlineKeyboardButton("📊 Аналіз усіх пар", callback_data="signal_all")])
        keyboard.append([InlineKeyboardButton("📈 Статистика", callback_data="show_stats")])
        await query.edit_message_text("Оберіть пару:", reply_markup=InlineKeyboardMarkup(keyboard))

def background_market_scanner_sync(bot):
    if not NOTIFICATION_CHAT_ID:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async def scan():
        for pair in PAIR_MAPPING.keys():
            try:
                res = await analyze_market(pair)
                if res["status"] and res["confidence"] >= 88:
                    text = f"🔥 <b>МУЛЬТИ-ТФ АВТО-СИГНАЛ</b>\n💱 {pair}\n{res['direction']} | Score: {res['confidence']}%"
                    await bot.send_message(chat_id=NOTIFICATION_CHAT_ID, text=text, parse_mode="HTML")
            except Exception as e:
                logging.error(f"Помилка у фоновому сканері для {pair}: {e}")
            await asyncio.sleep(2)
    loop.run_until_complete(scan())
    loop.close()

def background_trade_checker_sync(bot):
    async def check():
        try:
            conn = await asyncpg.connect(DATABASE_URL)
            try:
                rows = await conn.fetch('SELECT id, chat_id, pair, direction, entry_price, expiry_time FROM active_signals WHERE expiry_time <= $1', time.time())
            finally:
                await conn.close()
            for row in rows:
                sig_id, chat_id, pair, direction, entry_price = row['id'], row['chat_id'], row['pair'], row['direction'], row['entry_price']
                ticker = PAIR_MAPPING.get(pair)
                df = await cached_yf_download(ticker, period="1d", interval="1m")
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    is_call = "CALL" in direction
                    is_win = (current_price > entry_price) if is_call else (current_price < entry_price)
                    await update_outcome(pair, is_win)
                    try:
                        result_text = f"🤖 <b>Результат угоди ({pair}):</b> {'ПЛЮС ✅' if is_win else 'МІНУС ❌'}\nВхід: {entry_price:.5f} | Завершення: {current_price:.5f}"
                        await bot.send_message(chat_id=chat_id, text=result_text, parse_mode="HTML")
                    except Exception as e:
                        logging.error(f"Не вдалося надіслати результат у чат {chat_id}: {e}")
                conn = await asyncpg.connect(DATABASE_URL)
                try:
                    await conn.execute('DELETE FROM active_signals WHERE id = $1', sig_id)
                finally:
                    await conn.close()
        except Exception as e:
            logging.error(f"Помилка у фоновій перевірці угод: {e}")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(check())
    loop.close()

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Виняток при обробці оновлення: {context.error}")

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_db())
    
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_error_handler(error_handler)
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(background_trade_checker_sync, 'interval', seconds=30, args=[application.bot])
    if NOTIFICATION_CHAT_ID:
        scheduler.add_job(background_market_scanner_sync, 'interval', minutes=15, args=[application.bot])
    scheduler.start()
    
    webhook_url = f"{RENDER_EXTERNAL_URL.rstrip('/')}/{BOT_TOKEN}"
    logging.info(f"Запуск вебхука на URL: {webhook_url}")
    
    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=webhook_url
    )

if __name__ == "__main__":
    main()
