import asyncio
import logging
import yfinance as yf
import pandas as pd
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import os

# Налаштування логування
logging.basicConfig(level=logging.INFO)

# Конфігурація бота
BOT_TOKEN = os.getenv("BOT_TOKEN")  # Додай в Render Environment
DATABASE_URL = os.getenv("DATABASE_URL")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Форекс пари
PAIR_MAPPING = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/JPY": "JPY=X",
    "CAD/JPY": "CADJPY=X"
}

def calculate_indicators(df):
    # Тут твоя логіка індикаторів (наприклад, SMA, RSI)
    return df

def _fetch_yf(ticker, period, interval):
    # Додаємо спробу завантаження
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        return data
    except Exception as e:
        logging.error(f"Помилка завантаження {ticker}: {e}")
        return pd.DataFrame()

async def cached_yf_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    # Обробка помилок, щоб бот не падав
    try:
        df = await asyncio.to_thread(_fetch_yf, ticker, period, interval)
        if df is None or df.empty:
            logging.warning(f"Дані для {ticker} порожні (ринок закритий?).")
            return pd.DataFrame()
        
        # Очищення колонок, якщо приходить MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_indicators(df)
        return df
    except Exception as e:
        logging.warning(f"Ринок закритий або відсутні дані для {ticker}: {e}")
        return pd.DataFrame()

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    builder = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=pair, callback_data=pair)] for pair in PAIR_MAPPING.keys()
    ])
    await message.answer("Виберіть торгову пару:", reply_markup=builder)

@dp.callback_query(F.data.in_(PAIR_MAPPING.keys()))
async def handle_pair(callback: types.CallbackQuery):
    pair = callback.data
    ticker = PAIR_MAPPING[pair]
    
    await callback.message.edit_text(f"⏳ Сканування {pair}...")
    
    df = await cached_yf_download(ticker, "1d", "1m")
    
    if df.empty:
        await callback.message.edit_text("⚠️ Недостатньо даних котирувань. Ринок може бути закритий.")
    else:
        await callback.message.edit_text(f"✅ Дані по {pair} успішно отримані!")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
