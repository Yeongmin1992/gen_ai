# pip install yfinance

import warnings
warnings.filterwarnings('ignore')

# 필수 라이브러리 불러오기
from crewai.tools import tool
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# yfinance를 통해 재무정보 가져오기
from IPython.display import Image, display
Image("https://cdn.ilyoeconomy.com/news/photo/201809/39218_36305_1848.jpg")

ticker = yf.Ticker("META")
# help(ticker)
display(ticker.history(period='5d', interval='1d'))

# 재무제표
annual_financials = ticker.get_financials()
# 대차대조표
# annual_financials = ticker.get_balance_sheet()
# income statement
# annual_financials = ticker.get_income_stmt()
# annual_financials = ticker.get_financials(freq="yearly")
display(annual_financials)

# yfinance 라이브러리로 최근 종가 불러오는 Tool 만들기
# doc string 필수! 어떤 툴인지 crewai에게 알려줘야 하기 때문(없으면 오류)
@tool
def latest_stock_price(ticker):
    """
    get latest close price of the ticker
    """
    ticker = yf.Ticker(ticker)
    historical_prices = ticker.history(period='5d', interval='1d')
    latest_price = historical_prices['Close']
    return(latest_price)

display(latest_stock_price.run("AAPL"))

# yfinance 라이브러리로 재무제표 불러오는 Tool 만들기
@tool
def financial_analysis(ticker):
    """
    important info of yearly financial sheet
    """
    ticker = yf.Ticker(ticker)
    annual_financials = ticker.get_financials()
    summary = {}
    for date, data in annual_financials.items():
        date_str = date.strftime('%Y-%m-%d')
        summary[date_str] = {
            "TotalRevenue": data.get('TotalRevenue'),
            "OperatingIncome": data.get('OperatingIncome'),
            "NetIncome": data.get('NetIncome'),
            "EBITDA": data.get('EBITDA'),
            "DilutedEPS": f"${data.get('DilutedEPS'):.2f}" if pd.notna(data.get('DilutedEPS')) else "N/A"
        }
    return summary

display(financial_analysis.run("META"))

# 함수명과 다르게 직접 tool의 이름을 지어주고자 할 경우
@tool("Updated Comprehensive Stock Analysis")
def comprehensive_stock_analysis(ticker: str) -> str:
    """
    Analyze latest and comprehensive finance statement of given ticker.
    Provide latest stock price, finance status, growth rate, valuation and other important financial ratio.
    Use latest working day data.

    :param ticker: targeted ticker of analysis
    :return: string of result of finaicial analysis
    """
    # make the number fixed type
    def format_number(number):
        if number is None or pd.isna(number):
            return "N/A"
        # , for unit 1000 and .0f for remove every number below 1's digit
        return f"{number:,.0f}"