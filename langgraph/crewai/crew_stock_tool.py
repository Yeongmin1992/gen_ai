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

# display(financial_analysis.run("META"))

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

    def calculate_growth_rate(curr, prev):
        if prev and curr and prev != 0:
            return (curr - prev) / abs(prev) * 100
        return None
    
    def format_financial_summary(financials):
        summary = {}
        for date, data in financials.items():
            date_str = date.strftime('%Y-%m-%d')
            summary[date_str] = {
                "TotalRevenue": data.get('TotalRevenue'),
                "OperatingIncome": data.get('OperatingIncome'),
                "NetIncome": data.get('NetIncome'),
                "EBITDA": data.get('EBITDA'),
                "DilutedEPS": f"${data.get('DilutedEPS'):.2f}" if pd.notna(data.get('DilutedEPS')) else "N/A"
            }
        return summary
    
    ticker = yf.Ticker(ticker)
    historical_prices = ticker.history(period='1d', interval='1m')
    latest_price = historical_prices['Close'].iloc[-1]
    latest_time = historical_prices.index[-1].strftime('%Y-%m-%d %H:%M:%S')

    # 연간 및 분기별 재무제표 데이터 가져오기
    annual_financials = ticker.get_financials()
    quarterly_financials = ticker.get_financials(freq="quarterly")

    # 주요 재무 지표(연간)
    # 가장 첫 번째 열(가장 최근의 재무정보)에 있는 TotalRevenue 값을 가져오겟다.
    revenue = annual_financials.loc['TotalRevenue', annual_financials.columns[0]]
    cost_of_revenue = annual_financials.loc['CostOfRevenue', annual_financials.columns[0]]
    gross_profit = annual_financials.loc['GrossProfit', annual_financials.columns[0]]
    operating_income = annual_financials.loc['OperatingIncome', annual_financials.columns[0]]
    net_income = annual_financials.loc['NetIncome', annual_financials.columns[0]]
    abitda = annual_financials.loc['EBITDA', annual_financials.columns[0]]

    # 주요 비용 계산
    gross_margin = (gross_profit / revenue) * 100 if revenue != 0 else None
    operating_margin = (operating_income / revenue) * 100 if revenue != 0 else None
    net_margin = (net_income / revenue) * 100 if revenue != 0 else None

    # 성장성 지표 계산(연간)
    # 작년 대비 올해의 성장성 확인
    revenue_growth = calculate_growth_rate(revenue, annual_financials.loc['TotalRevenue', annual_financials.columns[1]])
    net_income_growth = calculate_growth_rate(net_income, annual_financials.loc['NetIncome', annual_financials.columns[1]])

    # 주당 지표
    diluted_eps = annual_financials.loc['DilutedEPS', annual_financials.columns[0]]

    # 분기별 데이터 분석
    quarterly_revenue = quarterly_financials.loc['TotalRevenue', quarterly_financials.columns[0]]
    quarterly_net_income = quarterly_financials.loc['NetIncome', quarterly_financials.columns[0]]

    quarterly_revenue_growth = calculate_growth_rate(
        quarterly_revenue,
        quarterly_financials.loc['TotalRevenue', quarterly_financials.columns[1]]
    )
    quarterly_net_income_growth = calculate_growth_rate(
        quarterly_net_income,
        quarterly_financials.loc['NetIncome', quarterly_financials.columns[1]]
    )

    return {
        "현재 주가":{
            "현재 주가": latest_price,
            "기준 시간": latest_time
        },
        "연간 데이터": {
            "매출": format_number(revenue),
            "매출원가": format_number(cost_of_revenue),
            "매출총이익": format_number(gross_profit),
            "영업이익": format_number(operating_income),
            "순이익": format_number(net_income),
            "EBITDA": format_number(abitda),
            "매출총이익률": f"{gross_margin:.2f}%" if gross_margin is not None else "N/A",
            "영업이익률": f"{operating_margin:.2f}%" if operating_margin is not None else "N/A",
            "순이익률": f"{net_margin:.2f}%" if net_margin is not None else "N/A",
            "매출 성장률": f"{revenue_growth:.2f}%" if revenue_growth is not None else "N/A",
            "순이익 성장률": f"{net_income_growth:.2f}%" if net_income_growth is not None else "N/A",
            "희석주당순이익(EPS)": f"{diluted_eps:.2f}%" if diluted_eps is not None else "N/A",
        },
        "분기 데이터": {
            "매출": format_number(quarterly_revenue),
            "순이익": format_number(quarterly_net_income),
            "매출 성장률(QoQ)": f"{quarterly_revenue_growth:.2f}%" if quarterly_revenue_growth is not None else "N/A",
            "순이익 성장률(QoQ)": f"{quarterly_net_income_growth:.2f}%" if quarterly_net_income_growth is not None else "N/A"
        },
        "연간재무제표요역": format_financial_summary(annual_financials),
        "분기별재무제표요역": format_financial_summary(quarterly_financials)
    }

display(comprehensive_stock_analysis.run("AAPL"))

