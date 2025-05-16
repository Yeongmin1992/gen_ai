import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# 필수라이브러리 불러오기
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from crewai_tools import SerperDevTool

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()
# 툴 설정

# 웹 검색 툴 설정
search_tool = SerperDevTool()

# 재무 분석 툴 설정
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
    balance_sheet = ticker.get_balance_sheet()

    # 주요 재무 지표(연간)
    # 가장 첫 번째 열(가장 최근의 재무정보)에 있는 TotalRevenue 값을 가져오겟다.
    revenue = annual_financials.loc['TotalRevenue', annual_financials.columns[0]]
    cost_of_revenue = annual_financials.loc['CostOfRevenue', annual_financials.columns[0]]
    gross_profit = annual_financials.loc['GrossProfit', annual_financials.columns[0]]
    operating_income = annual_financials.loc['OperatingIncome', annual_financials.columns[0]]
    net_income = annual_financials.loc['NetIncome', annual_financials.columns[0]]
    abitda = annual_financials.loc['EBITDA', annual_financials.columns[0]]

    # 부채비율 계산
    total_assets = balance_sheet.loc['TotalAssets', balance_sheet.columns[0]]
    total_liabilities = balance_sheet.loc['TotalLiabilitiesNetMinorityInterest', balance_sheet.columns[0]]
    debt_ratio = (total_liabilities / total_assets) * 100 if total_assets != 0 else None

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
            "부채비율": f"{debt_ratio:.2f}%" if debt_ratio is not None else "N/A",
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

# 에이전트 설정

# 현재 시간 가져오기
current_time = datetime.now()
llm = ChatOpenAI(model = "gpt-4o-mini")
invest_llm = ChatAnthropic(model = "claude-3-5-sonnet-20240620")

# 재무 분석가
financial_analyst = Agent(
    role="Financial Analyst",
    goal="회사의 재무 상태 및 성과 분석",
    backstory="당신은 재무 제표와 비율을 해석하는데 전문성을 갖춘 노련한 재무 분석가입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    tools=[comprehensive_stock_analysis],
    llm=llm,
    max_iter=3,
    allow_delegation=False,
    verbose=True
)

# 시장 분석가
market_analyst = Agent(
    role="Market Analyst",
    goal="회사의 시장 지위 및 업계 동향 분석",
    backstory="당신은 기업/산업 현황 및 경쟁 환경을 전문적으로 분석할 수 있는 숙련된 시장 분석가입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    tools=[search_tool],
    llm=llm,
    max_iter=3,
    allow_delegation=False,
    verbose=True
)

# 위험 분석가
risk_analyst = Agent(
    role="Risk Analyst",
    goal="주식과 관련된 잠재적 위험 식별 및 평가",
    backstory="당신은 투자에서 명백한 위험과 숨겨진 위험을 모두 식별하는 예리한 안목을 갖춘 신중한 위험 분석가입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    tools=[comprehensive_stock_analysis],
    llm=llm,
    max_iter=3,
    allow_delegation=False,
    verbose=True
)

# 투자 분석가
investment_advisor = Agent(
    role="Investment Analyst",
    goal="전체 분석을 기반으로 한 투자 추천 제공",
    backstory="다양한 분석을 종합하여 전략적 투자 조언을 제공하는 신뢰할 수 있는 투자 자문가 입니다. 날짜: {current_time:%Y년 %m월 %d일}",
    llm=invest_llm,
    allow_delegation=False,
    verbose=True
)

def get_user_input():
    ticker = input("투자 자문을 구하고 싶은 기업명을 입력해주세요: ")
    return ticker

def create_dynamic_task(ticker):
    financial_analysis = Task(
        description=f"""{ticker}에 대한 철저한 재무 분석을 수행합니다.
        주요 재무 지표에 집중하세요.
        회사의 재무 건정성 및 성과 추세에 대한 인사이트를 제공합니다. 날짜:  {current_time:%Y년 %m월 %d일}
        """,
        agent=financial_analyst,
        expected_output=f"""{ticker}의 재무상태에 대한 종합적인 분석 보고서.
        주요 재무지표, 수익성, 부채비율 등을 포함하며,
        회사의 재무 건전성과 성과 동향에 대한 인사이트를 제공하여야 합니다.
        """
    )

    market_analysis = Task(
        description=f"""{ticker}의 시장 위치를 분석합니다.
        경쟁우위, 시장점유율, 업계동향을 평가하세요,
        회사의 성장 잠재력과 시장 과제에 대한 인사이트를 제공하세요. 
        날짜:  {current_time:%Y년 %m월 %d일}
        """,
        agent=market_analyst,
        expected_output=f"""{ticker}의 시장위치에 대한 상세한 분석 보고서,
        경쟁 우위, 시장 점유율, 산업 동향을 평가하고,
        회사의 성장 잠재력과 시장 과제에 대한 인사이트를 포함해야 합니다.
        """
    )

    risk_assessment = Task(
        description=f"""{ticker}에 대한 투자와 관련된 주요 위험을 파악하고 평가합니다.
        시장 위험, 운영위험, 재무위험 및 회사별 위험을 고려하세요.
        통합적인 위험 프로필을 제공합니다.
        날짜:  {current_time:%Y년 %m월 %d일}
        """,
        agent=risk_analyst,
        expected_output=f"""{ticker} 투자와 관련된 주요 리스크에 대한 포괄적인 평과 보고서.
        시장 리스크, 운영 리스크, 재무 리스크, 회사의 특정 리스크를 고려하여
        종합적인 리스크 분석 결과를 제시해야 합니다.
        """
    )

    investment_recommendation = Task(
        description=f"""{ticker}의 재무분석, 시장분석, 위험평가를 바탕으로 종합적인 투자 추천을 제공합니다.
        주식의 잠재 수익률, 위험 및 다양한 유형의 투자자에 대한 적합성을 고려하세요.
        한글로 작성하세요.
        날짜:  {current_time:%Y년 %m월 %d일}
        """,
        agent=investment_advisor,
        expected_output=f"""
        1. 제목 및 기본 정보
            - 회사명, 티커, 현재 주가, 목표주가, 투자의견 등

        2. 요약(Executive Summary)
            - 핵심 투자 포인트와 주요 재무 지표를 간단히 정리

        3. 기업 개요
            - 회사의 주요 사업 영역, 연혁, 시장 점유율 등

        4. 산업 및 시장 분석
            - 해당 기업이 속한 산업의 트렌드와 전망

        5. 재무 분석
            - 매출, 영업이익, 순이익 등 주요 재무지표 분석
            - 수익성, 성장성, 안정성 지표 분석

        6. 밸류에이션
            - P/E, P/B, ROE 등 주요 밸류에이션 지표 분석
            - 경쟁사 대비 상대 밸류에이션

        7. 투자 의견 및 목표주가
            - 투자의견 제시 및 근거 설명
            - 목표주가 산정 바법과 근거

        8. 투자 위험 요인
            - 잠재적인 리스크 요인들을 나열

        9. 재무제표 요약
            - 최근 몇 년간의 요약 손익계산서, 재무상태표, 현금흐름표
        """
    )

    return [financial_analysis, market_analysis, risk_assessment, investment_recommendation]

ticker = get_user_input()
tasks = create_dynamic_task(ticker)

crew = Crew(
    agents=[financial_analyst, market_analyst, risk_analyst, investment_advisor],
    tasks=tasks,
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()

from IPython.display import display, Markdown
display(Markdown(result.raw))