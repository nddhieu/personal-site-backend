import asyncio
import json
import os

import httpx

import logging
from pydantic import BaseModel
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
_gemini = GeminiService()
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

MAX_TOKENS = 1024
GENERAL_RESTRICT_INSTRUCTION_PROMT = "Only provide the answer content. Do not include meta commentary, disclaimers, or statements about tokens or formatting."
from app.schemas.message import ChatRequest, ChatResponse

async def process_request(req: ChatRequest) -> ChatResponse:


    # STEP 1: PLANNER
    global planner_response, final_response, final_response_token
    # Initialize accumulators to avoid NameError before first use
    planner_response = ""
    final_response = ""
    final_response_token = 0

    planner_messages = [
        {"role": "system", "content": (
            "You are a routing agent. Analyze the user's request and output a JSON plan. "
            "Possible intents are 'stock_analysis', 'market_news', and 'general_chat'. "
            "The entity type is 'ticker'. Only output the JSON plan. Examples:\n"
            "User: 'Analyze Tesla (TSLA)' -> {\"intent\": \"stock_analysis\", \"entities\": [{\"type\": \"ticker\", \"value\": \"TSLA\"}]}\n"
            "User: 'give me the latest market news' -> {\"intent\": \"market_news\", \"entities\": []}\n"
            "User: 'Hi there' -> {\"intent\": \"general_chat\", \"entities\": []}"
        )},
        {"role": "user", "content": req.text},
    ]

    try:
        planner_response = _gemini.chat(planner_messages, temperature=0.0, max_tokens=1000)

        # Clean the raw string to remove markdown fences if they exist
        cleaned_response = planner_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response.removeprefix("```json").removesuffix("```").strip()

        plan = json.loads(cleaned_response)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Planner failed to decode JSON: {e}; falling back to general_chat. Raw: {planner_response}")
        plan = {"intent": "general_chat"}

    # STEP 2: EXECUTOR
    intent = plan.get("intent")

    if intent == "stock_analysis":
        entities = plan.get("entities", [])
        if not entities or entities[0].get("type") != "ticker":
            return ChatResponse(response="I can analyze a stock, but I need a valid ticker symbol.",
                                backend="gemini")
        ticker = entities[0].get("value")
        stock_data = await get_stock_analysis_data(ticker)

        if not stock_data:
            return ChatResponse(response=f"Sorry, I couldn't retrieve financial data for {ticker}.",
                                backend="gemini")

        # STEP 3: SYNTHESIZER for stock_analysis
        synthesizer_system_prompt = f"You are a smart stock analyst. Synthesize the following data into a concise analysis for a retail investor. Start the response directly with the analysis. Format response and provide response less than {MAX_TOKENS} tokens. {GENERAL_RESTRICT_INSTRUCTION_PROMT}"
        data_for_prompt = json.dumps(stock_data, indent=2)
        synthesizer_messages = [{"role": "system", "content": synthesizer_system_prompt},
                                {"role": "user", "content": f"Data for {ticker}:\n{data_for_prompt}"}]
        final_response = _gemini.chat(synthesizer_messages, temperature=0.5, max_tokens=1024)
        return ChatResponse(response=final_response, backend="gemini")

    elif intent == "market_news":
        news_data = await get_market_news_data()
        if not news_data:
            return ChatResponse(response="Sorry, I couldn't retrieve the latest market news at the moment.",
                                backend="gemini")
        else:
            logger.info(f"Market news data: {news_data}")
            final_response_token = 0
            final_response =""
            for item in news_data:

                synthesizer_system_prompt = f"You are a financial news assistant. Summarize the following market news headlines and summaries into a clear, easy-to-read list for a general audience.Format response and provide response  less than {MAX_TOKENS} tokens. {GENERAL_RESTRICT_INSTRUCTION_PROMT}"
                data_for_prompt = json.dumps(news_data, indent=2)
                synthesizer_messages = [{"role": "system", "content": synthesizer_system_prompt},
                                        {"role": "user", "content": f"Market News:\n{item}"}]
                resp = _gemini.chat(synthesizer_messages, temperature=0.5, max_tokens=1024)
                resp_token = _gemini.count_tokens(resp)
                if ((final_response_token or 0) + resp_token) > MAX_TOKENS:
                    break
                logger.info(
                    f"Market news synthesizer response: {resp} (token count: {_gemini.count_tokens(resp)})"
                )
                final_response = f"{final_response}\n{resp}" if final_response else resp
                final_response_token = _gemini.count_tokens(final_response);

        # STEP 3: SYNTHESIZER for market_news
        synthesizer_system_prompt = f"You are a financial news assistant. {req.text} Summarize the following market news headlines and summaries into a clear, easy-to-read list for a general audience.Format response and provide response  less than {MAX_TOKENS} tokens. {GENERAL_RESTRICT_INSTRUCTION_PROMT}"
        data_for_prompt = json.dumps(news_data, indent=2)
        synthesizer_messages = [{"role": "system", "content": synthesizer_system_prompt},
                                {"role": "user", "content": f"Market News:\n{final_response}"}]
        final_response = _gemini.chat(synthesizer_messages, temperature=0.5, max_tokens=MAX_TOKENS*2)
        return ChatResponse(response=final_response, backend="gemini")

    else:  # Fallback to general_chat
        general_messages = [
            {"role": "system", "content": (
                "You are a financial assistant. You can provide a detailed analysis of a stock if given a ticker "
                f"(e.g., 'analyze TSLA') or provide the latest general market news. For other topics, act as a helpful assistant. Format response and provide response  less than {MAX_TOKENS} tokens. {GENERAL_RESTRICT_INSTRUCTION_PROMT}"
            )},
            {"role": "user", "content": req.text},
        ]
        resp = _gemini.chat(general_messages, temperature=0.5, max_tokens=512)
        return ChatResponse(response=resp, backend="gemini")

async def get_market_news_data() -> list | None:
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY environment variable not set.")
        return None
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets&limit=5&apikey={ALPHA_VANTAGE_API_KEY}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200 and "feed" in response.json():
                news_feed = response.json().get("feed", [])
                return [{"title": item.get("title"), "summary": item.get("summary")} for item in news_feed]
            return None
        except httpx.RequestError as e:
            logger.exception(f"HTTP request error for market news: {e}")
            return None

async def get_stock_analysis_data(ticker: str) -> dict | None:
    if not ALPHA_VANTAGE_API_KEY:
        logger.error("ALPHA_VANTAGE_API_KEY environment variable not set.")
        return None
    base_url = "https://www.alphavantage.co/query"
    urls = {
        "quote": f"{base_url}?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}",
        "overview": f"{base_url}?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}",
        "news": f"{base_url}?function=NEWS_SENTIMENT&tickers={ticker}&limit=5&apikey={ALPHA_VANTAGE_API_KEY}"
    }
    async with httpx.AsyncClient() as client:
        try:
            tasks = [client.get(url) for url in urls.values()]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for resp in responses:
                if isinstance(resp, Exception):
                    logger.exception(f"An exception occurred during API call: {resp}")
                    return None
            quote_resp, overview_resp, news_resp = responses
            consolidated_data = {}
            if quote_resp.status_code == 200 and "Global Quote" in quote_resp.json():
                quote_data = quote_resp.json().get("Global Quote", {})
                consolidated_data['quote'] = {
                    "price": quote_data.get("05. price"),
                    "change_percent": quote_data.get("10. change percent"),
                    "volume": quote_data.get("06. volume"),
                }
            if overview_resp.status_code == 200:
                overview_data = overview_resp.json()
                consolidated_data['overview'] = {
                    "market_cap": overview_data.get("MarketCapitalization"),
                    "pe_ratio": overview_data.get("PERatio"),
                    "eps": overview_data.get("EPS"),
                    "52_week_high": overview_data.get("52WeekHigh"),
                    "52_week_low": overview_data.get("52WeekLow"),
                }
            if news_resp.status_code == 200 and "feed" in news_resp.json():
                news_feed = news_resp.json().get("feed", [])
                consolidated_data['news'] = [
                    {"title": item.get("title"), "sentiment": item.get("overall_sentiment_label")}
                    for item in news_feed[:3]
                ]
            logger.info(f"AlphaVantage responses status: quote={quote_resp.status_code}, overview={overview_resp.status_code}, news={news_resp.status_code}")
            return consolidated_data
        except httpx.RequestError as e:
            logger.exception(f"HTTP request error while fetching AlphaVantage data for {ticker}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error while fetching AlphaVantage data for {ticker}: {e}")
            return None