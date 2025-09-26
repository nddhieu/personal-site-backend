import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlencode
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from mistralai import Mistral
except Exception:
    Mistral = None

logger = logging.getLogger(__name__)

# Logging config toggles via environment
LOG_HTTP_BODY = os.getenv("LOG_HTTP_BODY", "false").lower() == "true"
LOG_LLM_PROMPTS = os.getenv("LOG_LLM_PROMPTS", "false").lower() == "true"

class AlphaVantageService:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.session = session or requests.Session()

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set")
        params = {**params, "apikey": self.api_key}
        url = f"{self.BASE_URL}?{urlencode(params)}"
        try:
            if LOG_HTTP_BODY:
                logger.debug(f"AlphaVantage request | url={url}")
            else:
                redacted = url.replace(self.api_key, "***") if self.api_key else url
                logger.debug(f"AlphaVantage request | url={redacted}")
            r = self.session.get(url, timeout=15)
            status = r.status_code
            logger.debug(f"AlphaVantage response status={status}")
            r.raise_for_status()
            data = r.json()
            if LOG_HTTP_BODY:
                logger.debug(f"AlphaVantage response body={data}")
            else:
                logger.debug(f"AlphaVantage response body keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
            if isinstance(data, dict) and (data.get("Note") or data.get("Information") or data.get("Error Message")):
                logger.warning("AlphaVantage API notice | keys_present=" + ",".join([k for k in ['Note','Information','Error Message'] if k in data]))
                return data
            return data
        except Exception as e:
            logger.warning(f"AlphaVantage request failed: {e}")
            raise

    # Core stock APIs
    def quote(self, symbol: str) -> Dict[str, Any]:
        return self._get({"function": "GLOBAL_QUOTE", "symbol": symbol})

    def time_series_intraday(self, symbol: str, interval: str = "5min", outputsize: str = "compact") -> Dict[str, Any]:
        return self._get({"function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": interval, "outputsize": outputsize})

    def time_series_daily(self, symbol: str, adjusted: bool = False, outputsize: str = "compact") -> Dict[str, Any]:
        func = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
        return self._get({"function": func, "symbol": symbol, "outputsize": outputsize})

    def time_series_weekly(self, symbol: str, adjusted: bool = False) -> Dict[str, Any]:
        func = "TIME_SERIES_WEEKLY_ADJUSTED" if adjusted else "TIME_SERIES_WEEKLY"
        return self._get({"function": func, "symbol": symbol})

    def time_series_monthly(self, symbol: str, adjusted: bool = False) -> Dict[str, Any]:
        func = "TIME_SERIES_MONTHLY_ADJUSTED" if adjusted else "TIME_SERIES_MONTHLY"
        return self._get({"function": func, "symbol": symbol})

    def search(self, keywords: str) -> Dict[str, Any]:
        return self._get({"function": "SYMBOL_SEARCH", "keywords": keywords})

    # Alpha Intelligence
    def news_sentiment(self, tickers: Optional[str] = None, topics: Optional[str] = None, time_from: Optional[str] = None, time_to: Optional[str] = None, sort: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"function": "NEWS_SENTIMENT"}
        if tickers: params["tickers"] = tickers
        if topics: params["topics"] = topics
        if time_from: params["time_from"] = time_from
        if time_to: params["time_to"] = time_to
        if sort: params["sort"] = sort
        if limit: params["limit"] = limit
        return self._get(params)

    def top_gainers_losers(self) -> Dict[str, Any]:
        return self._get({"function": "TOP_GAINERS_LOSERS"})

    # Fundamentals
    def overview(self, symbol: str) -> Dict[str, Any]:
        return self._get({"function": "OVERVIEW", "symbol": symbol})

    def income_statement(self, symbol: str) -> Dict[str, Any]:
        return self._get({"function": "INCOME_STATEMENT", "symbol": symbol})

    def balance_sheet(self, symbol: str) -> Dict[str, Any]:
        return self._get({"function": "BALANCE_SHEET", "symbol": symbol})

    def cash_flow(self, symbol: str) -> Dict[str, Any]:
        return self._get({"function": "CASH_FLOW", "symbol": symbol})

    def earnings(self, symbol: str) -> Dict[str, Any]:
        return self._get({"function": "EARNINGS", "symbol": symbol})

    def listing_status(self, date: Optional[str] = None, state: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"function": "LISTING_STATUS"}
        if date: params["date"] = date
        if state: params["state"] = state
        return self._get(params)

# Placeholder LLM service to preserve imports
class LLMService:
    def __init__(self):
        pass
    def chat(self, messages: list[dict[str, str]]) -> str:
        # Minimal placeholder; actual implementation may exist elsewhere in original app
        last = messages[-1]["content"] if messages else ""
        return f"Echo: {last}"