import os

from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "agentic-ai-487001")
GA_DATASET = os.environ.get("BQ_DATASET", "bigquery-public-data.google_analytics_sample")

_client: bigquery.Client | None = None


def get_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT_ID)
    return _client


class SessionQueryParams(BaseModel):
    start_date: str = Field(description="Start date in YYYYMMDD format, e.g. '20160801'")
    end_date: str = Field(description="End date in YYYYMMDD format, e.g. '20170801'")
    limit: int = Field(default=10000, description="Max rows to return")


class SessionData(BaseModel):
    rows: int
    columns: list[str]
    data: list[dict]


def query_sessions(params: SessionQueryParams) -> SessionData:
    """
    Fetch flattened GA session data from BigQuery for the given date range.
    Returns one row per session with key traffic, behavior, and conversion fields.
    Prices are converted from micros to USD.
    """
    sql = f"""
    SELECT
        date,
        fullVisitorId,
        visitNumber,
        channelGrouping,
        trafficSource.source        AS traffic_source,
        trafficSource.medium        AS traffic_medium,
        trafficSource.campaign      AS campaign,
        device.deviceCategory       AS device_category,
        device.browser              AS browser,
        device.operatingSystem      AS operating_system,
        geoNetwork.continent        AS continent,
        geoNetwork.country          AS country,
        CAST(totals.visits       AS INT64) AS visits,
        CAST(totals.hits         AS INT64) AS hits,
        CAST(totals.pageviews    AS INT64) AS pageviews,
        CAST(totals.timeOnSite   AS INT64) AS time_on_site_seconds,
        CAST(totals.bounces      AS INT64) AS bounces,
        CAST(totals.newVisits    AS INT64) AS new_visits,
        CAST(totals.transactions AS INT64) AS transactions,
        ROUND(CAST(totals.totalTransactionRevenue AS INT64) / 1000000, 2) AS revenue_usd
    FROM
        `{GA_DATASET}.ga_sessions_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '{params.start_date}' AND '{params.end_date}'
    LIMIT {params.limit}
    """
    client = get_client()
    df: pd.DataFrame = client.query(sql).to_dataframe()

    return SessionData(
        rows=len(df),
        columns=list(df.columns),
        data=df.to_dict(orient="records"),
    )


def query_channel_conversion(params: SessionQueryParams) -> SessionData:
    """
    Aggregate conversion rate and revenue by traffic channel.
    Useful for comparing organic vs paid vs direct, etc.
    """
    sql = f"""
    SELECT
        channelGrouping                                              AS channel,
        COUNT(*)                                                     AS sessions,
        SUM(CAST(totals.transactions AS INT64))                      AS total_transactions,
        ROUND(
            SAFE_DIVIDE(
                SUM(CAST(totals.transactions AS INT64)),
                COUNT(*)
            ) * 100, 4
        )                                                            AS conversion_rate_pct,
        ROUND(
            SUM(CAST(totals.totalTransactionRevenue AS INT64)) / 1000000,
            2
        )                                                            AS total_revenue_usd,
        ROUND(
            SAFE_DIVIDE(
                SUM(CAST(totals.totalTransactionRevenue AS INT64)) / 1000000,
                COUNT(*)
            ), 4
        )                                                            AS revenue_per_session
    FROM
        `{GA_DATASET}.ga_sessions_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '{params.start_date}' AND '{params.end_date}'
    GROUP BY channel
    ORDER BY sessions DESC
    LIMIT {params.limit}
    """
    client = get_client()
    df: pd.DataFrame = client.query(sql).to_dataframe()

    return SessionData(
        rows=len(df),
        columns=list(df.columns),
        data=df.to_dict(orient="records"),
    )


def query_device_conversion(params: SessionQueryParams) -> SessionData:
    """
    Aggregate conversion rate by device category (desktop / mobile / tablet).
    """
    sql = f"""
    SELECT
        device.deviceCategory                                        AS device,
        COUNT(*)                                                     AS sessions,
        SUM(CAST(totals.transactions AS INT64))                      AS total_transactions,
        ROUND(
            SAFE_DIVIDE(
                SUM(CAST(totals.transactions AS INT64)),
                COUNT(*)
            ) * 100, 4
        )                                                            AS conversion_rate_pct,
        ROUND(
            SUM(CAST(totals.totalTransactionRevenue AS INT64)) / 1000000,
            2
        )                                                            AS total_revenue_usd
    FROM
        `{GA_DATASET}.ga_sessions_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '{params.start_date}' AND '{params.end_date}'
    GROUP BY device
    ORDER BY sessions DESC
    """
    client = get_client()
    df: pd.DataFrame = client.query(sql).to_dataframe()

    return SessionData(
        rows=len(df),
        columns=list(df.columns),
        data=df.to_dict(orient="records"),
    )
