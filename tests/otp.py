import streamlit as st
import config
import pandas as pd

@st.cache(ttl=3600, allow_output_mutation=True)
def get_data(query: str, ttl=60) -> pd.DataFrame:
    df = get_data_no_cache(query, ttl)
    return df


def get_data_no_cache(query: str, ttl=60) -> pd.DataFrame:
    conn = config.get_rest_connector()
    df = pd.DataFrame(
        conn.jobs.create(query, cache_ttl=ttl, tws=0, twf=0).dataset.load()
    )
    if "_time" in df.columns:
        df["dt"] = pd.to_datetime(df["_time"], unit="s")
    return df


def render_query(query_template: str, query_params: dict) -> str:
    """
    Replace tokens in query with query_params dict.

    :param query_template: Query template with tokens or placeholders.
    :param query_params: Dict with replacements.
    :return: Query ready to run.
    """
    return_query = query_template
    for k, v in query_params.items():
        return_query = return_query.replace(k, v)
    return return_query
