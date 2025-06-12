import yfinance as yf
import pandas as pd

def get_bitcoin_history_yf(start_date, end_date):
    """ 
    This function fetches the historical price data for Bitcoin (BTC) using the yfinance library.

    Args:
        start_date (str): The start date for fetching historical data in 'YYYY-MM-DD' format.
        end_date (str): The end date for fetching historical data in 'YYYY-MM-DD'
    
    Returns:
        pandas.DataFrame: A DataFrame containing the date and closing price of Bitcoin.
    """
    btc = yf.Ticker("BTC-USD")
    hist = btc.history(start=start_date, end=end_date)
    return hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})

def process_dataframe(df):
    """ 
    This function processes the DataFrame to prepare it for forecasting.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the historical data.

    Returns:
        pandas.DataFrame: A processed DataFrame with columns 'ds', 'y', and 'unique_id'.
    """
    df = df.rename(
        columns={
            "date": "ds",
            "price": "y"
        }
    )
    df['ds'] = pd.to_datetime(df['ds'])
    df['unique_id'] = "bitcoin"
    return df

def predict_timesfm(df, model):
    """ 
    Makes predictions using a trained TimesFM model on the provided DataFrame.

    Args:

        df (pandas.DataFrame): The input DataFrame containing the data to be forecasted.
        model (TimesFM): A trained TimesFM model for forecasting.

    Returns:
        pandas.DataFrame: A DataFrame containing the forecasted values with columns 'ds', 'unique_id', and 'yhat'.
    """
    forecast_df = model.forecast_on_df(
        inputs=df,
        freq="D",  # monthly
        value_name="y",
        num_jobs=2,
    )
    forecast_df = forecast_df[['ds', 'unique_id', 'timesfm']]
    forecast_df = forecast_df.rename(
        columns={
            "timesfm": "y"
        }
    )
    return forecast_df