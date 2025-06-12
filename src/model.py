import timesfm

def get_model(forecast_horizon):
    """ 
    This function initializes and returns a TimesFM model for forecasting.

    Args:
        forecast_horizon (int): The number of time steps to forecast.

    Returns:
        TimesFM: An instance of the TimesFM model configured for the specified forecast horizon.
    """
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=32,
            horizon_len=forecast_horizon,
            num_layers=50,
            use_positional_embedding=False,
            context_len=2048,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )
    return model