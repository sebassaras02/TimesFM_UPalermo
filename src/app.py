import streamlit as st
import pandas as pd
import plotly.express as px
from model import get_model
from utils import get_bitcoin_history_yf, process_dataframe, predict_timesfm


st.title("Foundational Models para Series de Tiempo", anchor=None, help=None)
st.header("Creado por: Sebastian Sarasti", divider="gray")
st.markdown("More about me: [LinkedIn](https://www.linkedin.com/in/sebastiansarasti/)")
st.markdown("**Conferencia:** Universidad de Palermo")

st.markdown("Esta aplicación permite explorar modelos de series de tiempo utilizando modelos fundacionales basados en Transformers. El modelo fundacional seleccionado es TimesFM, fue desarrollado por Google. ")

st.markdown("**¿Cómo funciona?**")

st.markdown("1. **Selecciona las fechas**: Elige el rango de fechas para el cual deseas predecir los precios de Bitcoin.")
st.markdown("2. **Selecciona la ventana de forecast**: Permite configurar el modelo para predecir el horizonte de tiempo deseado.")
st.markdown("3. **Ejecuta el modelo**: Haz clic en el botón para ejecutar el modelo y obtener las predicciones.")

# create two columns for start date and end date
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Fecha de Inicio", value="2025-01-31")

with col2:
    end_date = st.date_input("Fecha de Fin", value="2025-06-10")

# create a slider for forecast horizon
forecast_horizon = st.slider(
    "Ventana de Forecast",
    min_value=1,
    max_value=365,
    value=st.session_state.get("forecast_horizon", 30),
    help="Selecciona el horizonte de tiempo para las predicciones (en días)."
)

# create a button to run the model
value = st.button("Ejecutar Modelo")

# ... después del botón "Ejecutar Modelo"
if value:
    assert start_date < end_date, "La fecha de inicio debe ser anterior a la fecha de fin."
    assert forecast_horizon > 0, "La ventana de forecast debe ser mayor a 0."

    with st.spinner("Descargando datos ..."):
        df = get_bitcoin_history_yf(start_date, end_date)
        df = process_dataframe(df)
        st.session_state["df"] = df

    with st.spinner("Ejecutando modelo ..."):
        model = get_model(forecast_horizon)
        forecast_df = predict_timesfm(df=df, model=model)
        forecast_df["type"] = "Forecast"
        st.session_state["forecast_df"] = forecast_df

# nuevo botón separado
if "forecast_df" in st.session_state and st.button("Graficar Predicciones"):
    df = st.session_state["df"]
    forecast_df = st.session_state["forecast_df"]
    df["type"] = "Historia"
    df_final = pd.concat([df, forecast_df], ignore_index=True)
    fig = px.line(
        df_final,
        x="ds",
        y="y",
        color="type",
        title="Predicciones de Bitcoin con TimesFM",
        labels={"ds": "Fecha", "y": "Precio (USD)", "type": "Tipo"},
    )
    st.plotly_chart(fig)
