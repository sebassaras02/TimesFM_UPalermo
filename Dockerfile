FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt requirements.txt
COPY src/ ./src

RUN pip install uv
RUN uv pip install --system -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
