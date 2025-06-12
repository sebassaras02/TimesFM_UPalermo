FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt requirements.txt
COPY src/ ./src

# Install uv and Python packages
RUN pip install uv
RUN uv pip install --system -r requirements.txt

# Create non-root user and give permissions
RUN useradd -m appuser && \
    mkdir -p /app/cache /app/.streamlit && \
    chown -R appuser:appuser /app

# Set environment variables for Hugging Face and Streamlit
ENV HF_HOME=/app/cache
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
