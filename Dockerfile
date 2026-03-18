FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hw_visualisation.py .
COPY backend_viz.py .
COPY circuit_viz.py .

# Use non-interactive matplotlib backend (no display needed)
ENV MPLBACKEND=Agg

CMD ["python", "hw_visualisation.py", "backend_viz.py", "circuit_viz.py"]
