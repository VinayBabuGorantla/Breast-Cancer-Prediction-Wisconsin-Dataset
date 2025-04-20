# Use the official Python image as base
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirement files first to leverage Docker cache
COPY requirements.txt requirements.txt
COPY setup.py setup.py

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all project files into the container
COPY . .

# Run Flask web app
CMD ["python", "app.py"]
