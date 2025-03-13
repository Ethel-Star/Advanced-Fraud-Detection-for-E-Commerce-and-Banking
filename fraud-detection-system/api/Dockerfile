# Use official Python runtime as base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy requirements (if any, or handle dependencies in serve_model.py)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "serve_model.py"]