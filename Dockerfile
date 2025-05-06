# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirement files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (Spaces expects 7860 or 5000)
EXPOSE 7860

# Set environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=7860
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask app
CMD ["flask", "run"]
