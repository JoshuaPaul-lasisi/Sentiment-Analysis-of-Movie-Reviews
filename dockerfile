# Base image
FROM python:3.8-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "app.py"]