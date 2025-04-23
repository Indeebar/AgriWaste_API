# Use official Python runtime
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the repo to the container's /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose port (optional if using uvicorn to run FastAPI)
EXPOSE 8000

# Run your FastAPI or Flask app
CMD ["python", "main.py"]
