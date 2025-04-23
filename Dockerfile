# Use official Python runtime
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the repo to the container's /app
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Expose the port that Render expects (default is 10000, but EXPOSE is just documentation)
EXPOSE 10000

# Run FastAPI with Uvicorn, binding to 0.0.0.0 and the Render-provided port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]

