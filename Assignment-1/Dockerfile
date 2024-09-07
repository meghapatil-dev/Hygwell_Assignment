# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into the container
COPY api.py .

# Copy the folder into the container
COPY all-MiniLM-L6-v2 all-MiniLM-L6-v2

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]