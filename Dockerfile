# --- Build stage ---
FROM python:3.10 AS build

# Set the working directory in the container
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the C source files
COPY ./c_src/ElpMpp02 /build/ElpMpp02

# Build the Python C API extension
WORKDIR /build/ElpMpp02
RUN pip install .

# --- Production stage ---
FROM python:3.10

# Create a new user
RUN adduser --disabled-password --gecos '' myuser

# Set the working directory in the container
WORKDIR /app

# Copy the compiled C binaries from the build stage
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Copy the application files
COPY ./app /app/app

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry gunicorn && \
    poetry config virtualenvs.create false && \
    poetry install --only main

# Switch to the new user
USER myuser

# Make port 8000 available to the world outside this container
EXPOSE 8000

WORKDIR /app/app

# Run the application with Gunicorn with Uvicorn workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]