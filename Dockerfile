# --- Build stage for C extensions ---
FROM python:3.10-slim as builder

# Set work directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy C source files
COPY ./c_src/elp2000_82b /build/c_src/elp2000_82b

# Compile C code
WORKDIR /build/c_src/elp2000_82b
RUN make

# --- Production stage ---
FROM python:3.10-slim

# Create a new user
RUN adduser --disabled-password --gecos '' myuser

# Set the working directory in the container
WORKDIR /app

# Install minimal runtime dependencies if necessary
RUN apt-get update && apt-get install -y \
    libffi6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled C binaries and other necessary files
COPY --from=builder /build/c_src/elp2000_82b /app/c_src/elp2000_82b
COPY ./app /app/app

# Install Python dependencies
COPY pyproject.toml poetry.lock /app/app/
RUN pip install poetry gunicorn && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Switch to the new user
USER myuser

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application with Gunicorn with Uvicorn workers
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]
    