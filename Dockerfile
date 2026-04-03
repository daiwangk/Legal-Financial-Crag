# Use Python 3.11 as the base image (the version we know works flawlessly with the pydantic shim)
FROM python:3.11-slim

# Create a non-root user that Hugging Face Spaces expects
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user:user requirements.txt .

# Install dependencies (ignoring the redundant llama-index-core in the file)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY --chown=user:user . .

# Expose the port Hugging Face Spaces expects for the main app
EXPOSE 7860

# We need to make the startup script executable
RUN chmod +x start.sh

# Run the startup script which kicks off both FastAPI and Streamlit
CMD ["./start.sh"]
