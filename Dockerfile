# Use an explicit version of the python base image for reproducibility
FROM python:3.9.17

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies and necessary system packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your application's code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Use an environment variable if needed
ENV NAME World

# Define the command to run your app
CMD ["python", "pixelshop.py"]
