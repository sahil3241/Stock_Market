# Use an official Python runtime as a parent image
FROM python:3.10

# Make port 5002 available to the world outside this container
ENV PORT=5000
EXPOSE 5000

# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /gui
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements-gui.txt



# Define environment variable
ENV FLASK_APP=gui.py

# Run gui.py when the container launches
CMD ["flask", "run", "--host", "0.0.0.0"]