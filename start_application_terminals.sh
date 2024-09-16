#!/bin/bash

# Function to start the Redis server on port 7777
start_redis() {
    echo "Starting Redis server on port 7777..."
    redis-server --port 7777
}

# Function to start the Celery worker
start_celery() {
    echo "Starting Celery worker..."
    celery -A app.celery worker --loglevel INFO
}

# Function to start the Flask application
start_flask() {
    echo "Starting Flask application..."
    flask run --debug
}

# Run the services in separate terminals
gnome-terminal -- bash -c "start_redis; exec bash"
gnome-terminal -- bash -c "start_celery; exec bash"
gnome-terminal -- bash -c "start_flask; exec bash"

# Keep the script running
wait
