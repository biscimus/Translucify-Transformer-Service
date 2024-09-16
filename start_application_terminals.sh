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

# Create a new tmux session and run the services in different panes
tmux new-session -d -s mysession

# Start Redis in the first tmux window
tmux send-keys "start_redis" C-m

# Split the window and start Celery in the new pane
tmux split-window -h
tmux send-keys "start_celery" C-m

# Split the window again and start Flask in the new pane
tmux split-window -v
tmux send-keys "start_flask" C-m

# Attach to the tmux session to monitor the processes
tmux attach-session -t mysession
