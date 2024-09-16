#!/bin/bash

# Start the Redis server on port 7777
echo "Starting Redis server on port 7777..."
redis-server --port 7777 &
REDIS_PID=$!
echo "Redis server started with PID $REDIS_PID"

# Wait a few seconds to ensure Redis is up and running
sleep 3

# Start the Celery worker
echo "Starting Celery worker..."
celery -A app.celery worker --loglevel INFO &
CELERY_PID=$!
echo "Celery worker started with PID $CELERY_PID"

# Wait a few seconds to ensure Celery is up and running
sleep 5

# Start the Flask application
echo "Starting Flask application..."
flask run --debug &
FLASK_PID=$!
echo "Flask application started with PID $FLASK_PID"

# Function to stop all processes on script exit
function stop_services {
  echo "Stopping services..."
  kill $FLASK_PID
  kill $CELERY_PID
  kill $REDIS_PID
  echo "Services stopped."
}

# Trap exit signals to stop all services
trap stop_services EXIT

# Keep the script running to allow the processes to stay active
wait
