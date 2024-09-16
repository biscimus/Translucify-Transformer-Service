#!/bin/bash

# Path to your virtual environment's activate script
VENV_PATH="venv/bin/activate"

# Create a new tmux session and run the services in different panes
tmux new-session -d -s mysession

# Start Redis in the first tmux window
tmux send-keys "source $VENV_PATH && redis-server --port 7777" C-m

# Split the window and start Celery in the new pane
tmux split-window -h
tmux send-keys "source $VENV_PATH && celery -A app.celery worker --loglevel INFO" C-m

# Split the window again and start Flask in the new pane
tmux split-window -v
tmux send-keys "source $VENV_PATH && flask run --debug" C-m

# Attach to the tmux session to monitor the processes
tmux attach-session -t mysession
