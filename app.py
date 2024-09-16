import os
from celery import Celery, Task, shared_task
from flask import Flask, jsonify, request
import pandas as pd
from simple_transformer import translucify_with_transformer
import requests

# Celery configuration
def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app

app = Flask(__name__)

# Add Celery
app.config.from_mapping(
    CELERY=dict(
        broker_url="redis://localhost:7777/0",
        result_backend="redis://localhost:7777/0",
        task_ignore_result=True,
        broker_connection_retry_on_startup=True,
    ),
)

# Use command "celery -A app.celery worker --loglevel INFO" to start the worker
celery = celery_init_app(app)

# For the POST request, the microservice gets an event log. Then, it creates a model instance using the event log, whose parametes are saved in the ./models directory. In subsequent requests, the microservice uses the model instance to make predictions.
# For the GET request, the service returns the translucent event log.
@app.route("/", methods=["GET", "POST"])
def transformer():
    if request.method == "POST":
        file = request.files.get("file")
        id, threshold = request.form.get("id"), float(request.form.get("threshold"))

        # Save the event log to the ./event-logs directory
        os.makedirs("./event-logs", exist_ok=True)

        file_path = os.path.join("./event-logs", f"{id}.csv")
        file.save(file_path)

        # convert file to dataframe
        process_translucent_log_with_transformer.delay(id, threshold)

        return "Gotcha!"
    
@shared_task
def process_translucent_log_with_transformer(id, threshold):
    df = translucify_with_transformer(id, threshold)

    # Post the translucent event log back to the main application
    requests.post(f"http://localhost:3000/{id}/callback", files={"file": df.to_csv()}, data={"id": id})