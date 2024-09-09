import os
from flask import Flask, jsonify, request
import pandas as pd
from simple_transformer import translucify_with_transformer

app = Flask(__name__)

# For the POST request, the microservice gets an event log. Then, it creates a model instance using the event log, whose parametes are saved in the ./models directory. In subsequent requests, the microservice uses the model instance to make predictions.
# For the GET request, the service returns the translucent event log.
@app.route("/", methods=["GET", "POST"])
def transformer():
    if request.method == "POST":
        file = request.files.get("file")
        id, threshold = request.form.get("id"), request.form.get("threshold")

        # Save the event log to the ./event-logs directory
        os.mkdir("./event-logs", exist_ok=True)
        file.save(f"./event-logs/{id}.csv")

        # convert file to dataframe
        log = pd.read_csv(f"./event-logs/{id}.csv", delimiter=";")

        # Create a transformer model instance using the event log
        return jsonify(translucify_with_transformer(id, log, threshold))





        