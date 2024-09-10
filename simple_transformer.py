import os
from torch import nn
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import evaluate

ACTIVITY_COLUMN = "concept:name"
CASE_COLUMN = "case:concept:name"
NEXT_ACTIVITY_COLUMN = "next_activity"
ENABLED_ACTIVITIES_COLUMN = "enabled_activities"

def translucify_with_transformer(id: int, log: pd.DataFrame, threshold: float, data_columns: list[str] = None) -> pd.DataFrame:
        
    # Get list of all activities
    labels = log[ACTIVITY_COLUMN].unique()

    # Append end activity to list
    labels = pd.Series(np.append(labels, ["end"]), dtype="string")

    print("Unique labels:\n", labels)

    # One to one map activities to integers
    le = LabelEncoder().fit(labels)

    # Add next activity column to the DataFrame and fill it
    log[NEXT_ACTIVITY_COLUMN] = None
    def fill_next_activity_column(group: pd.Series) -> pd.DataFrame:
        previous_index = None
        for index, row in group.iterrows():
            if previous_index is not None:
                group.at[previous_index, NEXT_ACTIVITY_COLUMN] = row[ACTIVITY_COLUMN]
            previous_index = index
        return group
    log = log.groupby(CASE_COLUMN, group_keys=False).apply(fill_next_activity_column).reset_index()
    # Fill None values with the number of unique labels as end activity
    log[NEXT_ACTIVITY_COLUMN] = log[NEXT_ACTIVITY_COLUMN].fillna("end")
    log[NEXT_ACTIVITY_COLUMN] = le.transform(log[NEXT_ACTIVITY_COLUMN])

    print("Log after next activity column gen: \n", log)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    is_gpu_available = torch.cuda.is_available()
    print(f"Is GPU available: {is_gpu_available}")
        
    device = torch.device("cuda" if is_gpu_available else "cpu")
    
    # check if model is already trained and saved in ./models directory
    if not os.path.exists(f"./models/{id}"):
        '''
        TODO: Currently we're simply concatenating all inputs into a single string. We need to tokenize the input in a more meaningful way
        The original ProcessTranformer paper uses only the activity prefixes as features. 
        They map each activity to a a unique integer and make a list of integers for each prefix trace. 
        They then just use that as input for the transformer. 
        We cant do this since we are also considering other user selected data attributes as feature space :)
        '''
        labels_list: list[str] = []
        inputs_list: list[str] = []

        def generate_instances_per_case(group: pd.Series) -> pd.DataFrame:
            input_prefix: str = ""
            for index, row in group.iterrows():
                input = row.values.tolist()
                input = ', '.join(map(str, input))
                input_prefix += input
                inputs_list.append(input_prefix)
                labels_list.append(row[NEXT_ACTIVITY_COLUMN])
                # Add separator to input
                input_prefix += "; "
            return group

        log.groupby(CASE_COLUMN, group_keys=False).apply(generate_instances_per_case).reset_index()

        print("LOG after creating instances: \n", log)

        # print("Labels: \n", labels_list)
        # print("Inputs: \n", inputs_list)

        train_inputs, test_inputs, train_labels, test_labels = train_test_split(
            inputs_list,
            labels_list,
            test_size=0.2,
        )

        
        train_encodings = tokenizer(train_inputs, padding="max_length", truncation=True, max_length=512)
        test_encodings = tokenizer(test_inputs, padding="max_length", truncation=True, max_length=512)

        # print(f"Train encodings: {train_encodings}")
        # print(f"Train labels: {train_labels}")
        # print(f"Test encodings: {test_encodings}")
        # print(f"Test labels: {test_labels}")

        class TextClassifierDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        train_dataset = TextClassifierDataset(train_encodings, train_labels)
        eval_dataset = TextClassifierDataset(test_encodings, test_labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            problem_type="multi_label_classification",
            num_labels=labels.size
        )


        training_arguments = TrainingArguments(
            output_dir="checkpoints",
            eval_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=50,
            use_cpu=not is_gpu_available,
            # learning_rate=1e-5
        )

        class MyTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)    

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs[0]
                loss = nn.CrossEntropyLoss()
                return (loss(logits, labels), outputs) if return_outputs else loss(logits, labels)

        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")


        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
            precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
            recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
            f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
            return {
                "accuracy": accuracy["accuracy"],
                "precision": precision["precision"],
                "recall": recall["recall"],
                "f1": f1["f1"]
            }


        trainer = MyTrainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

        eval_result = trainer.evaluate()
        # turn eval_result to dataframe and write the results to csv file
        eval_result_df = pd.DataFrame([eval_result])
        os.makedirs("./logs", exist_ok=True)
        eval_result_df.to_csv("./logs/transformer_eval_result.csv", index=False)

        # Save the model
        os.makedirs("./models", exist_ok=True)
        torch.save(model.state_dict(), f"./models/{id}")

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            problem_type="multi_label_classification",
            num_labels=labels.size
        )

    # If you have custom weights to load (optional)
    custom_state_dict = torch.load(f"./models/{id}", weights_only=True)
    model.load_state_dict(custom_state_dict)

    model = model.to(device)

    log[ENABLED_ACTIVITIES_COLUMN] = None

    def fill_enabled_activities_column(group: pd.Series) -> pd.DataFrame:

        enabled_activities = None

        print("Group: \n", group)

        for index, row in group.iterrows():
            # Generate input instance
            input = row.values.tolist()
            input = ', '.join(map(str, input))
            inputs = tokenizer(input, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            probabilities = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
            print(probabilities)

            # Convert probabilities to numpy array for further processing
            probabilities = probabilities.numpy()
            print("probs: ", probabilities)
            high_prob_indices = (probabilities > threshold).nonzero(as_tuple=True)[1]
            string_labels = le.inverse_transform(high_prob_indices)
            print("String labels", string_labels)
            #TODO: Add artificial start activity for training
            if enabled_activities is not None:
                group.at[index, ENABLED_ACTIVITIES_COLUMN] = sorted(enabled_activities, key=str.lower)
            enabled_activities = string_labels

        print("Group after enabled activities: \n", group)
        return group

    log = log.groupby(CASE_COLUMN, group_keys=False).apply(fill_enabled_activities_column).reset_index()
    return log

   

    

# Call e.g.: python simple_transformer.py 0.5
# if __name__ == "__main__":
#     log = import_csv('../logs/helpdesk.csv', separator=",")

#     parser = argparse.ArgumentParser("simple_transformer")
#     parser.add_argument("threshold", help="The cutoff percentage.", type=float)
#     args = parser.parse_args()
#     log = translucify_with_transformer(log, args.threshold)
#     print(log)
#     log.to_csv("../logs/helpdesk_log_translucified.csv", index=False)

# # Generate all prefixes of each trace and the corresponding next activity
# for trace in traces:
#     for i in range(1, len(trace)):
#         X.append(trace[:i])
#         y.append(trace[i])

# # Split the log into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize model and set up training essentials
# n_activities = log['Activity_encoded'].nunique()
# model = TracePredictor(n_activities)
# optimizer = Adam(model.parameters(), lr=1e-5)
# criterion = nn.CrossEntropyLoss()

# # Instantiate DataLoader for handling batches of data
# train_dataset = TraceDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Training loop
# for epoch in range(10):
#     model.train()
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# class TraceDataset(Dataset):
#     def __init__(self, X, y, max_len=50):
#         self.X = X
#         self.y = y
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         # Pad sequences so that all have the same length
#         x = self.X[idx]
#         x_padded = x + [0] * (self.max_len - len(x))  # Padding with zeros
#         return torch.tensor(x_padded, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

# class TracePredictor(nn.Module):
#     def __init__(self, n_activities):
#         super().__init__()
#         config = BertConfig.from_pretrained('distilbert-base-uncased', num_labels=n_activities)
#         self.bert = BertModel(config)
#         self.classifier = nn.Linear(config.hidden_size, n_activities)

#     def forward(self, x):
#         # Pass input through BERT model to get contextualized embeddings
#         outputs = self.bert(input_ids=x)
#         # Use the representation of the last token for predicting the next activity
#         sequence_output = outputs.last_hidden_state[:, -1, :]
#         # Classifier to predict next activity from the processed token
#         logits = self.classifier(sequence_output)
#         return logits

# Load your dataset
