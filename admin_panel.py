import os
# ---------------------------
# Force PyTorch-only, avoid TF/Keras issues
# ---------------------------
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"
os.environ["DISABLE_TRANSFORMERS_TF_IMPORT"] = "1"

import sys
sys.modules["tensorflow"] = None
sys.modules["tf_keras"] = None

# ---------------------------
# Imports
# ---------------------------
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import gradio as gr

# ---------------------------
# Paths
# ---------------------------
dataset_path = ".gradio/flagged/dataset3.csv"
feedback_path = ".gradio/flagged/feedback.csv"
model_path = "distilbert_airline"

# ---------------------------
# Load flagged dataset
# ---------------------------
def load_flagged():
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path, dtype=str)
    else:
        df = pd.DataFrame(columns=["text","output","timestamp"])
    if "crted" not in df.columns:
        df["crted"] = ""
    df = df.fillna("")
    return df[["text","output","crted","timestamp"]]

# ---------------------------
# Save corrections to feedback.csv and remove from dataset3.csv
# ---------------------------
def save_feedback(df):
    df_to_save = df[df["crted"].notna() & (df["crted"] != "")]
    if not df_to_save.empty:
        feedback_df = df_to_save[["text","crted","timestamp"]].rename(
            columns={"crted":"correct_label","timestamp":"timestamp"}
        )
        if os.path.exists(feedback_path):
            existing_feedback = pd.read_csv(feedback_path, dtype=str)
            feedback_df = pd.concat([existing_feedback, feedback_df], ignore_index=True)
            feedback_df = feedback_df.drop_duplicates(subset="text", keep="last")
        feedback_df.to_csv(feedback_path, index=False)

        # Remove saved rows from dataset3.csv
        dataset_df = pd.read_csv(dataset_path, dtype=str)
        dataset_df = dataset_df[~dataset_df["text"].isin(df_to_save["text"])]
        dataset_df.to_csv(dataset_path, index=False)

    # Check for retrain only if threshold reached
    auto_retrain()
    
    return load_flagged()


# ---------------------------
# Retrain model using feedback.csv if ≥10 rows
# ---------------------------
def auto_retrain():
    import os
    import pandas as pd

    feedback_path = ".gradio/flagged/feedback.csv"
    model_path = "distilbert_airline"
    original_dataset_path = "airline_customer_requests_31k_28classes.csv"

    if os.path.exists(feedback_path):
        feedback_df = pd.read_csv(feedback_path, dtype=str)
        print("Feedback rows:", len(feedback_df))
        if len(feedback_df) >= 2:  # threshold for retraining
            print("Retraining model on feedback.csv...")

            # ------------------------------
            # 1. Load a subset of original dataset to avoid forgetting
            # ------------------------------
            if os.path.exists(original_dataset_path):
                original_df = pd.read_csv(original_dataset_path, dtype=str)
            else:
                original_df = pd.DataFrame(columns=["utterance","intent"])

            # Rename columns to match retraining function
            if "utterance" in original_df.columns and "intent" in original_df.columns:
                original_df = original_df.rename(columns={"utterance": "text", "intent": "correct_label"})

            # Take a sample of original data (e.g., 500 rows) to mix with feedback
            if len(original_df) > 500:
                original_sample = original_df.sample(500, random_state=42)
            else:
                original_sample = original_df

            # Prepare feedback
            feedback_df = feedback_df.rename(columns={"crted": "correct_label"})
            if "text" not in feedback_df.columns:
                feedback_df = feedback_df.rename(columns={"utterance": "text"})

            # Combine original sample and feedback
            combined_df = pd.concat([original_sample, feedback_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset="text", keep="last")

            print("Combined dataset size for retraining:", len(combined_df))

            # ------------------------------
            # 2. Call retraining function
            # ------------------------------
            retrain_model_from_feedback(combined_df)

            # ------------------------------
            # 3. Reset feedback.csv
            # ------------------------------
            pd.DataFrame(columns=["text","correct_label","timestamp"]).to_csv(feedback_path, index=False)
            print("Retraining completed. Feedback reset.")



# ---------------------------
# Retraining pipeline
# ---------------------------
def retrain_model_from_feedback(df):
    # Map labels to integers
    labels = df["correct_label"].unique().tolist()
    label2id = {label:i for i,label in enumerate(labels)}
    id2label = {i:label for label,i in label2id.items()}
    df["label"] = df["correct_label"].map(label2id)

    # Load tokenizer & model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    num_labels=len(labels),
    ignore_mismatched_sizes=True 
)

    model.config.id2label = id2label
    model.config.label2id = label2id

    # Tokenize
    encodings = tokenizer(list(df["text"]), truncation=True, padding=True, max_length=128)

    # Dataset class
    class FeedbackDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key,val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    dataset = FeedbackDataset(encodings, df["label"].tolist())

    # Training
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        logging_steps=5,
        save_strategy="no",
        overwrite_output_dir=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print("Retraining finished. Model updated!")

# ---------------------------
# Admin UI
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Admin Panel: Correct Predicted Labels")
    
    table = gr.Dataframe(
        value=load_flagged(),
        headers=["text","output","crted","timestamp"],
        datatype=["str","str","str","str"],
        interactive=True,
        row_count=(1, "dynamic")  # allow deletion
    )

    save_btn = gr.Button("Save Corrections")
    save_btn.click(fn=save_feedback, inputs=[table], outputs=[table])

    refresh_btn = gr.Button("Refresh Table")
    refresh_btn.click(fn=load_flagged, inputs=[], outputs=[table])

# Launch
demo.launch(share=True)
