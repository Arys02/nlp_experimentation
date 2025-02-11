from transformers import pipeline, DistilBertTokenizer, AutoModelForTokenClassification
import torch

# pipe = pipeline("text-classification", model="Arys02/fine_tuned_tp8")


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained("Arys02/fine_tuned_tp8")

ner_model = pipeline('ner', model='foucheta/nlp_esgi_td4_ner', grouped_entities=True)

text = "Ask the python teacher when is the next class"

m = model
ner_extract = ner_model(text)

def extract_entities(text):
    ner_entities = ner_model(text)
    receiver = ""
    content = ""

    for entity in ner_entities:
        if entity['entity_group'] == "LABEL_1":
            receiver = entity["word"]
        if entity['entity_group'] == "LABEL_2":
            content = entity["word"]
    return receiver, content

def call_virtual_assistant(user_query: str) -> dict:
    inputs = tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    label = model.config.id2label[predicted_class_idx]

    if label == "LABEL_1":
        return {
            "task": "ask_RAG",
            "reply": f"asked_to_rag: {user_query}",
        }
    elif label == "LABEL_2":
        receiver, content = extract_entities(user_query)
        return {
            "task": "send_message",
            "receiver": receiver,
            "content": content,
        }
    else:
        return {
            "task": "unknown",
            "reply": "Je ne comprends pas votre requête.",
        }

# Exemple d'utilisation
if __name__ == "__main__":
    print(call_virtual_assistant("Does the React course cover the use of hooks?"))
    print(call_virtual_assistant("Ask the python teacher when is the next class"))

# classif = pipe(ner_extract)
# def call_virtual_assistant(user_query: str) -> dict:
#     classif = classifier(user_query)
#
#     label =