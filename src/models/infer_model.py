import torch

def predict_model(text_list, model, tokenizer):
    inputs = tokenizer(
        text_list,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(model.device)

    outputs = model(**inputs).logits
    print(outputs)
    predictions = torch.argmax(outputs, dim=-1)[0].cpu().numpy()
    return list(predictions[1:-1])