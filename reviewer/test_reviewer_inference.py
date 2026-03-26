import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load model and tokenizer
MODEL_PATH = "C:\\Users\\daisl\\PycharmProjects\\HeRoN\\model_directory"
print(f"Loading model from {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)


def generate_response(input_text, max_length=128):
    tokenized = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    query_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            query_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=1,
            temperature=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    print("=" * 60)
    print("quit per uscire")
    print("=" * 60 + "\n")

    while True:
        user_input = input(">>> ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            print("---\n")
            continue

        print("\nGenero...")
        response = generate_response(user_input)
        print(f"\RISPOSTA : {response}\n")
