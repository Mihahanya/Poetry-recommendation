from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "sberbank-ai/rugpt2large"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

input_text = """покажи мне стихи об одиночестве эпохи моедрна"""

input_ids = tokenizer.encode(input_text, return_tensors="pt")

print(len(input_ids))

output = model.generate(
    input_ids,
    max_length=100,
    temperature=1.0,
    do_sample=True,
#    pad_token_id=tokenizer.eos_token_id,  # Используйте pad_token_id вместо eos_token_id для ускорения генерации
#    early_stopping=True  # Останавливать генерацию после первого завершающего токена
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
