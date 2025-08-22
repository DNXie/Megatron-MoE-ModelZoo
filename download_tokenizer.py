from transformers import AutoTokenizer

# Correct model/tokenizer name
tokenizer_name = "deepseek-ai/DeepSeek-V2-Lite"

# Local directory to save the tokenizer
local_path = "/home/dxie/models/deepseek-v2/tokenizer"

# Download and save locally
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.save_pretrained(local_path)

print(f"Tokenizer saved to {local_path}")
