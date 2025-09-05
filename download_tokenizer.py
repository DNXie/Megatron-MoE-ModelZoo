import argparse

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Download DeepSeek tokenizer")
    parser.add_argument(
        "--dsv2", action="store_true", help="Use DeepSeek-V2-Lite tokenizer"
    )
    parser.add_argument("--dsv3", action="store_true", help="Use DeepSeek-V3 tokenizer")

    args = parser.parse_args()

    # Default to dsv2 if no flag is specified
    if args.dsv3:
        tokenizer_name = "deepseek-ai/DeepSeek-V3"
        local_path = "/home/ubuntu/models/deepseek-v3/tokenizer"
    else:  # Default to dsv2 (including when --dsv2 is explicitly passed)
        tokenizer_name = "deepseek-ai/DeepSeek-V2-Lite"
        local_path = "/home/ubuntu/models/deepseek-v2/tokenizer"

    print(f"Using tokenizer: {tokenizer_name}")
    print(f"Saving to: {local_path}")

    # Download and save locally
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(local_path)

    print(f"Tokenizer saved to {local_path}")


if __name__ == "__main__":
    main()
