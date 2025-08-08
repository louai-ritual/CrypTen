import os
import argparse
import crypten
from transformers import AutoModelForCausalLM, AutoTokenizer
from crypten.models.utils import generate, load_model

def parse_args():
    parser = argparse.ArgumentParser(description="Priveri Pipeline")

    parser.add_argument(
        "--model", type=str, required=True, help="Path for the model to run priveri on"
    )

    parser.add_argument(
        "--num_parties",
        type=int,
        default=1,
        help="Number of parties (sets CrypTen world size)",
    )

    parser.add_argument("--rank", type=int, default=0, help="Rank of party in SMPC")

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="The maximum number of tokens that the model will generate.",
    )

    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run the model on (cpu or cuda)"
    )

    parser.add_argument(
        "--prompt", type=str, default="Hello, World!",
        help="Prompt to run inference on."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.num_parties)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RENDEZVOUS"] = "env://"
    crypten.init()

    model, tokenizer = load_model(args.model)
    model = model.eval()

    if args.device == "cuda":
        model = model.cuda()

    response = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.device)
    print(response)

if __name__ == "__main__":
    main()