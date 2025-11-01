#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils.generate_description import generate_path_description


def main():
    # Usa la cartella "output" nella directory corrente
    input_dir = Path(os.getcwd()) / "output"
    print(f"Reading frames from: {input_dir}")

    try:
        description = generate_path_description(input_dir)
        print("\n--- GENERATED DESCRIPTION ---\n")
        print(description)
    except Exception as e:
        print(f"Failed to generate description: {e}")


if __name__ == "__main__":
    main()
