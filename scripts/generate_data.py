"""
CLI script to generate synthetic data.
Usage: python scripts/generate_data.py [--seed 42]
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation.generator import generate_all_data


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for Erie MCA demo")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to synthetic_data.yaml")
    args = parser.parse_args()

    print("Erie MCA -- Synthetic Data Generation")
    print("=" * 50)

    data = generate_all_data(config_path=args.config, seed=args.seed)

    print(f"\nData Summary:")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows x {len(df.columns)} columns")

    print("\nDone! Data saved to data/raw/")


if __name__ == "__main__":
    main()
