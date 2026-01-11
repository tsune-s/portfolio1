"""Main entry point for the portfolio project."""

import argparse
import sys
from pathlib import Path


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Data Analysis Portfolio - Main Entry Point"
    )

    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Generate sample datasets'
    )

    parser.add_argument(
        '--train-model',
        action='store_true',
        help='Train customer churn prediction model'
    )

    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Launch Streamlit dashboard'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test suite'
    )

    args = parser.parse_args()

    if args.generate_data:
        print("Generating sample datasets...")
        import subprocess
        subprocess.run([sys.executable, "scripts/generate_sample_data.py"])

    elif args.train_model:
        print("Training customer churn model...")
        import subprocess
        subprocess.run([sys.executable, "scripts/train_churn_model.py"])

    elif args.dashboard:
        print("Launching Streamlit dashboard...")
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboards/app.py"])

    elif args.test:
        print("Running tests...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])

    else:
        parser.print_help()
        print("\n" + "="*70)
        print("Welcome to the Data Analysis Portfolio!")
        print("="*70)
        print("\nQuick Start:")
        print("  1. Generate sample data:  python main.py --generate-data")
        print("  2. Train models:          python main.py --train-model")
        print("  3. Launch dashboard:      python main.py --dashboard")
        print("  4. Run tests:             python main.py --test")
        print("\nOr use individual scripts:")
        print("  - uv run python scripts/generate_sample_data.py")
        print("  - uv run python scripts/train_churn_model.py")
        print("  - uv run streamlit run dashboards/app.py")
        print("="*70)


if __name__ == "__main__":
    main()
