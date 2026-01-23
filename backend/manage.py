"""
Paraphrase Detection API - Management CLI

Simple command-line interface for managing the backend.

Usage:
    python manage.py serve          - Start API server
    python manage.py train          - Train the model
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def serve():
    """Start the API server"""
    from backend.cli.serve import main
    main()


def train():
    """Train the model"""
    from backend.cli.train import main
    main()


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase Detection API - Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage.py serve          Start the API server
  python manage.py train          Train the model
        """
    )
    
    parser.add_argument(
        "command",
        choices=["serve", "train"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    commands = {
        "serve": serve,
        "train": train
    }
    
    commands[args.command]()


if __name__ == "__main__":
    main()
