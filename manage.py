"""
Paraphrase Detection API - Management CLI

Professional command-line interface for managing the backend.

Usage:
    python manage.py serve          - Start API server
    python manage.py train          - Train the model
    python manage.py init-db        - Initialize database
    python manage.py create-admin   - Create admin user
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def serve():
    """Start the API server"""
    from backend.cli.serve import main
    main()


def train():
    """Train the model"""
    from backend.cli.train import main
    main()


def init_db():
    """Initialize the database"""
    from backend.cli.initialize import main
    main()


def create_admin():
    """Create admin user"""
    from backend.db.session import init_db, get_db_context
    from backend.services.user_service import UserService
    import getpass
    
    print("Creating admin user...")
    init_db()
    
    username = input("Username [admin]: ").strip() or "admin"
    email = input("Email [admin@example.com]: ").strip() or "admin@example.com"
    password = getpass.getpass("Password: ")
    
    if not password:
        print("❌ Password cannot be empty")
        return
    
    with get_db_context() as db:
        existing = UserService.get_user_by_username(db, username)
        if existing:
            print(f"❌ User '{username}' already exists")
            return
        
        user = UserService.create_user(
            db=db,
            email=email,
            username=username,
            password=password,
            is_admin=True
        )
        print(f"✅ Admin user '{username}' created successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase Detection API - Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage.py serve          Start the API server
  python manage.py train          Train the model
  python manage.py init-db        Initialize database
  python manage.py create-admin   Create admin user
        """
    )
    
    parser.add_argument(
        "command",
        choices=["serve", "train", "init-db", "create-admin"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    commands = {
        "serve": serve,
        "train": train,
        "init-db": init_db,
        "create-admin": create_admin
    }
    
    commands[args.command]()


if __name__ == "__main__":
    main()
