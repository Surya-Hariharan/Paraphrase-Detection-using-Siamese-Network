"""Initialize database with admin user"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.db.session import init_db, get_db_context
from backend.services.user_service import UserService


def main():
    print("Initializing database...")
    
    # Create tables
    init_db()
    print("✓ Tables created")
    
    # Create admin user
    with get_db_context() as db:
        # Check if admin exists
        admin = UserService.get_user_by_username(db, "admin")
        
        if not admin:
            admin = UserService.create_user(
                db=db,
                email="admin@example.com",
                username="admin",
                password="admin123",  # Change in production!
                is_admin=True
            )
            print("✓ Admin user created")
            print("  Username: admin")
            print("  Password: admin123")
            print("  ⚠️  CHANGE PASSWORD IN PRODUCTION!")
        else:
            print("✓ Admin user already exists")
    
    print("\n✅ Database setup complete")


if __name__ == "__main__":
    main()
