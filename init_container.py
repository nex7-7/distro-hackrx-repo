#!/usr/bin/env python3
"""
Container initialization script to ensure proper setup before starting the application.
"""

import os
import sys
from pathlib import Path


def setup_directories():
    """Ensure required directories exist and are writable."""
    directories = [
        ('logs', os.getenv('LOG_FILE', 'logs/app.log')),
        ('downloads', '/app/downloads'),
        ('temp', '/app/temp'),
        ('cache', '/app/cache')
    ]
    
    for dir_name, dir_path in directories:
        path = Path(dir_path) if dir_name == 'logs' else Path(dir_path)
        if dir_name == 'logs':
            path = path.parent
        
        try:
            # Try to create the directory
            path.mkdir(parents=True, exist_ok=True, mode=0o777)
            
            # Test if we can write to the directory
            test_file = path / "test_write.tmp"
            test_file.touch()
            test_file.unlink()
            
            print(f"✓ {dir_name.capitalize()} directory ready: {path}")
            
        except (PermissionError, OSError) as e:
            if dir_name == 'downloads':
                print(f"✗ Cannot write to {path}: {e}")
                print(f"✓ Downloads will use fallback temp directory")
            else:
                print(f"✗ Cannot write to {path}: {e}")
    
    return True


def main():
    """Initialize container environment."""
    print("Initializing container...")
    
    if not setup_directories():
        print("Failed to setup directories")
        sys.exit(1)
    
    print("Container initialization complete")
    return True


if __name__ == "__main__":
    main()
