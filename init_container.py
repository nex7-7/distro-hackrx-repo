#!/usr/bin/env python3
"""
Container initialization script to ensure proper setup before starting the application.
"""

import os
import sys
from pathlib import Path


def setup_logging_directory():
    """Ensure logging directory exists and is writable."""
    log_file = os.getenv('LOG_FILE', 'logs/app.log')
    log_path = Path(log_file)
    
    try:
        # Try to create the directory
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)
        
        # Test if we can write to the directory
        test_file = log_path.parent / "test_write.tmp"
        test_file.touch()
        test_file.unlink()
        
        print(f"✓ Logging directory ready: {log_path.parent}")
        return True
        
    except (PermissionError, OSError) as e:
        print(f"✗ Cannot write to {log_path.parent}: {e}")
        # Fall back to /tmp
        fallback_log = "/tmp/app.log"
        os.environ['LOG_FILE'] = fallback_log
        print(f"✓ Using fallback log location: {fallback_log}")
        return True


def main():
    """Initialize container environment."""
    print("Initializing container...")
    
    if not setup_logging_directory():
        print("Failed to setup logging directory")
        sys.exit(1)
    
    print("Container initialization complete")
    return True


if __name__ == "__main__":
    main()
