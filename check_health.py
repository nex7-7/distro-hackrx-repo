#!/usr/bin/env python3
"""
Health check script for the RAG Application API.

This script validates that all required dependencies and services
are available before starting the main application.

Usage:
    python check_health.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required. Current: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_environment_variables():
    """Check required environment variables."""
    print("\n🔧 Checking environment variables...")
    
    required_vars = [
        "API_BEARER_TOKEN",
        "GOOGLE_API_KEY",
        "WEAVIATE_URL",
        "EMBEDDING_MODEL"
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing.append(var)
            print(f"❌ Missing: {var}")
        else:
            # Hide sensitive values
            if "KEY" in var or "TOKEN" in var:
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"✅ {var}={display_value}")
    
    if missing:
        print(f"\n❌ Missing environment variables: {missing}")
        print("Please ensure your .env file contains all required variables")
        return False
    return True

def check_imports():
    """Check critical imports."""
    print("\n📦 Checking critical imports...")
    
    import_checks = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("weaviate", "Weaviate client"),
        ("google.generativeai", "Google AI client"),
        ("sentence_transformers", "Sentence transformers"),
        ("pydantic", "Pydantic data validation"),
    ]
    
    failed_imports = []
    for module, description in import_checks:
        try:
            __import__(module)
            print(f"✅ {module} - {description}")
        except ImportError as e:
            print(f"❌ {module} - {description}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {failed_imports}")
        print("Run: pip install -r requirements.txt")
        return False
    return True

def check_weaviate_connection():
    """Check Weaviate service connectivity."""
    print("\n🔗 Checking Weaviate connection...")
    
    try:
        import requests
        from config.settings import settings
        
        url = f"{settings.weaviate_url}/v1/.well-known/ready"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Weaviate is ready at {settings.weaviate_url}")
            return True
        else:
            print(f"❌ Weaviate health check failed (status: {response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to Weaviate: {e}")
        print("Make sure Weaviate is running: docker compose up weaviate")
        return False

def check_directories():
    """Check required directories exist."""
    print("\n📁 Checking directories...")
    
    required_dirs = ["logs", "temp", "downloads", "cache"]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            print(f"📁 Creating directory: {dir_name}")
            dir_path.mkdir(exist_ok=True)
        print(f"✅ {dir_name}/")
    
    return True

def main():
    """Main health check function."""
    print("🏥 RAG Application API - Health Check")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment Variables", check_environment_variables),
        ("Python Imports", check_imports),
        ("Directories", check_directories),
        ("Weaviate Connection", check_weaviate_connection),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Health Check Summary:")
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! Ready to start the RAG API.")
        print("Run: python run_dev.py")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Health check cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during health check: {e}")
        sys.exit(1)
