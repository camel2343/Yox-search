import subprocess
import sys
import os

def install():
    print("Installing Python dependencies...")
    pkgs = ["aiohttp", "requests", "Pillow", "playwright"]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + pkgs)
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.")
        sys.exit(1)
    
    print("Installing Playwright browsers (Chromium)...")
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
    except subprocess.CalledProcessError:
        print("Failed to install Playwright browsers.")
        # Don't exit, might still work with other methods or if already installed
    
    # Optional: Install semantic search dependencies
    print("\nWould you like to install semantic search dependencies? (sentence-transformers + torch)")
    print("This enables vector-based semantic search (~500MB download)")
    response = input("Install semantic search? (y/n): ").strip().lower()
    if response in ("y", "yes"):
        print("Installing sentence-transformers...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "torch"])
            print("Semantic search dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install semantic search dependencies. You can try later with:")
            print("  pip install sentence-transformers torch")
    
    print("\nSetup complete! You can now run 'run_local.bat' to start the search engine.")

if __name__ == "__main__":
    install()

