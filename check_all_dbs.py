import sqlite3
import os
import glob

def check_db(path):
    print(f"\n--- Checking: {path} ---")
    if not os.path.exists(path):
        print("  [ERROR] File not found.")
        return

    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
        
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Check integrity (brief)
        try:
            cursor.execute("PRAGMA quick_check;")
            integrity = cursor.fetchone()[0]
            print(f"  Integrity: {integrity}")
        except Exception as e:
            print(f"  [ERROR] Integrity check failed: {e}")

        # Check document count
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents';")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM documents")
                count = cursor.fetchone()[0]
                print(f"  Document Count: {count}")
            else:
                print("  [INFO] No 'documents' table found.")
        except Exception as e:
            print(f"  [ERROR] Could not query documents: {e}")

        conn.close()

    except sqlite3.DatabaseError as e:
        print(f"  [ERROR] Not a valid SQLite database or corrupted: {e}")
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")

def main():
    # Find all .db files recursively
    root_dir = "."
    db_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".db"):
                db_files.append(os.path.join(root, file))
    
    if not db_files:
        print("No .db files found.")
        return

    print(f"Found {len(db_files)} database files. Checking them now...")
    for db_path in db_files:
        check_db(db_path)

if __name__ == "__main__":
    main()
