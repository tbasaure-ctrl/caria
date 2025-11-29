
import os

def read_chunks(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    content = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

    print("--- FULL OUTPUT ---")
    print(content)

read_chunks(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/validation_results_real/output_real_v5.txt")
