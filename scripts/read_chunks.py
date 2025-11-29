
def read_chunks(path):
    try:
        with open(path, "r", encoding="utf-16") as f:
            content = f.read()
            print("--- CHUNK 1 ---")
            print(content[:2000])
            print("--- CHUNK 2 ---")
            print(content[2000:4000])
            print("--- CHUNK 3 ---")
            print(content[4000:])
    except Exception as e:
        print(f"Error: {e}")

read_chunks(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/validation_results/output.txt")
