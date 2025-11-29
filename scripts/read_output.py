
try:
    with open(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/validation_results/output.txt", "r", encoding="utf-16") as f:
        print(f.read())
except Exception as e:
    print(f"Error reading utf-16: {e}")
    try:
        with open(r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/validation_results/output.txt", "r", encoding="utf-8") as f:
            print(f.read())
    except Exception as e2:
        print(f"Error reading utf-8: {e2}")
