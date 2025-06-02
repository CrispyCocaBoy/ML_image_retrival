import json
from src.submit import submit  # importa la funzione dal file esistente

if __name__ == "__main__":
    try:
        with open("retrieval_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ File 'retrieval_results.json' not found.")
        exit(1)
    except json.JSONDecodeError:
        print("❌ Invalid JSON in 'retrieval_results.json'.")
        exit(1)

    groupname = "gaysking"  # Cambialo se necessario
    submit(results, groupname)
