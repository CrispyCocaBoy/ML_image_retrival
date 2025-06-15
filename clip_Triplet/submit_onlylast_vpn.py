import os
import json
from clip_Triplet.src.submit import submit

if __name__ == "__main__":
    results_dir = "repository/retrieval_repository"
    final_file = "retrieval_results_final.json"
    final_path = os.path.join(results_dir, final_file)

    try:
        if not os.path.exists(final_path):
            print("❌ File 'retrieval_results_final.json' not found.")
            exit(1)

        print(f"Loading file: {final_file}")

        with open(final_path, "r") as f:
            results = json.load(f)

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        exit(1)

    groupname = "simple_guys"
    submit(results, groupname)
