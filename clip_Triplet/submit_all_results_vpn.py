import os
import json
import csv
from clip_Triplet.src.submit import submit

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPO_DIR = os.path.join(BASE_DIR, "clip_Triplet", "repository")
    LOGS_DIR = os.path.join(REPO_DIR, "logs")
    RETRIEVAL_DIR = os.path.join(REPO_DIR, "retrieval_repository")

    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, "epoch_scores.csv")

    result_files = [
        f for f in os.listdir(RETRIEVAL_DIR)
        if f.startswith("retrieval_results_") and f.endswith(".json") and f.split("_")[-1].split(".")[0].isdigit()
    ]

    result_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "score"])

        for filename in result_files:
            epoch_num = int(filename.split("_")[-1].split(".")[0])
            filepath = os.path.join(RETRIEVAL_DIR, filename)

            print(f"Processing epoch {epoch_num}...")

            with open(filepath, "r") as f:
                results = json.load(f)

            groupname = "simple_guys"
            score = submit(results, groupname)

            writer.writerow([epoch_num, score])

    print(f"Scores saved in {csv_path}")

if __name__ == "__main__":
    main()
