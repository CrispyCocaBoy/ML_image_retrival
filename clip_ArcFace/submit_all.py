import os
import json
import csv
import re
from src.test import submit

# Directory dei risultati
results_dir = "repository/results"
output_csv = "repository/submission_accuracy.csv"

# Estrai tutti i file JSON
result_files = sorted([
    f for f in os.listdir(results_dir)
    if f.endswith(".json") and re.match(r"model_epoch_\d+\.json", f)
])

# Lista per salvare i risultati
submission_results = []

# Loop sui file
for filename in result_files:
    filepath = os.path.join(results_dir, filename)

    # Estrai l'epoca dal nome
    match = re.search(r"model_epoch_(\d+)\.json", filename)
    if not match:
        continue
    epoch = int(match.group(1))

    # Carica il contenuto del JSON
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Errore nel leggere {filename}: {e}")
        continue

    print(f"\n📤 Sottomettendo risultati per epoch {epoch}...")

    # Invia al server
    accuracy = submit(data, group_name="Simple_Guys")
    submission_results.append({"epoch": epoch, "accuracy": accuracy})

# Scrittura CSV finale
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["epoch", "accuracy"])
    writer.writeheader()
    writer.writerows(sorted(submission_results, key=lambda x: x["epoch"]))

print(f"\n✅ Accuracy salvata in {output_csv}")


