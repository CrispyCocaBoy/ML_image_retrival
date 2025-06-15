import json
import os
import re
import csv
from src.submit import submit

# --- Configuration for directories and output file ---
RESULTS_DIR = "repository/results"
OUTPUT_CSV_PATH = "repository/submission_accuracy.csv" # The path for your CSV file

if __name__ == "__main__":
    groupname = "SimpleGuys"  
    
    print(f"🚀 Starting analysis and submission of all retrieval results in '{RESULTS_DIR}'...")

    # --- Find all JSON files in the results directory ---
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Results directory '{RESULTS_DIR}' not found.")
        print("Please ensure your evaluation script has generated JSON files in this location.")
        exit(1)

    json_files = sorted([
        f for f in os.listdir(RESULTS_DIR)
        if f.endswith('.json') and f.startswith('retrieval_results_epoch_')
    ])

    if not json_files:
        print(f"No matching JSON files found in '{RESULTS_DIR}'.")
        print("Please ensure your evaluation script (e.g., submit_all.py) has created them.")
        exit(0)

    print(f"Found {len(json_files)} result files in '{RESULTS_DIR}'. Processing...")
    print("-" * 50)

    # --- List to store collected epoch and accuracy data ---
    submission_results = []

    # --- Iterate through each JSON file ---
    for filename in json_files:
        filepath = os.path.join(RESULTS_DIR, filename)
        
        # Extract epoch number from filename 
        epoch_match = re.search(r'epoch_(\d+)\.json', filename)
        if not epoch_match:
            print(f"Skipping {filename}: could not extract epoch number. Check filename format.")
            continue
        epoch = int(epoch_match.group(1))

        print(f"Processing file: {filename} (Epoch {epoch})")

        try:
            with open(filepath, "r") as f:
                results = json.load(f)
            
            accuracy = submit(results, groupname) 
            
            # Check if submit returned a valid accuracy before adding to results list
            if accuracy is not None:
                submission_results.append({"epoch": epoch, "accuracy": accuracy})
            else:
                print(f"Submission for epoch {epoch} did not return a valid accuracy. Skipping this result in CSV.")

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found at '{filepath}'. Skipping.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{filename}'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{filename}': {e}. Skipping.")
        
        print("-" * 50)

    # --- Save all collected accuracies to a CSV file ---
    if submission_results: 
        # Ensure the 'repository/' directory exists
        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        
        with open(OUTPUT_CSV_PATH, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["epoch", "accuracy"])
            writer.writeheader() # Write the 'epoch' and 'accuracy' column headers
            writer.writerows(sorted(submission_results, key=lambda x: x["epoch"])) 
        print(f"\nAll accuracy scores saved to {OUTPUT_CSV_PATH}")
    else:
        print(f"\nNo successful submissions were processed to save to {OUTPUT_CSV_PATH}.")

    print("\nFinished processing all available JSON result files.")
