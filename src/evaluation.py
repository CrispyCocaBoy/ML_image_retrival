import json
import os

def evaluation(predictions_path, ground_truth_path, top_k=3):
    """
    predictions_path: list of predictions OR path to a JSON file
    ground_truth_path: path to ground truth JSON
    """

    # Se predictions è un path, carica da file
    if isinstance(predictions_path, str):
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
    else:
        predictions = predictions_path

    # Carica ground truth
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    # Trasforma ground truth in dizionario
    gt_dict = {
        os.path.basename(entry["filename"]):
        [os.path.basename(img) for img in entry["gallery_images"]]
        for entry in ground_truth
    }

    total = 0
    correct_total = 0

    print("=== Retrieval Evaluation ===\n")

    for entry in predictions:
        query_file = os.path.basename(entry["filename"])
        predicted_files = [os.path.basename(img) for img in entry["gallery_images"][:top_k]]
        expected_files = gt_dict.get(query_file, [])

        # confronto insiemistico
        predicted_set = set(predicted_files)
        correct_set = set(expected_files)
        correct = predicted_set & correct_set
        incorrect = predicted_set - correct_set

        correct_count = len(correct)
        total += top_k
        correct_total += correct_count

        print(f"{query_file}: {correct_count}/{top_k} correct")
        print(f"   ✅ Expected: {expected_files}")
        print(f"   🔎 Predicted: {predicted_files}")
        if incorrect:
            print(f"   ❌ Incorrect: {sorted(list(incorrect))}")
        print()

    accuracy = correct_total / total * 100 if total > 0 else 0.0
    print(f"✅ Accuracy totale: {accuracy:.2f}% su {len(predictions)} query\n")
    return accuracy
