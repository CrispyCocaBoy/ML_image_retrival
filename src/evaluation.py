import json
import os
import torch.nn.functional as F

def evaluation(predictions_path, ground_truth_path, top_k=3, verbose=True):
    """
    Valuta un sistema di image retrieval confrontando le predizioni con la ground truth.

    Args:
        predictions_path (str or list): lista di predizioni o path al file JSON delle predizioni.
        ground_truth_path (str or dict): path al file JSON con ground truth o dizionario già caricato.
        top_k (int): numero di predizioni da considerare per query.
        verbose (bool): se True, stampa i dettagli per ogni query.

    Returns:
        float: accuratezza percentuale sulle top_k predizioni.
    """

    # Carica le predizioni
    if isinstance(predictions_path, str):
        if not os.path.isfile(predictions_path):
            raise FileNotFoundError(f"File delle predizioni non trovato: {predictions_path}")
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
    elif isinstance(predictions_path, list):
        predictions = predictions_path
    else:
        raise ValueError("`predictions_path` deve essere un path stringa o una lista di dizionari.")

    # Carica la ground truth
    if isinstance(ground_truth_path, str):
        if not os.path.isfile(ground_truth_path):
            raise FileNotFoundError(f"File di ground truth non trovato: {ground_truth_path}")
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)
    elif isinstance(ground_truth_path, dict):
        ground_truth = ground_truth_path
    else:
        raise ValueError("`ground_truth_path` deve essere un path o un dizionario.")

    # Costruisci il dizionario query → immagini corrette
    if isinstance(ground_truth, dict):
        gt_dict = {
            os.path.basename(query): [os.path.basename(img) for img in gallery_images]
            for query, gallery_images in ground_truth.items()
        }
    else:
        raise ValueError("Formato ground truth non valido: ci si aspetta un dizionario.")

    total = 0
    correct_total = 0

    if verbose:
        print("=== Retrieval Evaluation ===\n")

    # Ordina le predizioni per nome file (facilita il debug)
    for entry in sorted(predictions, key=lambda x: x["filename"]):
        query_file = os.path.basename(entry["filename"])
        predicted_files = [os.path.basename(img) for img in entry["gallery_images"][:top_k]]
        expected_files = gt_dict.get(query_file, [])

        predicted_set = set(predicted_files)
        correct_set = set(expected_files)
        correct = predicted_set & correct_set
        incorrect = predicted_set - correct_set

        correct_count = len(correct)
        total += top_k
        correct_total += correct_count

        if verbose:
            print(f"{query_file}: {correct_count}/{top_k} correct")
            print(f"   ✅ Expected: {expected_files}")
            print(f"   🔎 Predicted: {predicted_files}")
            if incorrect:
                print(f"   ❌ Incorrect: {sorted(list(incorrect))}")
            print()

    accuracy = (correct_total / total) * 100 if total > 0 else 0.0
    if verbose:
        print(f"✅ Accuracy totale: {accuracy:.2f}% su {len(predictions)} query\n")

    return accuracy
