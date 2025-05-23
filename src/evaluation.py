import json
import os
import torch.nn.functional as F

import json
import os
import torch.nn.functional as F

def evaluation(predictions_path, ground_truth_path, top_k=3, verbose=True, debug=False):
    """
    Valuta un sistema di image retrieval confrontando le predizioni con la ground truth.
    
    Args:
        predictions_path (str or list): lista di predizioni o path al file JSON delle predizioni.
        ground_truth_path (str or dict): path al file JSON con ground truth o dizionario già caricato.
        top_k (int): numero di predizioni da considerare per query.
        verbose (bool): se True, stampa i dettagli per ogni query.
        debug (bool): se True, stampa informazioni di debug aggiuntive.
    
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
    
    # DEBUG: Stampa la struttura dei dati
    if debug:
        print("=== DEBUG INFO ===")
        print(f"Predictions type: {type(predictions)}")
        print(f"Ground truth type: {type(ground_truth)}")
        if predictions:
            print(f"First prediction entry: {predictions[0]}")
        if isinstance(ground_truth, dict):
            first_key = next(iter(ground_truth.keys()))
            print(f"First ground truth entry: {first_key} -> {ground_truth[first_key]}")
        print("==================\n")
    
    # Costruisci il dizionario query → immagini corrette
    if isinstance(ground_truth, dict):
        gt_dict = {}
        for query, gallery_images in ground_truth.items():
            # Prova diverse strategie per il matching dei nomi file
            query_basename = os.path.basename(query)
            query_name_no_ext = os.path.splitext(query_basename)[0]
            
            # Converti le gallery images in basename
            if isinstance(gallery_images, list):
                gallery_basenames = [os.path.basename(img) for img in gallery_images]
            else:
                gallery_basenames = [os.path.basename(gallery_images)]
            
            # Salva multiple versioni per il matching
            gt_dict[query_basename] = gallery_basenames
            gt_dict[query_name_no_ext] = gallery_basenames
            gt_dict[query] = gallery_basenames  # Mantieni anche il path completo
    else:
        raise ValueError("Formato ground truth non valido: ci si aspetta un dizionario.")
    
    total = 0
    correct_total = 0
    
    if verbose:
        print("=== Retrieval Evaluation ===\n")
    
    # Ordina le predizioni per nome file (facilita il debug)
    for entry in sorted(predictions, key=lambda x: x.get("filename", "")):
        query_file = entry.get("filename", "")
        if not query_file:
            if debug:
                print(f"WARNING: Entry senza filename: {entry}")
            continue
            
        # Prova diverse strategie per trovare la query nella ground truth
        query_basename = os.path.basename(query_file)
        query_name_no_ext = os.path.splitext(query_basename)[0]
        
        expected_files = None
        matched_key = None
        
        # Prova a matchare con basename completo
        if query_basename in gt_dict:
            expected_files = gt_dict[query_basename]
            matched_key = query_basename
        # Prova a matchare senza estensione
        elif query_name_no_ext in gt_dict:
            expected_files = gt_dict[query_name_no_ext]
            matched_key = query_name_no_ext
        # Prova a matchare con path completo
        elif query_file in gt_dict:
            expected_files = gt_dict[query_file]
            matched_key = query_file
        
        if expected_files is None:
            if debug:
                print(f"WARNING: Query '{query_file}' non trovata nella ground truth")
                print(f"  Available GT keys: {list(gt_dict.keys())[:5]}...")
            continue
        
        # Estrai le predizioni
        gallery_images = entry.get("gallery_images", [])
        if not gallery_images:
            if debug:
                print(f"WARNING: Nessuna gallery_images per query '{query_file}'")
            continue
            
        predicted_files = [os.path.basename(img) for img in gallery_images[:top_k]]
        
        # Calcola l'intersezione
        predicted_set = set(predicted_files)
        correct_set = set(expected_files)
        correct = predicted_set & correct_set
        incorrect = predicted_set - correct_set
        
        correct_count = len(correct)
        total += top_k
        correct_total += correct_count
        
        if verbose:
            print(f"{query_basename}: {correct_count}/{top_k} correct (matched with key: {matched_key})")
            print(f"  ✅ Expected: {expected_files}")
            print(f"  🔎 Predicted: {predicted_files}")
            if correct:
                print(f"  ✅ Correct matches: {sorted(list(correct))}")
            if incorrect:
                print(f"  ❌ Incorrect: {sorted(list(incorrect))}")
            print()
    
    accuracy = (correct_total / total) * 100 if total > 0 else 0.0
    
    if verbose:
        print(f"✅ Accuracy totale: {accuracy:.2f}% su {len([p for p in predictions if p.get('filename')])} query")
        print(f"   Correct predictions: {correct_total}/{total}\n")
    
    return accuracy


def debug_data_formats(predictions_path, ground_truth_path):
    """
    Funzione di debug per ispezionare i formati dei dati.
    """
    print("=== DATA FORMAT DEBUG ===\n")
    
    # Carica e ispeziona predictions
    if isinstance(predictions_path, str):
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
    else:
        predictions = predictions_path
    
    print("PREDICTIONS:")
    print(f"Type: {type(predictions)}")
    print(f"Length: {len(predictions) if isinstance(predictions, list) else 'N/A'}")
    if predictions:
        print(f"First entry keys: {list(predictions[0].keys()) if isinstance(predictions, list) else 'N/A'}")
        print(f"First entry: {predictions[0] if isinstance(predictions, list) else predictions}")
    print()
    
    # Carica e ispeziona ground truth
    if isinstance(ground_truth_path, str):
        with open(ground_truth_path, "r") as f:
            ground_truth = json.load(f)
    else:
        ground_truth = ground_truth_path
    
    print("GROUND TRUTH:")
    print(f"Type: {type(ground_truth)}")
    print(f"Keys count: {len(ground_truth) if isinstance(ground_truth, dict) else 'N/A'}")
    if isinstance(ground_truth, dict):
        first_key = next(iter(ground_truth.keys()))
        print(f"First key: {first_key}")
        print(f"First value: {ground_truth[first_key]}")
        print(f"Sample keys: {list(ground_truth.keys())[:5]}")
    print()
    
    # Confronta i formati dei nomi file
    if isinstance(predictions, list) and predictions and isinstance(ground_truth, dict):
        print("FILENAME COMPARISON:")
        pred_files = [os.path.basename(p.get("filename", "")) for p in predictions[:3]]
        gt_keys = [os.path.basename(k) for k in list(ground_truth.keys())[:3]]
        
        print(f"Sample prediction filenames: {pred_files}")
        print(f"Sample ground truth keys: {gt_keys}")
        print()


