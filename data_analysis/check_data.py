import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def distribuzione_classi(train_path):
    """
    Ritorna un DataFrame con la distribuzione delle classi in base al numero di elementi.
    Colonne:
      - Numero di elementi: da 1 al numero massimo trovato
      - Numero di classi: quante classi hanno esattamente quel numero di elementi
    """
    counts = {}
    # Conta il numero di file per ogni classe
    for classe in os.listdir(train_path):
        classe_path = os.path.join(train_path, classe)
        if os.path.isdir(classe_path):
            num_elem = sum(
                1 for f in os.listdir(classe_path)
                if os.path.isfile(os.path.join(classe_path, f))
            )
            counts[num_elem] = counts.get(num_elem, 0) + 1

    # Costruisci la distribuzione completa da 1 al massimo numero di elementi
    max_count = max(counts.keys()) if counts else 0
    distribuzione = [(n, counts.get(n, 0)) for n in range(1, max_count + 1)]

    # Crea il DataFrame
    return pd.DataFrame(distribuzione, columns=["Number of elements", "Number of classes"])

if __name__ == "__main__":
    # Determina il percorso BASE del progetto rispetto a questo file
    base_dir = Path(__file__).resolve().parent.parent
    # Costruisci il path alla cartella train nella struttura data_original
    train_dir = base_dir / "data" / "train"

    # Converte in stringa per compatibilità con funzionalità os
    train_path_str = str(train_dir)

    # Calcola distribuzione
    df = distribuzione_classi(train_path_str)

    # Mostra la tabella (opzionale)
    print(df.to_string(index=False))

    # Disegna il bar chart
    plt.bar(df["Number of elements"], df["Number of classes"])
    plt.xlabel("Number of elements")
    plt.ylabel("Number of classes")
    plt.title("Distribution of classes x number of elements")
    plt.xticks(df["Number of elements"])
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
