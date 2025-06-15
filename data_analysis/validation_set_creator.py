import os
import random
import shutil


def crea_validation_set(
        train_dir='/Users/matteomassari/image_retrival/siamese_clip_no_head/data/train',
        val_dir='/Users/matteomassari/image_retrival/siamese_clip_no_head/data/validation',
        n_val_per_class=3
):
    # Verifica che la directory di train esista
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Directory di training non trovata: {train_dir}")

    os.makedirs(val_dir, exist_ok=True)

    # Per ogni classe
    for classe in os.listdir(train_dir):
        classe_train_path = os.path.join(train_dir, classe)
        classe_val_path = os.path.join(val_dir, classe)

        if not os.path.isdir(classe_train_path):
            continue  # salta se non è una cartella

        # Trova immagini valide
        immagini = [f for f in os.listdir(classe_train_path)
                    if os.path.isfile(os.path.join(classe_train_path, f)) and
                    f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(immagini) < n_val_per_class:
            print(f"[SKIP] Classe '{classe}' ha solo {len(immagini)} immagini.")
            continue

        # Crea la cartella per la classe in validation
        os.makedirs(classe_val_path, exist_ok=True)

        # Seleziona immagini casuali e spostale
        selezionate = random.sample(immagini, n_val_per_class)
        for nome_img in selezionate:
            src = os.path.join(classe_train_path, nome_img)
            dst = os.path.join(classe_val_path, nome_img)
            shutil.move(src, dst)

        print(f"[OK] Classe '{classe}': spostate {n_val_per_class} immagini in validation.")


if __name__ == "__main__":
    crea_validation_set()
