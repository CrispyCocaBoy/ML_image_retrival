import torch
from src.siamsese import SiameseNetwork  # assicurati che il path sia corretto

# Imposta il device (puoi adattare questa parte come nel tuo script principale)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica i pesi dal file
state_dict = torch.load("model_repository/siamese_model.pth", map_location=device)

# Stampa le prime 5 chiavi per vedere se contengono "_orig_mod."
print("🔍 Prime chiavi dello state_dict:")
for k in list(state_dict.keys())[:5]:
    print("-", k)

# Se servisse, rimuove il prefisso _orig_mod. dalle chiavi
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    print("⚠️ Trovato prefisso '_orig_mod.': lo rimuovo...")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# Crea un'istanza del modello e carica i pesi
model = SiameseNetwork().to(device)
model.load_state_dict(state_dict)

print("✅ Pesi caricati correttamente nel modello.")
