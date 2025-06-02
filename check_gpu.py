import subprocess
import torch

def check_pytorch_cuda():
    print("🔍 Verifica tramite PyTorch...")
    print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- Device count: {torch.cuda.device_count()}")
        print(f"- GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Nessuna GPU CUDA visibile da PyTorch.")

def check_nvidia_smi():
    print("\n🔍 Verifica `nvidia-smi` da terminale...")
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        print("✅ `nvidia-smi` funziona correttamente:")
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print("❌ Errore nell'esecuzione di `nvidia-smi`:")
        print(e.output.decode())
    except FileNotFoundError:
        print("❌ `nvidia-smi` non trovato. Probabilmente i driver NVIDIA non sono installati.")

if __name__ == "__main__":
    print("📦 Controllo stato GPU sulla VM Azure...\n")
    check_pytorch_cuda()
    check_nvidia_smi()
