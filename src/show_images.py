import os
import matplotlib.pyplot as plt
from PIL import Image

def show_image_results(results, top_k=3, max_queries=10,
                       base_query_dir="data_example_animal/test/query",
                       base_gallery_dir="data_example_animal/test/gallery"):
    """
    Visualizza i risultati della retrieval con immagini, ricostruendo i path.
    """
    for result in results[:max_queries]:
        query_filename = result["filename"]
        gallery_filenames = result["gallery_images"][:top_k]

        query_path = os.path.join(base_query_dir, os.path.basename(query_filename))
        gallery_paths = [os.path.join(base_gallery_dir, os.path.basename(g)) for g in gallery_filenames]

        fig, axs = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))
        fig.suptitle(f"Query: {os.path.basename(query_path)}", fontsize=12)

        try:
            axs[0].imshow(Image.open(query_path))
            axs[0].set_title("Query")
            axs[0].axis("off")

            for i, gallery_path in enumerate(gallery_paths):
                axs[i+1].imshow(Image.open(gallery_path))
                axs[i+1].set_title(f"Top {i+1}")
                axs[i+1].axis("off")

        except Exception as e:
            print(f"Errore visualizzando immagini: {e}")

        plt.tight_layout()
        plt.show()
