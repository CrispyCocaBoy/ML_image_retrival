
def run():
    train_loader, query_loader, gallery_loader = retrival_data_loading(
        train_data_root="path/to/train",
        query_data_root="path/to/query",
        gallery_data_root="path/to/gallery",
        triplet_loss=True,
        batch_size=32
    )

if __name__ == "__main__":
    run()
