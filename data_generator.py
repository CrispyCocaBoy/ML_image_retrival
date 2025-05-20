from icrawler.builtin import BingImageCrawler

list_animal = ["crocodile", "Parrots"]

def data_generator(animal_list, directory):
    for item in animal_list:
        crawler = BingImageCrawler(storage={"root_dir": f"{directory}/{item}"})
        crawler.crawl(keyword=item, max_num=20)

if __name__ == "__main__":
    data_generator(list_animal, "data_example_rota/train")

