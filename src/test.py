from utils import list_images, load_dataset
paths = list_images("../data/images")
print(len(paths), paths[:5])
imgs, ps = load_dataset("../data/images", max_size=800)
print(len(imgs), len(ps))
