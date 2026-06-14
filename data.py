from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loader():

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        root="dataset",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    return loader
