import torch

# Path to your .pth file containing the dataset
dataset_path = 'eeg_5_95_std.pth'

# Load the dataset
dataset = torch.load(dataset_path)

# Check the type of the dataset
print(f"Dataset type: {type(dataset)}")

# If the dataset is a dictionary, you can inspect its keys
if isinstance(dataset, dict):
    print("Keys in the dataset:", dataset.keys())

# To view the dataset contents (may be a tensor, list, or dict)
print("Dataset contents:")

# If it's a list or dictionary, iterate through the elements
if isinstance(dataset, dict):
    for key, value in dataset.items():
        print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")

elif isinstance(dataset, list):
    for i, item in enumerate(dataset):
        print(f"Item {i}: {item.shape if isinstance(item, torch.Tensor) else type(item)}")

elif isinstance(dataset, torch.Tensor):
    print(dataset)

# You can also print individual samples if it's not too large

print(dataset[0])