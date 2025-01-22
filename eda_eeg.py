import torch
import matplotlib.pyplot as plt
import numpy as np

# Path to your .pth file containing the dataset
dataset_path = 'eeg_5_95_std.pth'

# Load the dataset
dataset = torch.load(dataset_path)

# Inspect dataset type and contents
print(f"Dataset type: {type(dataset)}")
print("Keys in the dataset:", dataset.keys())
eeg_data_list = dataset['dataset']

# Check the first few items in the dataset for debugging
for i in range(5):  # Print first 5 items
    item = eeg_data_list[i]
    print(f"Item {i}:")
    print(f"  EEG shape: {item['eeg'].shape if isinstance(item['eeg'], torch.Tensor) else type(item['eeg'])}")
    print(f"  Image: {item['image']} (type: {type(item['image'])})")
    print(f"  Label: {item['label']} (type: {type(item['label'])})")
    print(f"  Subject: {item['subject']}\n")

# Determine the target length (use the minimum length for truncation)
target_length = min([item['eeg'].shape[1] for item in eeg_data_list])  # Min length for truncation

# Truncate or pad the EEG data to the target length
eeg_signals = []
images = []
labels = []

for item in eeg_data_list:
    eeg_signal = item['eeg']
    
    # Truncate or pad the EEG signal
    if eeg_signal.shape[1] > target_length:  # Truncate if necessary
        eeg_signal = eeg_signal[:, :target_length]
    elif eeg_signal.shape[1] < target_length:  # Pad if necessary
        pad_amount = target_length - eeg_signal.shape[1]
        eeg_signal = torch.nn.functional.pad(eeg_signal, (0, pad_amount), mode='constant', value=0)
    
    eeg_signals.append(eeg_signal)


    # Handle images
    img = item['image']
    '''
    if isinstance(img, int):
        print("..")
        # You may want to replace the image with a placeholder or load the actual image if available
        #img = np.zeros((128, 128))  # Placeholder for missing images; adjust size as needed
    elif isinstance(img, torch.Tensor):
        img = img.numpy()  # Convert to NumPy array if necessary
    elif isinstance(img, list):
        img = np.array(img)  # Convert list to numpy array if necessary
    '''    

    
    
    images.append(img)
    #print(img)
    labels.append(item['label'])
    #print(item['label'])



# Stack the EEG signals into a single tensor
eeg_data_tensor = torch.stack(eeg_signals)

# Check the shape of the EEG tensor
print(f"EEG data tensor shape: {eeg_data_tensor.shape}")  # Should be (samples, channels, timepoints)

# Create a separate dataset with just images and labels
image_label_dataset = [{'image': img, 'label': lbl} for img, lbl in zip(images, labels)]

# Basic statistics function for the EEG data
def basic_statistics(data_tensor):
    print("Basic statistics:")
    print(f"Mean: {data_tensor.mean().item()}")
    print(f"Std: {data_tensor.std().item()}")
    print(f"Min: {data_tensor.min().item()}")
    print(f"Max: {data_tensor.max().item()}")

# Call the basic statistics function
basic_statistics(eeg_data_tensor)

# Visualization of EEG signals with corresponding images and labels
def plot_eeg_with_images(eeg_data, images, labels, sample_idx=0, n_channels=5):
    """
    Plots EEG signals along with the corresponding image and label for a given sample index.
    """
    if len(eeg_data.shape) == 3:  # Assuming the shape is (samples, channels, timepoints)
        channels = eeg_data[sample_idx, :n_channels, :]  # Select first n_channels for a single sample
        timepoints = np.arange(channels.shape[1])
        
        # Create a figure with subplots
        #fig, axs = plt.subplots(n_channels + 1, 1, figsize=(12, 6 + n_channels * 2))
        colors = plt.cm.viridis(np.linspace(0, 1, n_channels))  # Use a colormap for colors

        # Plot EEG signals
        for i in range(n_channels):
            plt.plot(timepoints, channels[i] + i * 5, color=colors[i], label=f'Channel {i+1}')  # Shift signals for better visualization
            plt.title(f"EEG Signal (Channel {i + 1})", fontsize=14)
            plt.xlabel("Timepoints", fontsize=12)
            plt.ylabel("Amplitude", fontsize=12)
            plt.legend()
            plt.grid(True)
        
        '''
        # Display the corresponding image
        img = images[sample_idx]
        axs[n_channels].axis('off')  # Turn off axis for the image display

        if img.ndim == 2:  # Grayscale
            axs[n_channels].imshow(img, cmap='gray', aspect='auto')
        elif img.ndim == 3:  # Color
            axs[n_channels].imshow(img.transpose((1, 2, 0)), aspect='auto')

        axs[n_channels].set_title(f"Image (Label: {labels[sample_idx]})", fontsize=14)

        plt.tight_layout()
        plt.show()
        '''
    else:
        print("Data shape is unexpected for plotting EEG signals.")




# Plot EEG signals with corresponding images and labels for the first sample
plot_eeg_with_images(eeg_data_tensor, images, labels, sample_idx=0, n_channels=5)

# Histogram of EEG signal values
def plot_signal_distribution(eeg_data, sample_idx=0, n_channels=5):
    """
    Plots the distribution of EEG signal values for a given sample.
    """
    if len(eeg_data.shape) == 3:  # Assuming the shape is (samples, channels, timepoints)
        channels = eeg_data[sample_idx, :n_channels, :]

        plt.figure(figsize=(10, 6))
        for i in range(n_channels):
            plt.hist(channels[i].flatten(), bins=50, alpha=0.6, label=f'Channel {i+1}')
        plt.title(f"Signal Distribution (Sample {sample_idx})")
        plt.xlabel("Signal Amplitude")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
    else:
        print("Data shape is unexpected for plotting signal distributions.")

# Plot the signal distribution for the first sample
plot_signal_distribution(eeg_data_tensor, sample_idx=0, n_channels=5)
