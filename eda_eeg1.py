import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
dataset_path = 'eeg_5_95_std.pth'
dataset = torch.load(dataset_path)

# Extract EEG data, images, and labels
eeg_data_list = dataset['dataset']
eeg_signals = []
images = []
labels = []

# Determine the target length (use the minimum length for truncation)
target_length = min([item['eeg'].shape[1] for item in eeg_data_list])  # Min length for truncation

# Process each item in the dataset
for item in eeg_data_list:
    eeg_signal = item['eeg']
    # Truncate or pad the EEG signal
    if eeg_signal.shape[1] > target_length:
        eeg_signal = eeg_signal[:, :target_length]
    elif eeg_signal.shape[1] < target_length:
        pad_amount = target_length - eeg_signal.shape[1]
        eeg_signal = torch.nn.functional.pad(eeg_signal, (0, pad_amount), mode='constant', value=0)
    
    eeg_signals.append(eeg_signal)

    # Handle images
    img = item['image']
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    elif isinstance(img, list):
        img = np.array(img)

    images.append(img)
    labels.append(item['label'])

# Stack the EEG signals into a single tensor
eeg_data_tensor = torch.stack(eeg_signals)

# Basic statistics function for the EEG data
def basic_statistics(data_tensor):
    return {
        "mean": data_tensor.mean().item(),
        "std": data_tensor.std().item(),
        "min": data_tensor.min().item(),
        "max": data_tensor.max().item()
    }

# Call the basic statistics function
stats = basic_statistics(eeg_data_tensor)
print("Basic statistics:", stats)

# Visualization of EEG signals with corresponding images and labels
def plot_eeg_with_images(eeg_data, images, labels, sample_idx=0, n_channels=5):
    if len(eeg_data.shape) == 3:  # Assuming the shape is (samples, channels, timepoints)
        channels = eeg_data[sample_idx, :n_channels, :]  # Select first n_channels for a single sample
        timepoints = np.arange(channels.shape[1])
        
        # Create a figure with subplots
        fig, axs = plt.subplots(n_channels + 1, 1, figsize=(12, 8 + n_channels * 1.5))
        fig.suptitle(f"EEG Signals and Image for Sample {sample_idx}", fontsize=16)

        # Plot EEG signals
        for i in range(n_channels):
            axs[i].plot(timepoints, channels[i] + i * 10, label=f'Channel {i + 1}', lw=2)  # Shift signals for better visualization
            axs[i].set_title(f"EEG Signal (Channel {i + 1})", fontsize=14)
            axs[i].set_xlabel("Timepoints", fontsize=12)
            axs[i].set_ylabel("Amplitude (shifted)", fontsize=12)
            axs[i].legend()
            axs[i].grid(True)

        # Display the corresponding image
        img = images[sample_idx]
        axs[n_channels].imshow(img, cmap='gray' if img.ndim == 2 else 'viridis', aspect='auto')
        axs[n_channels].set_title(f"Image (Label: {labels[sample_idx]})", fontsize=14)
        axs[n_channels].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
        plt.show()
    else:
        print("Data shape is unexpected for plotting EEG signals.")

# Plot EEG signals with corresponding images and labels for the first sample
plot_eeg_with_images(eeg_data_tensor, images, labels, sample_idx=0, n_channels=5)

# Histogram of EEG signal values
def plot_signal_distribution(eeg_data, sample_idx=0, n_channels=5):
    if len(eeg_data.shape) == 3:  # Assuming the shape is (samples, channels, timepoints)
        channels = eeg_data[sample_idx, :n_channels, :]

        plt.figure(figsize=(12, 6))
        for i in range(n_channels):
            plt.hist(channels[i].flatten(), bins=50, alpha=0.6, label=f'Channel {i + 1}', edgecolor='black')
        plt.title(f"Signal Distribution (Sample {sample_idx})", fontsize=16)
        plt.xlabel("Signal Amplitude", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Data shape is unexpected for plotting signal distributions.")

# Plot the signal distribution for the first sample
plot_signal_distribution(eeg_data_tensor, sample_idx=0, n_channels=5)
