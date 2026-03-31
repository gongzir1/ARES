from collections import namedtuple
import torch
import matplotlib.pyplot as plt  # lazily import this here
from torchvision import datasets, transforms
import os
os.makedirs("Figures", exist_ok=True)
import torch
import matplotlib.pyplot as plt
import os
import torchaudio
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import make_interp_spline

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'lines.linewidth': 2.5,
})

# def plot_audio_from_mel_polished(user_data, sample_rate=16000, save_path=None, print_labels=False):
#     """
#     Convert Mel-spectrogram [1, F, T] back to waveform and plot with polished style.
#     """
#     data = user_data["data"].clone().detach().cpu()
#     labels = user_data.get("labels", None)
#     if labels is not None:
#         labels = labels.clone().detach().cpu()

#     n_samples = data.shape[0]
#     n_cols = 1
#     n_rows = (n_samples + n_cols - 1) // n_cols
#     # fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 3*n_rows))
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 4))
#     if isinstance(axes, plt.Axes):
#         axes = [axes]
#     else:
#         axes = axes.flatten()

#     color = '#866AA3'  # line color
#     # marker = 'o'

#     for i, mel in enumerate(data):
#         mel = mel.squeeze()  # [F, T]

#         # Invert Mel -> linear spectrogram
#         mel_transform = torchaudio.transforms.InverseMelScale(
#             n_stft=1024//2 + 1,
#             n_mels=mel.shape[0],
#             sample_rate=sample_rate
#         )
#         linear_spec = mel_transform(mel.unsqueeze(0))

#         # Reconstruct waveform with Griffin-Lim
#         griffin_lim = torchaudio.transforms.GriffinLim(
#             n_fft=1024,
#             n_iter=32,
#             win_length=1024,
#             hop_length=256
#         )
#         waveform = griffin_lim(linear_spec.squeeze(0))  # [T]
#         waveform = waveform.numpy()

#         # Smooth waveform for nicer plotting
#         x = np.arange(len(waveform))
#         x_smooth = np.linspace(x.min(), x.max(), 1000)
#         spline = make_interp_spline(x, waveform, k=2)
#         y_smooth = spline(x_smooth)

#         axes[i].plot(x_smooth, y_smooth, color=color, linewidth=2)
#         # axes[i].scatter(x[::len(x)//50], waveform[::len(x)//50], color=color, s=30)  # optional markers

#         # axes[i].set_xlabel("Samples")
#         # axes[i].set_ylabel("Amplitude")
#         axes[i].grid(True, alpha=0.3)
#         # if print_labels and labels is not None:
#         #     axes[i].set_title(f"Label: {labels[i].item()}", fontsize=18)

#     # Turn off unused axes
#     for j in range(i+1, len(axes)):
#         axes[j].axis("off")

#     plt.tight_layout()
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, format='pdf', bbox_inches='tight')
#         print(f"[INFO] Figure saved to: {save_path}")
#     plt.show()


def plot_audio_from_mel_polished(user_data, sample_rate=16000, save_path=None):
    """
    Convert Mel-spectrograms [B, F, T] back to waveform(s) and plot with polished style.
    Handles both single sample and batch plotting.
    """
    data = user_data["data"].clone().detach().cpu()  # [B, F, T]

    n_samples = data.shape[0]
    color = '#866AA3'

    # ---- Case 1: Single sample ----
    if n_samples == 1:
        mel = data[0].squeeze()

        # Invert Mel -> linear spectrogram
        mel_transform = torchaudio.transforms.InverseMelScale(
            n_stft=1024 // 2 + 1,
            n_mels=mel.shape[0],
            sample_rate=sample_rate
        )
        linear_spec = mel_transform(mel.unsqueeze(0))

        # Reconstruct waveform with Griffin-Lim
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1024,
            n_iter=32,
            win_length=1024,
            hop_length=256
        )
        waveform = griffin_lim(linear_spec.squeeze(0)).numpy()

        # Time axis in seconds
        x = np.arange(len(waveform)) / sample_rate
        x_smooth = np.linspace(x.min(), x.max(), 1000)
        spline = make_interp_spline(x, waveform, k=2)
        y_smooth = spline(x_smooth)

        plt.figure(figsize=(6, 4))
        plt.plot(x_smooth, y_smooth, color=color, linewidth=2)
        plt.ylim(-0.1, 0.1)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    # ---- Case 2: Batch of samples ----
    else:
        n_cols = min(8, n_samples)
        n_rows = (n_samples + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, mel in enumerate(data):
            mel = mel.squeeze()

            # Invert Mel -> linear spectrogram
            mel_transform = torchaudio.transforms.InverseMelScale(
                n_stft=1024 // 2 + 1,
                n_mels=mel.shape[0],
                sample_rate=sample_rate
            )
            linear_spec = mel_transform(mel.unsqueeze(0))

            # Reconstruct waveform
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=1024,
                n_iter=32,
                win_length=1024,
                hop_length=256
            )
            waveform = griffin_lim(linear_spec.squeeze(0)).numpy()

            # Time axis in seconds
            x = np.arange(len(waveform)) / sample_rate
            x_smooth = np.linspace(x.min(), x.max(), 1000)
            spline = make_interp_spline(x, waveform, k=2)
            y_smooth = spline(x_smooth)

            axes[i].plot(x_smooth, y_smooth, color=color, linewidth=2)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()

    # Save figure if path is given
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")

    plt.show()


def plot_audio_from_mel(user_data, sample_rate=16000, save_path=None, print_labels=False):
    """
    Converts Mel-spectrogram [1, F, T] back to waveform and plots it.
    user_data["data"]: [N, 1, F, T] or [N, F, T]
    """
    data = user_data["data"].clone().detach().cpu()
    labels = user_data.get("labels", None)
    if labels is not None:
        labels = labels.clone().detach().cpu()

    n_samples = data.shape[0]

    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, mel in enumerate(data):
        mel = mel.squeeze()  # [F, T]

        # Invert Mel -> linear spectrogram
        mel_transform = torchaudio.transforms.InverseMelScale(
            n_stft=1024//2 + 1,  # n_fft//2 + 1
            n_mels=mel.shape[0],
            sample_rate=sample_rate
        )
        linear_spec = mel_transform(mel.unsqueeze(0))  # [1, F', T]

        # Reconstruct waveform with Griffin-Lim
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=1024,
            n_iter=32,
            win_length=1024,
            hop_length=256
        )
        waveform = griffin_lim(linear_spec.squeeze(0))  # [T]

        axes[i].plot(waveform.numpy())
        axes[i].set_xlabel("Samples")
        axes[i].set_ylabel("Amplitude")
        if print_labels and labels is not None:
            axes[i].set_title(f"Label: {labels[i].item()}", fontsize=10)
        axes[i].grid(True)

    # Turn off unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")
    plt.show()


def plot_audio(cfg, user_data, setup, scale=False, print_labels=False, save_path=None, sample_rate=16000):
    """
    Plot user audio data to output. Can plot multiple audio samples in a grid.
    user_data["data"] should have shape [N, C, T] (N=batch, C=channel, T=time).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = user_data["data"].to(device).clone().detach()
    labels = user_data["labels"].to(device).clone().detach() if user_data["labels"] is not None else None

    if labels is None:
        print_labels = False

    # Optionally scale to [0,1] per sample
    if scale:
        min_val, max_val = data.amin(dim=2, keepdim=True), data.amax(dim=2, keepdim=True)
        data = (data - min_val) / (max_val - min_val)
    else:
        # Normalize using mean/std if available
        if hasattr(cfg, "mean") and hasattr(cfg, "std"):
            dm = torch.as_tensor(cfg.mean, **setup).to(device)
            ds = torch.as_tensor(cfg.std, **setup).to(device)
            data = data * ds.view(1, -1, 1) + dm.view(1, -1, 1)

    data = data.to(dtype=torch.float32).cpu()

    n_samples = data.shape[0]
    n_cols = 2  # adjust for grid
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 2.5*n_rows))

    # Handle the case where axes is a single Axes object
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, audio in enumerate(data):
        audio = audio.squeeze()  # remove channel dim if mono
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        axes[i].plot(audio)
        axes[i].set_xlabel("Samples")
        axes[i].set_ylabel("Amplitude")
        if print_labels and labels is not None:
            axes[i].set_title(f"Label: {labels[i].item()}", fontsize=10)
        axes[i].grid(True)

    # Turn off unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")

    plt.show()


def plot_data(cfg, user_data, setup, scale=False, print_labels=False,save_path=None):
    """Plot user data to output. Probably best called from a jupyter notebook."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # convert mean
    dm = torch.as_tensor(cfg.mean, **setup)
    if dm.dim() == 0:        # scalar
        dm = dm.unsqueeze(0) # make shape [1]
    dm = dm[None, :, None, None].to(device)  # shape [1, C, 1, 1]

    # convert std
    ds = torch.as_tensor(cfg.std, **setup)
    if ds.dim() == 0:        # scalar
        ds = ds.unsqueeze(0)
    ds = ds[None, :, None, None].to(device)  # shape [1, C, 1, 1]

    # dm = torch.as_tensor(cfg.mean, **setup)[None, :, None, None].to(device)
    # ds = torch.as_tensor(cfg.std, **setup)[None, :, None, None].to(device)

    data = user_data["data"].to(device).clone().detach()
    labels = user_data["labels"].to(device).clone().detach() if user_data["labels"] is not None else None
    
    imagenet_val = datasets.ImageNet(
        root="/home/test/GIA/robbing_the_fed/data/",
        split="val",
        transform=transforms.ToTensor()
    )

    classes = imagenet_val.classes  # <-- class names (e.g., "n01440764", "n01443537", ...)
    # class_to_idx = imagenet_val.class_to_idx  # dict mapping name to label
    # idx_to_class = {v: k for k, v in class_to_idx.items()}  # optional: label -> name

    
    # classes = []
     # If you want to get class labels, you need to fill this in. 
                 # e.g. for CIFAR-10, you want classes = ['Airplane', 'Automobile', ...]
    if labels is None:
        print_labels = False

    if scale:
        min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
        # print(f'min_val: {min_val} | max_val: {max_val}')
        data = (data - min_val) / (max_val - min_val)
    else:
        data.mul_(ds).add_(dm).clamp_(0, 1)
    data = data.to(dtype=torch.float32)

    if data.shape[0] == 1:
        plt.axis("off")
        plt.imshow(data[0].permute(1, 2, 0).cpu())
        if print_labels:
            plt.title(f"Data with label {classes[labels]}")
    else:
        n_cols = 8  # you can adjust this
        n_images = data.shape[0]
        n_rows = (n_images + n_cols - 1) // n_cols  # ceil division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
        axes = axes.flatten()  # flatten in case of single row/column

        for i, im in enumerate(data):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
            axes[i].axis("off")
            if labels is not None and print_labels:
                axes[i].set_title(classes[labels[i]], fontsize=8)

        # Turn off any unused axes
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        # grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
        # s = 24 if data.shape[3] > 150 else 6
        # fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
        # label_classes = []
        # for i, (im, axis) in enumerate(zip(data, axes.flatten())):
        #     axis.imshow(im.permute(1, 2, 0).cpu())
        #     if labels is not None and print_labels:
        #         label_classes.append(classes[labels[i]])
        #     axis.axis("off")
        # if print_labels:
        #     print(label_classes)
    # Save figure to PDF if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")

    # plt.close(fig)

def plot_data_select(cfg, user_data, setup, scale=False, print_labels=False,save_path=None):
    """Plot user data to output. Probably best called from a jupyter notebook."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = torch.as_tensor(cfg.mean, **setup)[None, :, None, None].to(device)
    ds = torch.as_tensor(cfg.std, **setup)[None, :, None, None].to(device)

    data = user_data["data"].to(device).clone().detach()
    labels = user_data["labels"].to(device).clone().detach() if user_data["labels"] is not None else None
    
    imagenet_val = datasets.ImageNet(
        root="/home/test/GIA/robbing_the_fed/data/",
        split="val",
        transform=transforms.ToTensor()
    )

    classes = imagenet_val.classes  # <-- class names (e.g., "n01440764", "n01443537", ...)
    # class_to_idx = imagenet_val.class_to_idx  # dict mapping name to label
    # idx_to_class = {v: k for k, v in class_to_idx.items()}  # optional: label -> name

    
    # classes = []
     # If you want to get class labels, you need to fill this in. 
                 # e.g. for CIFAR-10, you want classes = ['Airplane', 'Automobile', ...]
    if labels is None:
        print_labels = False

    if scale:
        min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
        # print(f'min_val: {min_val} | max_val: {max_val}')
        data = (data - min_val) / (max_val - min_val)
    else:
        data.mul_(ds).add_(dm).clamp_(0, 1)
    data = data.to(dtype=torch.float32)

    if data.shape[0] == 1:
        plt.axis("off")
        plt.imshow(data[0].permute(1, 2, 0).cpu())
        if print_labels:
            plt.title(f"Data with label {classes[labels]}")
    else:
        grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
        s = 24 if data.shape[3] > 150 else 6
        fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
        label_classes = []
        for i, (im, axis) in enumerate(zip(data, axes.flatten())):
            axis.imshow(im.permute(1, 2, 0).cpu())
            if labels is not None and print_labels:
                label_classes.append(classes[labels[i]])
            axis.axis("off")
        if print_labels:
            print(label_classes)
    # Save figure to PDF if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")

    plt.close(fig)
           
class data_cfg_default:
    size = (1_281_167,)
    classes = 1000
    shape = (3, 32, 32)
    # shape = (3, 224, 224)
    normalize = True
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)



class attack_cfg_default:
    type = "analytic"
    attack_type = "imprint-readout"
    # label_strategy = "random"  # Labels are not actually required for this attack
    label_strategy = "iDLG"
    normalize_gradients = False
    impl = namedtuple("impl", ["dtype", "mixed_precision", "JIT"])("float", False, "")


