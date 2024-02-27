import torch
from rich import get_console, print

console = get_console()


def print_torch_info() -> None:
    """Print info about Torch version and GPU availability."""
    console.rule("[purple]TORCH DEVICE INFO", style="green_yellow")

    # Check if GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Create a list of GPU devices
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]

        # Display GPU information
        print(f"[purple]{num_gpus}[/purple] GPU(s) are available!")
        for i, device in enumerate(devices):
            print(f"GPU {i}: [purple]{torch.cuda.get_device_name(i)}[/purple]")

        print(f"Using GPUs: [purple]{', '.join([str(i) for i in range(num_gpus)])}[/purple]")
    else:
        # If GPU is not available, fall back to CPU
        device = torch.device("cpu")
        print("GPU is not available. Falling back to CPU.")

    # Display general PyTorch version and device information
    print(f"PyTorch version: [purple]{torch.__version__}[/purple]")
    print(f"Using device: [purple]{device}[/purple]")

    # Additional information if GPU is available
    if torch.cuda.is_available():
        print(f"CUDA version: [purple]{torch.version.cuda}[/purple]")
        print(f"CUDNN version: [purple]{torch.backends.cudnn.version()}[/purple]")

    console.rule("", style="green_yellow")


if __name__ == "__main__":
    print_torch_info()
