from __future__ import annotations

import platform
import sys


def main() -> int:
    print("=" * 70)
    print("OurAI self-hosted GPU runner smoke test")
    print("=" * 70)
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())

    try:
        import torch
    except Exception as exc:
        print("ERROR: PyTorch could not be imported:", repr(exc))
        return 2

    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available to the runner process.")
        return 3

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print("GPU:", props.name)
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")

    # Small real CUDA computation. This is intentionally tiny so the smoke
    # test is safe to run even on the validated GTX 1650 4 GB laptop GPU.
    x = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()

    checksum = float(y.float().mean().item())
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2

    print(f"CUDA matmul checksum: {checksum:.6f}")
    print(f"GPU memory allocated: {allocated:.1f} MiB")
    print(f"GPU memory reserved: {reserved:.1f} MiB")
    print("SUCCESS: self-hosted runner can execute PyTorch CUDA workloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
