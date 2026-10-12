import torch

from ai.config import ModelConfig
from ai.model import OurAI


device = "cuda" if torch.cuda.is_available() else "cpu"

config = ModelConfig()

model = OurAI(config).to(device)

num_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("=" * 50)
print("OUR AI v0.1")
print("=" * 50)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print(f"Parameters: {num_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

input_ids = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 128),
    device=device
)

targets = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(2, 128),
    device=device
)

print("\nRunning forward pass...")

logits, loss = model(
    input_ids,
    targets
)

print("Input shape:", input_ids.shape)
print("Output shape:", logits.shape)
print("Loss:", loss.item())

loss.backward()

print("Backward pass: OK")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"GPU memory allocated: {allocated:.1f} MB")
    print(f"GPU memory reserved: {reserved:.1f} MB")

print("\nSUCCESS")
print("Our Transformer is running.")