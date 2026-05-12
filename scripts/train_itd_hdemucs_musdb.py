import os
import glob
import torch
from torch.utils.data import DataLoader

from demucs.musdb_binaural_dataset import BinauralMUSDBDataset
from demucs.composite_loss import CompositeLoss
from demucs.hdemucs import HDemucs


def train(
    data_root="/content/binaural_musdb18hq",
    subset="train",
    batch_size=1,
    epochs=1,
    lr=1e-4,
    lambda_itd=0.1,
    segment_seconds=3.0,
    device="cuda",
):
    
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU.")
        device = "cpu"
    device = torch.device(device)

    ckpt_dir = "/content/demucs/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_pattern = os.path.join(ckpt_dir, "hdemucs_itd_musdb_epoch*.pth")
    ckpt_list = sorted(glob.glob(ckpt_pattern))

    if ckpt_list:
        latest_ckpt = ckpt_list[-1]
        print(f"Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        resume_epoch = ckpt["epoch"] + 1
    else:
        print("No checkpoints found — starting fresh.")
        ckpt = None
        resume_epoch = 1

    print(f"Loading dataset from {data_root}, subset={subset}")
    ds = BinauralMUSDBDataset(
        root=data_root,
        subset=subset,
        segment_seconds=segment_seconds,
        sample_rate=44100,
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    print("Building HDemucs model (4 stems, binaural)...")

    model = HDemucs(
        sources=["drums", "bass", "other", "vocals"],
        audio_channels=2,   
        channels=48,        
        depth=5,           
        growth=2,
    ).to(device)

    # If checkpoint exists-load weights
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CompositeLoss(lambda_itd=lambda_itd).to(device)

    if ckpt:
        print("Loading model + optimizer state…")
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])

    model.train()

    for epoch in range(resume_epoch, epochs + 1):

        print(f"\n Starting Epoch {epoch}/{epochs}")
        total_loss = 0.0

        for step, (mix, stems) in enumerate(dl, start=1):

            mix = mix.to(device).float()
            stems = stems.to(device).float()

            est = model(mix)

            # Composite loss per stem
            loss = 0
            for s in range(4):
                loss += criterion(est[:, s], stems[:, s])
            loss /= 4

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"[Epoch {epoch}] Step {step}/{len(dl)} "
                      f"- Loss: {loss.item():.4f}", flush=True)

        mean_loss = total_loss / len(dl)
        print(f"[Epoch {epoch}] Mean Loss = {mean_loss:.4f}")

        ckpt_path = os.path.join(ckpt_dir, f"hdemucs_itd_musdb_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        }, ckpt_path)
        print(f"Saved checkpoint → {ckpt_path}")

    print(f"Training complete. Final model saved at epoch {epochs}.")
    final_path = os.path.join(ckpt_dir, "hdemucs_itd_musdb_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved → {final_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/content/binaural_musdb18hq")
    parser.add_argument("--subset", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_itd", type=float, default=0.1)
    parser.add_argument("--segment_seconds", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        subset=args.subset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_itd=args.lambda_itd,
        segment_seconds=args.segment_seconds,
        device=args.device,
    )
