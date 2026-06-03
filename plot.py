import torch
import matplotlib.pyplot as plt

def plot_results():
    # Load samples
    encoded = torch.load("sample_encoded.pt")
    target = torch.load("sample_target.pt")
    pred = torch.load("sample_pred.pt")

    seq_len = target.shape[0]
    time_steps = range(seq_len)

    plt.figure(figsize=(15, 10))

    # Plot 1: The generated 2D O-U process (Target)
    plt.subplot(3, 1, 1)
    plt.title("Generated 2D O-U Process (Target Trajectory)")
    plt.plot(time_steps, target[:, 0].numpy(), label="X1 (Target)", color="blue")
    plt.plot(time_steps, target[:, 1].numpy(), label="X2 (Target)", color="green")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    # Plot 2: The raw 6-bit "zebra plot" (imshow)
    plt.subplot(3, 1, 2)
    plt.title("Raw 6-bit Quantum Reservoir Output ('Zebra Plot')")
    # Transpose so time is on x-axis, qubits on y-axis
    plt.imshow(encoded.numpy().T, aspect="auto", cmap="binary", origin="upper")
    plt.xlabel("Time Step")
    plt.ylabel("Measured Qubit Index (0-5)")
    plt.yticks(range(6))

    # Plot 3: Reconstructed vs Target Trajectory
    plt.subplot(3, 1, 3)
    plt.title("Reconstructed vs Target Trajectory")
    plt.plot(time_steps, target[:, 0].numpy(), label="X1 (Target)", color="blue", linestyle="--", alpha=0.7)
    plt.plot(time_steps, pred[:, 0].numpy(), label="X1 (Predicted)", color="darkblue")

    plt.plot(time_steps, target[:, 1].numpy(), label="X2 (Target)", color="green", linestyle="--", alpha=0.7)
    plt.plot(time_steps, pred[:, 1].numpy(), label="X2 (Predicted)", color="darkgreen")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("stoqastiq_results.png")
    print("Plots saved to stoqastiq_results.png")

if __name__ == "__main__":
    plot_results()
