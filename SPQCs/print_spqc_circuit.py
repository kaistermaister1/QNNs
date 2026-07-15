"""
Utility script to construct an SPQC circuit with
parameters t=2, r=2, m=3, n=2 using the builder in `star_spqc.py`,
render it nicely, and save it to an image file.
"""

from pathlib import Path
import matplotlib

# Use a non-interactive backend suitable for servers/CLI runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

from star_spqc import create_spqc_circuit  # noqa: E402


def main() -> None:
    qc = create_spqc_circuit(t=2, m=3, n=2, r=2)

    # Draw using matplotlib with a clean preset style and modest folding
    fig = qc.draw(output="mpl", style="iqx", fold=120)

    out_path = Path(__file__).with_name("spqc_t2_r2_m3_n2.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved circuit figure to {out_path}")


if __name__ == "__main__":
    main()


