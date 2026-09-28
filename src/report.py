"""Save CSV log, JSON summary and a bar chart of vehicle counts."""
import csv
import json
from pathlib import Path


def save_reports(counter, out_dir, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "events.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id", "class", "direction"])
        w.writeheader()
        w.writerows(counter.events)

    totals = counter.totals()
    summary = {**meta, "per_class": totals,
               "grand_total": sum(v["total"] for v in totals.values())}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if totals:
            names = list(totals)
            ins = [totals[n]["in"] for n in names]
            outs = [totals[n]["out"] for n in names]
            x = list(range(len(names)))
            plt.figure(figsize=(7, 4))
            plt.bar([i - 0.2 for i in x], ins, 0.4, label="In")
            plt.bar([i + 0.2 for i in x], outs, 0.4, label="Out")
            plt.xticks(x, names)
            plt.ylabel("Vehicles")
            plt.title("Vehicle counts by category")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "counts.png", dpi=150)
            plt.close()
    except ImportError:
        pass
    return summary
