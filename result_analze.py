import json
import os
import sys

# import evaluate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="bleurt-large-512")
#


def load_results(experiment_folder_path: str):
    result_files = os.listdir(experiment_folder_path)
    results = {}
    for result_file in result_files:
        if result_file.startswith("experiment_metadata"):
            continue
        if result_file.endswith(".json"):
            with open(os.path.join(experiment_folder_path, result_file), "r") as f:
                result = json.load(f)
            results[result_file] = result
    return results


# def evaluate_response(request_metrics: list[dict]) -> dict[str, list[float]]:
#     references = [request_metric["answer_text"] for request_metric in request_metrics]
#     predictions = [
#         request_metric["generated_text"] for request_metric in request_metrics
#     ]
#     scores = bleurt.compute(references=references, predictions=predictions)
#     return scores


def _plot_cdf_with_percentiles(
    data: list[float],
    metric_label: str,
    title: str = None,
    save_path: str = None,
    color: str = "b",
):
    """
    Helper function to plot CDF and annotate percentiles.
    """
    if not data:
        print(f"No data provided for {metric_label}")
        return

    # Setup plot
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=data, color=color, label=metric_label)

    # Calculate percentiles
    percentiles = [50, 90, 95, 99]
    p_values = np.percentile(data, percentiles)

    # Plot vertical lines for percentiles
    colors = ["green", "orange", "red", "purple"]
    for p, val, c in zip(percentiles, p_values, colors):
        plt.axvline(x=val, color=c, linestyle="--", alpha=0.7, label=f"P{p}: {val:.4f}")
        # Add text annotation near the top or slightly offset
        plt.text(
            val,
            0.05,
            f"P{p}\n{val:.2f}",
            rotation=90,
            verticalalignment="bottom",
            color=c,
            fontweight="bold",
        )

    if title:
        plt.title(f"{title}\nCDF of {metric_label}")
    else:
        plt.title(f"CDF of {metric_label}")
    plt.xlabel(f"{metric_label} (s)")
    plt.ylabel("Cumulative Probability")
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def ttft_cdf(request_metrics: list[dict]):
    """
    Extract TTFT values.
    """
    return [m["ttft"] for m in request_metrics if m.get("ttft") is not None]


def plot_ttft_cdf(
    request_metrics: list[dict], title: str = None, save_path: str = None
):
    """
    plot the CDF of the TTFT with range between [min, max] and indicate the value of p50,p90, p95, p99.
    """
    data = ttft_cdf(request_metrics)
    _plot_cdf_with_percentiles(
        data,
        "Time To First Token (TTFT)",
        title=title,
        save_path=save_path,
        color="blue",
    )


def plot_e2e_latency_cdf(
    request_metrics: list[dict], title: str = None, save_path: str = None
):
    """
    plot the CDF of the E2E latency with range between [min, max] and indicate the value of p50,p90, p95, p99.
    """
    data = [
        m["e2e_latency"] for m in request_metrics if m.get("e2e_latency") is not None
    ]
    _plot_cdf_with_percentiles(
        data, "End-to-End Latency", title=title, save_path=save_path, color="darkcyan"
    )


def main():
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # Default path for demonstration if not provided
        folder_path = "/storage/ice1/5/0/hchang367/research3/experiments/sharegpt4o_image_caption/no_compression_3"

    print(f"Processing results in: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    results = load_results(folder_path)

    if not results:
        print("No result files found.")
        return

    for filename, data in results.items():
        if "individual_request_metrics" in data:
            metrics = data["individual_request_metrics"]

            # Construct base output filename from JSON filename
            base_name = os.path.splitext(filename)[0]

            # Plot TTFT
            ttft_filename = f"{base_name}_ttft_cdf.png"
            ttft_path = os.path.join(folder_path, ttft_filename)
            plot_ttft_cdf(metrics, title=filename, save_path=ttft_path)

            # Plot E2E Latency
            e2e_filename = f"{base_name}_e2e_cdf.png"
            e2e_path = os.path.join(folder_path, e2e_filename)
            plot_e2e_latency_cdf(metrics, title=filename, save_path=e2e_path)
        else:
            print(f"Skipping {filename}: 'individual_request_metrics' not found.")


if __name__ == "__main__":
    main()
