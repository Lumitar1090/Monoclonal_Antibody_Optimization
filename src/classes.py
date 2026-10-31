from pathlib import Path

class BatchResultsAnalyzer:
    def __init__(self, output_directory):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def save_top_batches(self, optimization_results, n=10):
        top_batches = optimization_results.sort_values(
            "Space-Time Profit [$/L/d]",
            ascending=False,).head(n)

        output_path = self.output_directory / "Top_Batches.csv"
        top_batches.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")
        return top_batches


