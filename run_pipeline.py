"""One-click reproduction pipeline for Hiver Take-Home Assignment.

Reproduces all headline numbers, baselines, and evaluation harness metrics
on the 200-sample hand-labelled Golden Evaluation Set in under 2 minutes.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluate import run_benchmark


def main():
    start_time = time.time()
    print("=" * 75)
    print(" HIVER SDE INTERN ASSIGNMENT — AI SUPPORT AGENT FOR @AppleSupport")
    print(" Pipeline: Intent Classification + Historical RAG Grounding + Escalation")
    print("=" * 75)
    print(f"Working Directory: {BASE_DIR}")
    print("Reproducing headline benchmark results...\n")

    results = run_benchmark()

    elapsed = time.time() - start_time
    print(f"\n[OK] Headline results successfully reproduced in {elapsed:.2f} seconds!")
    print("Artifacts generated:")
    print(f"  - Table:      {BASE_DIR / 'artifacts' / 'headline_results_table.md'}")
    print(f"  - Full JSON:  {BASE_DIR / 'artifacts' / 'benchmark_results.json'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
