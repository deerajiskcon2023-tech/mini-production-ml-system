"""Benchmark /predict latency using at least 100 sequential requests."""
import argparse
import json
import statistics
import time
import requests

def percentile(values, p):
    return float(sorted(values)[round((len(values) - 1) * p)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--payload", default="tests/sample_request.json")
    args = parser.parse_args()
    if args.requests < 100:
        raise ValueError("Use at least 100 requests for the assignment benchmark.")
    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)
    session = requests.Session()
    # Warm-up requests are excluded from measured latency.
    for _ in range(10):
        response = session.post(args.url, json=payload, timeout=10)
        response.raise_for_status()
    latencies_ms = []
    for _ in range(args.requests):
        start = time.perf_counter()
        response = session.post(args.url, json=payload, timeout=10)
        elapsed = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        latencies_ms.append(elapsed)
    total_seconds = sum(latencies_ms) / 1000
    report = {
        "requests_measured": args.requests,
        "p50_latency_ms": percentile(latencies_ms, .50),
        "p95_latency_ms": percentile(latencies_ms, .95),
        "p99_latency_ms": percentile(latencies_ms, .99),
        "mean_latency_ms": statistics.mean(latencies_ms),
        "throughput_requests_per_second": args.requests / total_seconds if total_seconds else None,
        "note": "Measured from the client process against the locally running API; includes HTTP overhead."
    }
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
