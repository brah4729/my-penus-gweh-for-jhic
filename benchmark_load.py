"""
Concurrent load test against the RAG API -- measures real throughput under
the same kind of concurrent load your competition stress test will apply,
not just single-request speed.

Usage:
    uv run python benchmark_load.py --concurrency 2 --requests 10
    uv run python benchmark_load.py --url http://<VPS_IP>:8000 --api-key XXX --concurrency 2 --requests 10
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TEST_QUESTIONS = [
    "Siapa yang mengajar matematika?",
    "Apa saja ekstrakurikuler di sekolah?",
    "Apa prestasi yang pernah diraih sekolah?",
    "Siapa saja mitra industri sekolah?",
    "Apa berita terbaru dari sekolah?",
]


def send_request(url: str, api_key: str | None, question: str):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    start = time.perf_counter()
    try:
        resp = requests.post(f"{url}/chat", json={"question": question}, headers=headers, timeout=120)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
        return {
            "ok": True,
            "elapsed": elapsed,
            "completion_tokens": data.get("completion_tokens"),
            "had_sources": len(data.get("sources", [])) > 0,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"ok": False, "elapsed": elapsed, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--concurrency", type=int, default=2, help="simultaneous requests in flight")
    parser.add_argument("--requests", type=int, default=10, help="total requests to send")
    args = parser.parse_args()

    questions = [TEST_QUESTIONS[i % len(TEST_QUESTIONS)] for i in range(args.requests)]

    print(f"Sending {args.requests} requests, {args.concurrency} concurrent, to {args.url}")
    print("(watch your llama-server terminal alongside this for per-request tg t/s)\n")

    wall_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send_request, args.url, args.api_key, q) for q in questions]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "OK" if result["ok"] else f"FAIL ({result.get('error')})"
            print(f"  [{status}] {result['elapsed']:.2f}s" +
                  (f", {result['completion_tokens']} tokens" if result.get("completion_tokens") else ""))

    wall_elapsed = time.perf_counter() - wall_start

    ok_results = [r for r in results if r["ok"]]
    failed = len(results) - len(ok_results)
    total_tokens = sum(r["completion_tokens"] or 0 for r in ok_results)
    avg_latency = sum(r["elapsed"] for r in ok_results) / len(ok_results) if ok_results else 0

    print("\n--- Summary ---")
    print(f"Total wall-clock time     : {wall_elapsed:.2f}s")
    print(f"Requests: {len(results)} sent, {len(ok_results)} succeeded, {failed} failed")
    print(f"Average per-request time  : {avg_latency:.2f}s")
    if total_tokens > 0:
        print(f"Total completion tokens   : {total_tokens}")
        print(f"Aggregate throughput      : {total_tokens / wall_elapsed:.2f} tokens/sec (across all concurrent requests)")
    else:
        print("No completion tokens recorded (all requests may have hit the no-context fallback -- try again with on-topic questions)")


if __name__ == "__main__":
    main()
