"""
Test: Compare sync polling vs async polling time consumption.
Scenario: PDF parsing polling, 5 polls with 2-second intervals each.
"""
import time
import asyncio

# ============ Simulation parameters ============
POLL_INTERVAL = 2  # polling interval (seconds)
POLL_COUNT = 5      # number of polls
TASK_COUNT = 3      # concurrent task count
# ===============================================


def sync_polling(task_id: int) -> dict:
    """Sync polling: simulates PDF parsing polling."""
    start = time.perf_counter()
    for i in range(POLL_COUNT):
        # Simulate API request (assumed 0.5 seconds)
        time.sleep(0.5)
        # Simulate polling interval
        time.sleep(POLL_INTERVAL)
    elapsed = time.perf_counter() - start
    return {"task_id": task_id, "elapsed": elapsed, "type": "sync"}


async def async_polling(task_id: int) -> dict:
    """Async polling: simulates PDF parsing polling."""
    start = time.perf_counter()
    for i in range(POLL_COUNT):
        # Simulate async API request (0.5 seconds, non-blocking)
        await asyncio.sleep(0.5)
        # Simulate polling interval (non-blocking)
        await asyncio.sleep(POLL_INTERVAL)
    elapsed = time.perf_counter() - start
    return {"task_id": task_id, "elapsed": elapsed, "type": "async"}


def test_sync_polling():
    """Test sync polling."""
    print("=" * 60)
    print(f"Sync polling test: {TASK_COUNT} tasks, {POLL_COUNT} polls each, {POLL_INTERVAL}s interval")
    print("=" * 60)

    start = time.perf_counter()
    results = []
    for i in range(TASK_COUNT):
        result = sync_polling(i)
        results.append(result)
        print(f"Task {i}: done, took {result['elapsed']:.2f}s")
    total = time.perf_counter() - start

    print(f"\nTotal time: {total:.2f}s")
    print(f"Theoretical time: {TASK_COUNT} × {POLL_COUNT} × ({POLL_INTERVAL}s interval + 0.5s request) = {TASK_COUNT * POLL_COUNT * (POLL_INTERVAL + 0.5):.2f}s")
    print(f"Actual serial time: {total:.2f}s")
    return total


async def test_async_polling():
    """Test async polling."""
    print("\n" + "=" * 60)
    print(f"Async polling test: {TASK_COUNT} concurrent tasks, {POLL_COUNT} polls each, {POLL_INTERVAL}s interval")
    print("=" * 60)

    start = time.perf_counter()
    # Create multiple concurrent coroutines
    tasks = [async_polling(i) for i in range(TASK_COUNT)]
    results = await asyncio.gather(*tasks)
    total = time.perf_counter() - start

    for result in results:
        print(f"Task {result['task_id']}: done, took {result['elapsed']:.2f}s")

    print(f"\nTotal time: {total:.2f}s")
    print(f"Theoretical time: {POLL_COUNT} × ({POLL_INTERVAL}s interval + 0.5s request) = {POLL_COUNT * (POLL_INTERVAL + 0.5):.2f}s (concurrent)")
    return total


async def main():
    print("\n" + "=" * 60)
    print("PDF parsing polling: sync vs async time comparison")
    print("=" * 60)
    print(f"Polling interval: {POLL_INTERVAL} seconds")
    print(f"Polling count: {POLL_COUNT}")
    print(f"Concurrent tasks: {TASK_COUNT}\n")

    # Sync polling
    sync_total = test_sync_polling()

    # Async polling
    async_total = await test_async_polling()

    # Comparison
    print("\n" + "=" * 60)
    print("Comparison results")
    print("=" * 60)
    print(f"Sync polling total time: {sync_total:.2f}s")
    print(f"Async polling total time: {async_total:.2f}s")
    print(f"Time saved: {sync_total - async_total:.2f}s")
    print(f"Performance improvement: {(sync_total - async_total) / sync_total * 100:.1f}%")
    print(f"\nConclusion:")
    print(f"  - With {TASK_COUNT} tasks, sync runs serially and takes longer")
    print(f"  - Async runs concurrently, {TASK_COUNT} tasks poll simultaneously, time close to single task")
    print(f"  - More tasks means greater async advantage")


if __name__ == "__main__":
    asyncio.run(main())
