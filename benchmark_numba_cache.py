"""
Benchmark Numba cache=True vs cache=False when workers are constantly restarted
(like your pool with maxtasksperchild=1 or 50000).

Uses a Pool with maxtasksperchild=1 so every task runs in a fresh worker.
Compares total time for many tasks: with cache (workers load from disk) vs
without cache (workers recompile every time).

Run: python benchmark_numba_cache.py
"""

import multiprocessing
import os
import time
import numba
import numpy as np

# Config: bigger = longer run, more worker restarts
N_WORKERS = min(20, os.cpu_count() or 4)
N_TASKS = 200
ARRAY_SHAPE = (1000, 1000)  # per-task workload
CALLS_PER_TASK = 1       # Numba calls per task (first = compile/load, rest = execution)

# Two versions at module level so pool workers see them when they import this file
@numba.njit(fastmath=True, cache=True)
def add_arrays_cached(a, b):
    c = np.empty_like(a)
    for i in range(a.size):
        c.flat[i] = a.flat[i] + b.flat[i]
    return c


@numba.njit(fastmath=True, cache=False)
def add_arrays_uncached(a, b):
    c = np.empty_like(a)
    for i in range(a.size):
        c.flat[i] = a.flat[i] + b.flat[i]
    return c


def run_one_task(use_cached: bool, calls_per_task: int):
    """Run in a pool worker. First call = compile or load; then (calls_per_task - 1) more calls."""
    np.random.seed(42)
    a = np.random.rand(*ARRAY_SHAPE)
    b = np.random.rand(*ARRAY_SHAPE)
    n = calls_per_task
    if use_cached:
        for _ in range(n):
            add_arrays_cached(a, b)
    else:
        for _ in range(n):
            add_arrays_uncached(a, b)


def run_bench(pool, n_tasks: int, calls_per_task: int, use_cached: bool) -> float:
    """Run n_tasks with given calls_per_task; return total time."""
    t0 = time.perf_counter()
    # Pass calls_per_task via a tuple (worker unpacks it)
    pool.starmap(
        run_one_task,
        [(use_cached, calls_per_task)] * n_tasks,
        chunksize=1,
    )
    return time.perf_counter() - t0


def main():
    print("Numba cache benchmark: pool with worker recycling (maxtasksperchild=1)")
    print(f"  Workers: {N_WORKERS}, Tasks: {N_TASKS}\n")

    # Populate cache in main process so workers can load from disk
    print("Populating cache (one call in main process)...")
    a0 = np.random.rand(*ARRAY_SHAPE)
    b0 = np.random.rand(*ARRAY_SHAPE)
    add_arrays_cached(a0, b0)
    print("  Done.\n")

    with multiprocessing.Pool(processes=N_WORKERS, maxtasksperchild=1) as pool:
        # --- Light workload: 1 call per task (cache overhead dominates) ---
        print("--- Light workload (1 Numba call per task) ---")
        print("  -> Most time is compile vs load; cache benefit should be large.\n")
        t_light_cached = run_bench(pool, N_TASKS, 1, use_cached=True)
        t_light_uncached = run_bench(pool, N_TASKS, 1, use_cached=False)
        print(f"  cache=True:  {t_light_cached:.2f} s")
        print(f"  cache=False: {t_light_uncached:.2f} s")
        if t_light_cached > 0:
            print(f"  -> cache=True is {t_light_uncached / t_light_cached:.1f}x faster.\n")

        # --- Heavy workload: many calls per task (execution dominates) ---
        print("--- Heavy workload ({0} calls per task) ---".format(CALLS_PER_TASK))
        print("  -> Most time is execution (same either way); cache benefit is small.\n")
        print("Running cache=True...")
        t_heavy_cached = run_bench(pool, N_TASKS, CALLS_PER_TASK, use_cached=True)
        print(f"  Total time: {t_heavy_cached:.2f} s\n")
        print("Running cache=False...")
        t_heavy_uncached = run_bench(pool, N_TASKS, CALLS_PER_TASK, use_cached=False)
        print(f"  Total time: {t_heavy_uncached:.2f} s\n")

    print("--- Summary ---")
    print("Light (1 call/task):  cache matters a lot (each worker: compile vs load).")
    print("Heavy (many calls):   cache matters little (most time is running the same code).")
    if t_light_cached > 0 and t_heavy_cached > 0:
        print(f"  Light speedup: {t_light_uncached / t_light_cached:.1f}x")
        print(f"  Heavy speedup: {t_heavy_uncached / t_heavy_cached:.1f}x")


if __name__ == "__main__":
    main()
