"""Bounded thread pool: timeout is measured from when a task starts, not submit time."""

from __future__ import annotations

import concurrent.futures
import sys
import time
from typing import Callable, List, Optional, Sequence, TypeVar

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

T = TypeVar("T")
R = TypeVar("R")


def map_in_flight(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    max_workers: int,
    timeout: float,
    on_timeout: Callable[[T], R],
    on_error: Callable[[T, BaseException], R],
    desc: str = "eval",
) -> List[R]:
    results: List[Optional[R]] = [None] * len(items)
    if not items:
        return []
    workers = max(1, int(max_workers))
    next_i = 0
    in_flight: dict = {}
    started: dict = {}

    def submit(executor: concurrent.futures.Executor, i: int) -> None:
        future = executor.submit(fn, items[i])
        in_flight[future] = i
        started[future] = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while next_i < len(items) and len(in_flight) < workers:
            submit(executor, next_i)
            next_i += 1
        bar = (
            tqdm(total=len(items), desc=desc, unit="qa", file=sys.stdout)
            if tqdm is not None
            else None
        )
        try:
            while in_flight:
                done, _ = concurrent.futures.wait(
                    list(in_flight),
                    timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    idx = in_flight.pop(future)
                    started.pop(future, None)
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        results[idx] = on_error(items[idx], exc)
                    if bar:
                        bar.update(1)
                    if next_i < len(items):
                        submit(executor, next_i)
                        next_i += 1
                if timeout <= 0:
                    continue
                now = time.monotonic()
                for future in list(in_flight):
                    if (now - started.get(future, now)) <= timeout:
                        continue
                    future.cancel()
                    idx = in_flight.pop(future)
                    started.pop(future, None)
                    results[idx] = on_timeout(items[idx])
                    if bar:
                        bar.update(1)
                    if next_i < len(items):
                        submit(executor, next_i)
                        next_i += 1
        finally:
            if bar:
                bar.close()

    out: List[R] = []
    for idx, row in enumerate(results):
        if row is None:
            out.append(on_error(items[idx], RuntimeError("missing")))
        else:
            out.append(row)
    return out
