import threading
import time

import pytest

from goldpass.parallel import ordered_map


def test_results_keep_input_order_whatever_finishes_first() -> None:
    def slow_first(n: int) -> int:
        time.sleep(0.05 if n == 0 else 0.0)
        return n * n

    assert ordered_map(slow_first, list(range(6)), workers=4) == [0, 1, 4, 9, 16, 25]


def test_one_worker_is_a_plain_loop_on_the_calling_thread() -> None:
    seen: set[int] = set()
    ordered_map(lambda n: seen.add(threading.get_ident()), [1, 2, 3], workers=1)
    assert seen == {threading.get_ident()}


def test_workers_really_overlap() -> None:
    running = peak = 0
    lock = threading.Lock()

    def work(n: int) -> int:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return n

    ordered_map(work, list(range(8)), workers=4)
    assert 2 <= peak <= 4  # concurrent, and never more than asked for


def test_exceptions_propagate() -> None:
    def boom(n: int) -> int:
        raise ValueError(n)

    with pytest.raises(ValueError):
        ordered_map(boom, [1, 2], workers=2)
