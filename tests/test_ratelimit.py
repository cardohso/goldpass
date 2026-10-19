import threading
import time

from goldpass.ratelimit import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []
        self.lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self.lock:
            self.sleeps.append(round(seconds, 6))
            self.now += seconds


def limiter(interval: float, clock: FakeClock) -> RateLimiter:
    return RateLimiter(interval, clock=clock, sleep=clock.sleep)


def test_first_call_is_free_then_calls_are_spaced() -> None:
    clock = FakeClock()
    rl = limiter(12.0, clock)
    for _ in range(3):
        rl.wait()
    assert clock.sleeps == [12.0, 12.0]


def test_time_already_spent_counts() -> None:
    clock = FakeClock()
    rl = limiter(12.0, clock)
    rl.wait()
    clock.now += 5.0  # the request itself took 5 s
    rl.wait()
    assert clock.sleeps == [7.0]


def test_no_interval_never_sleeps() -> None:
    clock = FakeClock()
    rl = limiter(0.0, clock)
    for _ in range(5):
        rl.wait()
    assert clock.sleeps == []


def test_defer_pushes_everyone_back() -> None:
    clock = FakeClock()
    rl = limiter(0.0, clock)
    rl.wait()
    rl.defer(30.0)  # one worker was told Retry-After: 30
    rl.wait()  # another worker must wait too
    assert clock.sleeps == [30.0]
    rl.defer(5.0)
    rl.defer(2.0)  # a shorter deferral never shortens an existing one
    rl.wait()
    assert clock.sleeps == [30.0, 5.0]


def test_waiters_get_distinct_slots_under_real_threads() -> None:
    """8 threads at 20 ms spacing: starts are spread over at least 7 intervals."""
    rl = RateLimiter(0.02)
    starts: list[float] = []
    lock = threading.Lock()

    def work() -> None:
        rl.wait()
        with lock:
            starts.append(time.monotonic())

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    starts.sort()
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert min(gaps) > 0.012  # none start together (timer slack allowed)
    assert starts[-1] - starts[0] >= 0.13
