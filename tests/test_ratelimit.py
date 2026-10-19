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


def test_real_threads_never_start_faster_than_the_limit() -> None:
    """The guarantee is a floor, not even spacing: a loaded machine may oversleep one thread
    and release the next right after it. So the k-th start (in time order) can never come
    before t0 + k * interval."""
    interval = 0.02
    rl = RateLimiter(interval)
    starts: list[float] = []
    lock = threading.Lock()

    def work() -> None:
        rl.wait()
        with lock:
            starts.append(time.monotonic())

    threads = [threading.Thread(target=work) for _ in range(8)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    starts.sort()
    slack = 0.005
    assert [k for k, s in enumerate(starts) if s < t0 + k * interval - slack] == []
    assert starts[-1] - t0 >= 7 * interval - slack  # so 8 calls take at least 7 intervals
