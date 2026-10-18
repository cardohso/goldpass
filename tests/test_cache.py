from pathlib import Path

from goldpass.cache import DiskCache, NoCache, cache_key


def test_key_is_stable_and_order_independent() -> None:
    assert cache_key({"a": 1, "b": [1, 2]}) == cache_key({"b": [1, 2], "a": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})


def test_disk_cache_persists(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "judge.sqlite"
    c = DiskCache(path)
    assert c.get("k") is None
    c.set("k", '{"score": 5}')
    assert c.get("k") == '{"score": 5}'
    assert (c.hits, c.misses) == (1, 1)
    c.close()
    again = DiskCache(path)
    assert again.get("k") == '{"score": 5}'


def test_no_cache() -> None:
    c = NoCache()
    c.set("k", "v")
    assert c.get("k") is None and c.misses == 1
