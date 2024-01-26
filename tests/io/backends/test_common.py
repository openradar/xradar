from xradar.io.backends import common


def test_lazy_dict():
    d = common.LazyLoadDict({"key1": "value1", "key2": "value2"})
    assert d["key1"] == "value1"
    lazy_func = lambda: 999
    d.set_lazy("lazykey1", lazy_func)
    assert d["lazykey1"] == 999
