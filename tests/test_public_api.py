import tracebook


def test_top_level_package_exports_core_api():
    assert tracebook.__version__
    assert tracebook.OrderBook("BTCUSD").symbol == "BTCUSD"
    assert int(tracebook.OrderSide.BUY) == 1
