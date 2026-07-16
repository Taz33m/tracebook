import tracebook


def test_top_level_package_exports_core_api():
    assert tracebook.__version__
    assert tracebook.OrderBook("BTCUSD").symbol == "BTCUSD"
    assert int(tracebook.OrderSide.BUY) == 1
    assert tracebook.Order
    assert tracebook.MarketEvent
    assert tracebook.ReplayTrade
    assert tracebook.load_market_events
    assert tracebook.replay_market_events
    assert tracebook.CampaignProfile
    assert tracebook.CampaignResult
    assert tracebook.run_campaign
    assert tracebook.write_campaign_artifacts
    assert tracebook.run_qualification
    assert tracebook.write_qualification_artifacts
