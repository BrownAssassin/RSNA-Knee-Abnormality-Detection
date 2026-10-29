from monitor_logs import event_payload


def test_live_event_payload():
    assert event_payload('data: {"data":"progress", "time":4}') == {'data':'progress','time':4}
    assert event_payload('data: END_OF_LOG') is None
    assert event_payload(': heartbeat') is None


def test_non_json_log_line():
    assert event_payload('data: trace line') == {'data':'trace line'}
