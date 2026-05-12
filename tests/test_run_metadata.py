from core.run_metadata import run_metadata


def test_run_metadata_redacts_api_keys():
    payload = run_metadata()
    assert "api_key" not in str(payload).lower()
    assert "package_versions" in payload
    assert "settings" in payload
