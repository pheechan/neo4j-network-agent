import requests


def test_chat_smoke():
    """Simple smoke test: POST to /api/chat and expect a JSON response with 'answer'"""
    url = "http://localhost:8000/api/chat"
    payload = {"message": "Who is in the Santisook network?"}
    try:
        r = requests.post(url, json=payload, timeout=10)
    except Exception as e:
        raise AssertionError(f"Could not reach backend at {url}: {e}")

    assert r.status_code == 200, f"Unexpected status code: {r.status_code} - {r.text}"
    j = r.json()
    assert 'answer' in j, f"Response JSON missing 'answer' field: {j}"
