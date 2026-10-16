"""Retries in make_authenticated_request for a provider's passing server errors."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.services.providers.api_client import MAX_RETRIES, make_authenticated_request

URL = "https://api.example.com/v1/points"


def _response(status: int, method: str = "GET") -> httpx.Response:
    return httpx.Response(status, json={"ok": status == 200}, request=httpx.Request(method, URL))


def _request(responses: list[httpx.Response], method: str = "GET") -> tuple[object, MagicMock, MagicMock]:
    """Run one request against ``responses`` in turn; return its result (or error) and the mocks."""
    client = MagicMock()
    client.__enter__.return_value.request.side_effect = responses
    with (
        patch("app.services.providers.api_client._get_valid_token", return_value="token"),
        patch("app.services.providers.api_client.httpx.Client", return_value=client),
        patch("app.services.providers.api_client.time.sleep") as sleep,
    ):
        try:
            result: object = make_authenticated_request(
                db=MagicMock(),
                user_id=uuid4(),
                connection_repo=MagicMock(),
                oauth=MagicMock(),
                api_base_url="https://api.example.com",
                provider_name="google_health",
                endpoint="/v1/points",
                method=method,
            )
        except HTTPException as e:
            result = e
    return result, client.__enter__.return_value.request, sleep


@pytest.mark.parametrize("status", [502, 503, 504])
def test_a_get_is_retried_through_a_passing_server_error(status: int) -> None:
    result, request, sleep = _request([_response(status), _response(status), _response(200)])

    assert result == {"ok": True}
    assert request.call_count == 3
    assert sleep.call_count == 2


def test_a_get_gives_up_after_the_last_retry() -> None:
    result, request, _ = _request([_response(503)] * (MAX_RETRIES + 1))

    assert isinstance(result, HTTPException)
    assert result.status_code == 503
    assert request.call_count == MAX_RETRIES + 1


def test_a_post_is_not_sent_twice() -> None:
    result, request, sleep = _request([_response(503, "POST"), _response(200, "POST")], method="POST")

    assert isinstance(result, HTTPException)
    assert result.status_code == 503
    assert request.call_count == 1
    sleep.assert_not_called()


def test_other_server_errors_are_not_retried() -> None:
    result, request, _ = _request([_response(500), _response(200)])

    assert isinstance(result, HTTPException)
    assert result.status_code == 500
    assert request.call_count == 1
