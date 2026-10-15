"""
Tests for reading time series by when OW stored the samples (``created_after``).

A phone that syncs late delivers samples recorded before others already stored, so a
reader that resumes from the newest recorded time it has seen never gets them. One that
resumes from when it last read does.
"""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.factories import (
    ApiKeyFactory,
    DataPointSeriesFactory,
    DataSourceFactory,
    SeriesTypeDefinitionFactory,
    UserFactory,
)
from tests.utils import api_key_headers

RECORDED = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
READ_AT = datetime(2024, 3, 2, 8, 0, tzinfo=timezone.utc)


class TestTimeseriesCreatedAfter:
    def _samples(self) -> object:
        """Two samples stored before READ_AT, and a late one stored after it but recorded earliest."""
        user = UserFactory()
        watch = DataSourceFactory(user=user)
        heart_rate = SeriesTypeDefinitionFactory.get_or_create_heart_rate()
        for minutes, value, stored in (
            (10, 60, READ_AT - timedelta(hours=1)),
            (20, 70, READ_AT - timedelta(minutes=1)),
            (0, 50, READ_AT + timedelta(minutes=1)),
        ):
            DataPointSeriesFactory(
                mapping=watch,
                series_type=heart_rate,
                recorded_at=RECORDED + timedelta(minutes=minutes),
                value=value,
                created_at=stored,
            )
        return user

    def _get(self, client: TestClient, user: object, **params: str) -> list[float]:
        response = client.get(
            f"/api/v1/users/{user.id}/timeseries",
            params={
                "types": "heart_rate",
                "start_time": (RECORDED - timedelta(days=1)).isoformat(),
                "end_time": (RECORDED + timedelta(days=1)).isoformat(),
                **params,
            },
            headers=api_key_headers(ApiKeyFactory().plain_key),
        )
        assert response.status_code == 200, response.text
        return [sample["value"] for sample in response.json()["data"]]

    def test_returns_only_samples_stored_after_the_instant(self, client: TestClient, db: Session) -> None:
        user = self._samples()

        assert self._get(client, user, created_after=READ_AT.isoformat()) == [50]

    def test_a_late_sample_is_returned_though_recorded_before_the_rest(self, client: TestClient, db: Session) -> None:
        user = self._samples()

        stored_since = self._get(client, user, created_after=(READ_AT - timedelta(minutes=5)).isoformat())

        assert stored_since == [50, 70]

    def test_without_it_every_sample_in_the_window_is_returned(self, client: TestClient, db: Session) -> None:
        user = self._samples()

        assert self._get(client, user) == [50, 60, 70]

    def test_an_unparseable_instant_is_refused(self, client: TestClient, db: Session) -> None:
        user = self._samples()

        response = client.get(
            f"/api/v1/users/{user.id}/timeseries",
            params={
                "types": "heart_rate",
                "start_time": RECORDED.isoformat(),
                "end_time": (RECORDED + timedelta(days=1)).isoformat(),
                "created_after": "yesterday",
            },
            headers=api_key_headers(ApiKeyFactory().plain_key),
        )

        assert response.status_code in (400, 422), response.text
