"""
Tests for Apple HealthKit sleep service processing.

Tests the sleep pipeline (handle_sleep_data, _apply_transition, _calculate_final_metrics,
finish_sleep) using synthetic payloads modeled after real Apple HealthKit SDK data.

Apple Watch sleep data patterns:
- Older Apple Watch (pre-watchOS 9): only "in_bed" and "sleeping" stages
- Newer Apple Watch (watchOS 9+): "in_bed", "awake", "light", "deep", "rem" stages
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.constants.sleep import SleepStageType
from app.integrations.redis_client import get_redis_client
from app.models import DataSource, EventRecord
from app.schemas.model_crud.activities import (
    EventRecordCreate,
    EventRecordDetailCreate,
)
from app.schemas.providers.mobile_sdk import (
    SleepState,
    SleepStateStage,
    SyncRequest,
)
from app.services.sdk.sleep_service import (
    _calculate_final_metrics,
    finish_sleep,
    handle_sleep_data,
    key,
    load_sleep_state,
    save_sleep_state,
    writer_id,
    writers_key,
)
from tests.factories import UserFactory


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Synthetic payload: older Apple Watch (pre-watchOS 9) with sleeping + in_bed
# Mimics pattern: Watch sends "sleeping" segments, iPhone sends "in_bed"
# ---------------------------------------------------------------------------
OLD_WATCH_PAYLOAD = {
    "provider": "apple",
    "sdkVersion": "0.5.0",
    "syncTimestamp": "2026-03-11T13:28:04Z",
    "data": {
        "records": [],
        "workouts": [],
        "sleep": [
            {
                "id": "aaaa1111-0000-0000-0000-000000000001",
                "parentId": None,
                "stage": "sleeping",
                "startDate": "2026-03-10T23:00:00Z",
                "endDate": "2026-03-10T23:50:00Z",
                "source": {
                    "device_type": "watch",
                    "device_model": "Watch3,3",
                },
            },
            {
                "id": "aaaa1111-0000-0000-0000-000000000002",
                "parentId": None,
                "stage": "sleeping",
                "startDate": "2026-03-10T23:52:00Z",
                "endDate": "2026-03-11T00:45:00Z",
                "source": {
                    "device_type": "watch",
                    "device_model": "Watch3,3",
                },
            },
            {
                "id": "aaaa1111-0000-0000-0000-000000000003",
                "parentId": None,
                "stage": "sleeping",
                "startDate": "2026-03-11T00:47:00Z",
                "endDate": "2026-03-11T01:30:00Z",
                "source": {
                    "device_type": "watch",
                    "device_model": "Watch3,3",
                },
            },
            {
                "id": "aaaa1111-0000-0000-0000-000000000004",
                "parentId": None,
                "stage": "in_bed",
                "startDate": "2026-03-10T22:55:00Z",
                "endDate": "2026-03-11T01:35:00Z",
                "source": {
                    "device_type": "phone",
                    "device_model": "iPhone15,2",
                },
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Synthetic payload: detailed stages (watchOS 9+ style)
# ---------------------------------------------------------------------------
DETAILED_STAGES_PAYLOAD = {
    "provider": "apple",
    "sdkVersion": "1.0.0",
    "syncTimestamp": "2026-03-11T13:28:04Z",
    "data": {
        "records": [],
        "workouts": [],
        "sleep": [
            {
                "id": "A001",
                "stage": "in_bed",
                "startDate": "2026-03-10T22:00:00Z",
                "endDate": "2026-03-11T06:00:00Z",
                "source": {"device_type": "phone", "device_model": "iPhone15,2"},
            },
            {
                "id": "A002",
                "stage": "light",
                "startDate": "2026-03-10T22:15:00Z",
                "endDate": "2026-03-10T23:00:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
            {
                "id": "A003",
                "stage": "deep",
                "startDate": "2026-03-10T23:00:00Z",
                "endDate": "2026-03-11T00:30:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
            {
                "id": "A004",
                "stage": "rem",
                "startDate": "2026-03-11T00:30:00Z",
                "endDate": "2026-03-11T01:15:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
            {
                "id": "A005",
                "stage": "awake",
                "startDate": "2026-03-11T01:15:00Z",
                "endDate": "2026-03-11T01:25:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
            {
                "id": "A006",
                "stage": "deep",
                "startDate": "2026-03-11T01:25:00Z",
                "endDate": "2026-03-11T02:30:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
            {
                "id": "A007",
                "stage": "light",
                "startDate": "2026-03-11T02:30:00Z",
                "endDate": "2026-03-11T04:00:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
            {
                "id": "A008",
                "stage": "rem",
                "startDate": "2026-03-11T04:00:00Z",
                "endDate": "2026-03-11T05:00:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
            {
                "id": "A009",
                "stage": "light",
                "startDate": "2026-03-11T05:00:00Z",
                "endDate": "2026-03-11T05:45:00Z",
                "source": {"device_type": "watch", "device_model": "Watch7,1"},
            },
        ],
    },
}


class TestCalculateFinalMetrics:
    """Tests for _calculate_final_metrics with different stage combinations."""

    def test_sleeping_stages_only(self) -> None:
        """Older Apple Watch data: only 'sleeping' stages should NOT map to deep."""
        stages = [
            SleepStateStage(
                stage=SleepStageType.SLEEPING,
                start_time=_dt("2026-03-10T23:00:00Z"),
                end_time=_dt("2026-03-10T23:50:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.SLEEPING,
                start_time=_dt("2026-03-10T23:52:00Z"),
                end_time=_dt("2026-03-11T00:45:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.SLEEPING,
                start_time=_dt("2026-03-11T00:47:00Z"),
                end_time=_dt("2026-03-11T01:30:00Z"),
            ),
        ]

        metrics, cleaned = _calculate_final_metrics(stages)

        # "sleeping" should go to sleeping_seconds, NOT deep_seconds
        assert metrics["deep_seconds"] == 0
        assert metrics["light_seconds"] == 0
        assert metrics["rem_seconds"] == 0
        assert metrics["sleeping_seconds"] > 0

        # All cleaned stages should be SLEEPING type
        for s in cleaned:
            assert s.stage == SleepStageType.SLEEPING

        # Total sleeping time: 50min + 53min + 43min = 146min = 8760s
        total_sleeping = metrics["sleeping_seconds"]
        assert total_sleeping == pytest.approx(8760, abs=60)

    def test_sleeping_plus_in_bed(self) -> None:
        """Mixed old-style data: sleeping (watch) + in_bed (phone)."""
        stages = [
            SleepStateStage(
                stage=SleepStageType.SLEEPING,
                start_time=_dt("2026-03-10T23:00:00Z"),
                end_time=_dt("2026-03-11T01:30:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.IN_BED,
                start_time=_dt("2026-03-10T22:55:00Z"),
                end_time=_dt("2026-03-11T01:35:00Z"),
            ),
        ]

        metrics, cleaned = _calculate_final_metrics(stages)

        # deep should be 0 — sleeping is not deep
        assert metrics["deep_seconds"] == 0
        assert metrics["sleeping_seconds"] > 0
        # in_bed calculated from in_bed intervals
        assert metrics["in_bed_seconds"] > 0
        # Cleaned stages should only include sleeping (not in_bed)
        assert all(s.stage == SleepStageType.SLEEPING for s in cleaned)

    def test_detailed_stages(self) -> None:
        """Modern Apple Watch data with deep/light/rem/awake breakdown."""
        stages = [
            SleepStateStage(
                stage=SleepStageType.LIGHT, start_time=_dt("2026-03-10T22:15:00Z"), end_time=_dt("2026-03-10T23:00:00Z")
            ),
            SleepStateStage(
                stage=SleepStageType.DEEP, start_time=_dt("2026-03-10T23:00:00Z"), end_time=_dt("2026-03-11T00:30:00Z")
            ),
            SleepStateStage(
                stage=SleepStageType.REM, start_time=_dt("2026-03-11T00:30:00Z"), end_time=_dt("2026-03-11T01:15:00Z")
            ),
            SleepStateStage(
                stage=SleepStageType.AWAKE, start_time=_dt("2026-03-11T01:15:00Z"), end_time=_dt("2026-03-11T01:25:00Z")
            ),
            SleepStateStage(
                stage=SleepStageType.DEEP, start_time=_dt("2026-03-11T01:25:00Z"), end_time=_dt("2026-03-11T02:30:00Z")
            ),
        ]

        metrics, cleaned = _calculate_final_metrics(stages)

        assert metrics["sleeping_seconds"] == 0
        assert metrics["light_seconds"] == 45 * 60  # 45 min
        assert metrics["deep_seconds"] == (90 + 65) * 60  # 155 min
        assert metrics["rem_seconds"] == 45 * 60  # 45 min
        assert metrics["awake_seconds"] == 10 * 60  # 10 min

    def test_in_bed_fallback_includes_sleeping(self) -> None:
        """When no in_bed stages exist, in_bed_seconds should sum all sleep types."""
        stages = [
            SleepStateStage(
                stage=SleepStageType.SLEEPING,
                start_time=_dt("2026-03-11T00:00:00Z"),
                end_time=_dt("2026-03-11T06:00:00Z"),
            ),
        ]

        metrics, _ = _calculate_final_metrics(stages)

        # No in_bed stages → fallback includes sleeping_seconds
        assert metrics["in_bed_seconds"] == metrics["sleeping_seconds"] + metrics["awake_seconds"]

    def test_empty_stages(self) -> None:
        """Empty stages list should return zero metrics."""
        metrics, cleaned = _calculate_final_metrics([])

        assert metrics["sleeping_seconds"] == 0
        assert metrics["deep_seconds"] == 0
        assert metrics["in_bed_seconds"] == 0
        assert cleaned == []

    def test_only_in_bed_is_not_sleep(self) -> None:
        """When only in_bed stages exist (no sleep phases), nothing counts as asleep.

        An iPhone with a sleep schedule writes in_bed for its bedtime window on its own.
        """
        stages = [
            SleepStateStage(
                stage=SleepStageType.IN_BED,
                start_time=_dt("2026-04-10T22:30:00Z"),
                end_time=_dt("2026-04-11T06:00:00Z"),
            ),
        ]

        metrics, cleaned = _calculate_final_metrics(stages)

        assert metrics["sleeping_seconds"] == 0
        assert metrics["deep_seconds"] == 0
        assert metrics["light_seconds"] == 0
        assert metrics["rem_seconds"] == 0
        # in_bed_seconds still calculated from original in_bed intervals
        assert metrics["in_bed_seconds"] == 7.5 * 3600
        assert cleaned == []

    def test_detailed_plus_sleeping_wrapper_excludes_sleeping(self) -> None:
        """When detailed phases + sleeping wrapper coexist, sleeping is dropped."""
        stages = [
            SleepStateStage(
                stage=SleepStageType.SLEEPING,
                start_time=_dt("2026-04-10T22:00:00Z"),
                end_time=_dt("2026-04-11T06:00:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.LIGHT,
                start_time=_dt("2026-04-10T22:10:00Z"),
                end_time=_dt("2026-04-10T23:00:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.DEEP,
                start_time=_dt("2026-04-10T23:00:00Z"),
                end_time=_dt("2026-04-11T01:00:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.REM,
                start_time=_dt("2026-04-11T01:00:00Z"),
                end_time=_dt("2026-04-11T02:00:00Z"),
            ),
        ]

        metrics, cleaned = _calculate_final_metrics(stages)

        # sleeping wrapper must NOT be counted
        assert metrics["sleeping_seconds"] == 0
        assert metrics["light_seconds"] == 50 * 60
        assert metrics["deep_seconds"] == 2 * 3600
        assert metrics["rem_seconds"] == 1 * 3600
        # Hypnogram should not contain sleeping
        stage_types = {s.stage for s in cleaned}
        assert SleepStageType.SLEEPING not in stage_types
        assert SleepStageType.IN_BED not in stage_types

    def test_detailed_plus_sleeping_plus_in_bed(self) -> None:
        """Full modern scenario: in_bed + sleeping wrapper + detailed phases."""
        stages = [
            SleepStateStage(
                stage=SleepStageType.IN_BED,
                start_time=_dt("2026-04-10T22:00:00Z"),
                end_time=_dt("2026-04-11T06:00:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.SLEEPING,
                start_time=_dt("2026-04-10T22:00:00Z"),
                end_time=_dt("2026-04-11T06:00:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.DEEP,
                start_time=_dt("2026-04-10T22:30:00Z"),
                end_time=_dt("2026-04-11T00:00:00Z"),
            ),
            SleepStateStage(
                stage=SleepStageType.LIGHT,
                start_time=_dt("2026-04-11T00:00:00Z"),
                end_time=_dt("2026-04-11T02:00:00Z"),
            ),
        ]

        metrics, cleaned = _calculate_final_metrics(stages)

        # Only detailed phases should be counted
        assert metrics["sleeping_seconds"] == 0
        assert metrics["deep_seconds"] == 1.5 * 3600
        assert metrics["light_seconds"] == 2 * 3600
        # in_bed still calculated from original intervals
        assert metrics["in_bed_seconds"] == 8 * 3600
        # Hypnogram: only deep + light
        stage_types = {s.stage for s in cleaned}
        assert stage_types == {SleepStageType.DEEP, SleepStageType.LIGHT}


class TestFinishSleep:
    """Tests for finish_sleep with different stage compositions."""

    @patch("app.services.sdk.sleep_service.event_record_service")
    @patch("app.services.sdk.sleep_service.delete_sleep_state")
    def test_finish_sleep_with_sleeping_stages(
        self,
        mock_delete_state: MagicMock,
        mock_event_service: MagicMock,
        db: Session,
    ) -> None:
        """Finish sleep with old-style 'sleeping' data should set correct totals."""
        user_id = str(uuid4())
        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_event_service.create.return_value = mock_record
        mock_event_service.find_adjacent_sleep_record.return_value = None

        state = SleepState(
            uuid=str(uuid4()),
            source_name="Apple Watch",
            device_model="Watch3,3",
            provider="apple",
            start_time=_dt("2026-03-15T23:00:00Z"),
            end_time=_dt("2026-03-16T01:30:00Z"),
            last_start_timestamp=_dt("2026-03-16T00:47:00Z"),
            last_end_timestamp=_dt("2026-03-16T01:30:00Z"),
            sleeping_seconds=8760.0,
            stages=[
                SleepStateStage(
                    stage=SleepStageType.SLEEPING,
                    start_time=_dt("2026-03-15T23:00:00Z"),
                    end_time=_dt("2026-03-15T23:50:00Z"),
                ),
                SleepStateStage(
                    stage=SleepStageType.SLEEPING,
                    start_time=_dt("2026-03-15T23:52:00Z"),
                    end_time=_dt("2026-03-16T00:45:00Z"),
                ),
                SleepStateStage(
                    stage=SleepStageType.SLEEPING,
                    start_time=_dt("2026-03-16T00:47:00Z"),
                    end_time=_dt("2026-03-16T01:30:00Z"),
                ),
            ],
        )

        finish_sleep(db, user_id, state)

        # Verify create was called
        mock_event_service.create.assert_called_once()
        mock_event_service.create_detail.assert_called_once()

        # Check the detail payload
        detail_call = mock_event_service.create_detail.call_args
        detail = detail_call[0][1]  # second positional arg

        # Total duration should include sleeping time
        assert detail.sleep_total_duration_minutes > 0
        # Deep/rem/light should all be 0 (no breakdown available)
        assert detail.sleep_deep_minutes == 0
        assert detail.sleep_rem_minutes == 0
        assert detail.sleep_light_minutes == 0
        # Stages should be present
        assert detail.sleep_stages is not None
        assert len(detail.sleep_stages) == 3
        assert all(s.stage == SleepStageType.SLEEPING for s in detail.sleep_stages)

    @patch("app.services.sdk.sleep_service.event_record_service")
    @patch("app.services.sdk.sleep_service.delete_sleep_state")
    def test_finish_sleep_with_detailed_stages(
        self,
        mock_delete_state: MagicMock,
        mock_event_service: MagicMock,
        db: Session,
    ) -> None:
        """Finish sleep with detailed stages should set deep/rem/light correctly."""
        user_id = str(uuid4())
        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_event_service.create.return_value = mock_record
        mock_event_service.find_adjacent_sleep_record.return_value = None

        state = SleepState(
            uuid=str(uuid4()),
            source_name="Apple Watch",
            device_model="Watch7,1",
            provider="apple",
            start_time=_dt("2026-03-10T22:15:00Z"),
            end_time=_dt("2026-03-11T02:30:00Z"),
            last_start_timestamp=_dt("2026-03-11T01:25:00Z"),
            last_end_timestamp=_dt("2026-03-11T02:30:00Z"),
            light_seconds=2700.0,  # 45 min
            deep_seconds=9300.0,  # 155 min
            rem_seconds=2700.0,  # 45 min
            awake_seconds=600.0,  # 10 min
            stages=[
                SleepStateStage(
                    stage=SleepStageType.LIGHT,
                    start_time=_dt("2026-03-10T22:15:00Z"),
                    end_time=_dt("2026-03-10T23:00:00Z"),
                ),
                SleepStateStage(
                    stage=SleepStageType.DEEP,
                    start_time=_dt("2026-03-10T23:00:00Z"),
                    end_time=_dt("2026-03-11T00:30:00Z"),
                ),
                SleepStateStage(
                    stage=SleepStageType.REM,
                    start_time=_dt("2026-03-11T00:30:00Z"),
                    end_time=_dt("2026-03-11T01:15:00Z"),
                ),
                SleepStateStage(
                    stage=SleepStageType.AWAKE,
                    start_time=_dt("2026-03-11T01:15:00Z"),
                    end_time=_dt("2026-03-11T01:25:00Z"),
                ),
                SleepStateStage(
                    stage=SleepStageType.DEEP,
                    start_time=_dt("2026-03-11T01:25:00Z"),
                    end_time=_dt("2026-03-11T02:30:00Z"),
                ),
            ],
        )

        finish_sleep(db, user_id, state)

        detail = mock_event_service.create_detail.call_args[0][1]

        assert detail.sleep_deep_minutes == 155
        assert detail.sleep_light_minutes == 45
        assert detail.sleep_rem_minutes == 45
        assert detail.sleep_awake_minutes == 10
        assert detail.sleep_total_duration_minutes == 245  # light+deep+rem (no sleeping)


class TestHandleSleepDataIntegration:
    """Integration tests for handle_sleep_data with real payload structures."""

    @patch("app.integrations.celery.tasks.finalize_stale_sleep_task.finalize_stale_sleeps")
    @patch("app.services.sdk.sleep_service.event_record_service")
    @patch("app.services.sdk.sleep_service.get_redis_client")
    def test_handle_real_payload_sleeping_stages(
        self,
        mock_redis_func: MagicMock,
        mock_event_service: MagicMock,
        mock_finalize: MagicMock,
        db: Session,
    ) -> None:
        """Process a synthetic payload with in_bed + sleeping stages.

        Modeled after older Apple Watch pattern:
        - 3 sleeping segments (Watch3,3)
        - 1 in_bed segment (iPhone15,2)
        The watch and the iPhone are two writers → two sessions.
        """
        user_id = str(uuid4())

        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # No existing state
        mock_redis_func.return_value = mock_redis

        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_event_service.create.return_value = mock_record
        mock_event_service.find_adjacent_sleep_record.return_value = None

        request = SyncRequest.model_validate(OLD_WATCH_PAYLOAD)

        handle_sleep_data(db, request, user_id)

        # Sleep state should be saved to Redis (session not yet finalized —
        # that happens via finalize_stale_sleeps.delay())
        assert mock_redis.set.called

        # The finalize task should be dispatched
        mock_finalize.delay.assert_called_once()

        # Verify saved states: one per writer
        states = {
            state.device_model: state
            for state in (SleepState.model_validate_json(call[0][1]) for call in mock_redis.set.call_args_list)
        }
        assert set(states) == {"Watch3,3", "iPhone15,2"}
        state = states["Watch3,3"]

        # sleeping_seconds should be populated, NOT deep_seconds
        assert state.sleeping_seconds > 0
        assert state.deep_seconds == 0
        assert state.light_seconds == 0
        assert state.rem_seconds == 0

        # The watch's entries are within the gap threshold so they merge into one session;
        # the iPhone's in_bed is a session of its own.
        sleeping_stages = [s for s in state.stages if s.stage == SleepStageType.SLEEPING]
        assert len(sleeping_stages) >= 1
        assert [s.stage for s in states["iPhone15,2"].stages] == [SleepStageType.IN_BED]

    @patch("app.integrations.celery.tasks.finalize_stale_sleep_task.finalize_stale_sleeps")
    @patch("app.services.sdk.sleep_service.event_record_service")
    @patch("app.services.sdk.sleep_service.get_redis_client")
    def test_handle_detailed_stages_payload(
        self,
        mock_redis_func: MagicMock,
        mock_event_service: MagicMock,
        mock_finalize: MagicMock,
        db: Session,
    ) -> None:
        """Process a modern payload with detailed sleep stages."""
        user_id = str(uuid4())

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_func.return_value = mock_redis

        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_event_service.create.return_value = mock_record
        mock_event_service.find_adjacent_sleep_record.return_value = None

        request = SyncRequest.model_validate(DETAILED_STAGES_PAYLOAD)

        handle_sleep_data(db, request, user_id)

        # Verify saved state
        last_set_call = mock_redis.set.call_args_list[-1]
        state = SleepState.model_validate_json(last_set_call[0][1])

        assert state.sleeping_seconds == 0
        assert state.deep_seconds > 0
        assert state.light_seconds > 0
        assert state.rem_seconds > 0
        assert state.awake_seconds > 0

        # Verify stage types
        stage_types = {s.stage for s in state.stages}
        assert SleepStageType.DEEP in stage_types
        assert SleepStageType.LIGHT in stage_types
        assert SleepStageType.REM in stage_types
        assert SleepStageType.AWAKE in stage_types


class TestSDKSyncEndpointSleep:
    """Test the /sdk/users/{user_id}/sync endpoint with sleep payloads."""

    def test_sync_endpoint_accepts_sleeping_stage(
        self,
        client: MagicMock,
        db: Session,
    ) -> None:
        """Endpoint should validate payload with 'sleeping' stage (older Apple Watch)."""
        from app.services.sdk_token_service import create_sdk_user_token

        user_id = str(uuid4())
        token = create_sdk_user_token("test_app", user_id)

        with patch("app.api.routes.v1.sdk_sync.process_sdk_upload") as mock_task:
            mock_task.delay.return_value = None

            response = client.post(
                "/api/v1/sdk/users/" + user_id + "/sync/",
                headers={"Authorization": f"Bearer {token}"},
                json=OLD_WATCH_PAYLOAD,
            )

        assert response.status_code == 202
        data = response.json()
        assert data["status_code"] == 202
        mock_task.delay.assert_called_once()

    def test_sync_endpoint_accepts_detailed_stages(
        self,
        client: MagicMock,
        db: Session,
    ) -> None:
        """Endpoint should validate payload with detailed sleep stages."""
        from app.services.sdk_token_service import create_sdk_user_token

        user_id = str(uuid4())
        token = create_sdk_user_token("test_app", user_id)

        with patch("app.api.routes.v1.sdk_sync.process_sdk_upload") as mock_task:
            mock_task.delay.return_value = None

            response = client.post(
                "/api/v1/sdk/users/" + user_id + "/sync/",
                headers={"Authorization": f"Bearer {token}"},
                json=DETAILED_STAGES_PAYLOAD,
            )

        assert response.status_code == 202
        data = response.json()
        assert data["status_code"] == 202


class TestNoIntermediateRedisSaves:
    """Regression test: Redis state must only be saved once per batch, not per stage.

    Previously, save_sleep_state was called inside the per-stage loop, exposing
    partially-accumulated intermediate states to the concurrent finalize_stale_sleeps
    task.  That task could read a partial state, decide it was stale, and finalize it
    — producing a duplicate (subset) sleep record.  Moving the save outside the loop
    prevents this race condition.
    """

    @patch("app.integrations.celery.tasks.finalize_stale_sleep_task.finalize_stale_sleeps")
    @patch("app.services.sdk.sleep_service.event_record_service")
    @patch("app.services.sdk.sleep_service.get_redis_client")
    def test_redis_set_called_once_per_batch(
        self,
        mock_redis_func: MagicMock,
        mock_event_service: MagicMock,
        mock_finalize: MagicMock,
        db: Session,
    ) -> None:
        """Redis .set() should be called once per writer after processing all stages."""
        user_id = str(uuid4())

        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis_func.return_value = mock_redis

        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_event_service.create.return_value = mock_record
        mock_event_service.find_adjacent_sleep_record.return_value = None

        request = SyncRequest.model_validate(DETAILED_STAGES_PAYLOAD)

        handle_sleep_data(db, request, user_id)

        # Count how many times set() was called (each call = one Redis state save).
        # With the fix, this should be once per writer (the iPhone's in_bed and the
        # watch's stages) — after the loop finishes.
        # Before the fix, it was called once per stage (9 times for this payload).
        set_calls = mock_redis.set.call_args_list
        assert len(set_calls) == 2, (
            f"Expected exactly 1 Redis save per writer, got {len(set_calls)}. "
            "Intermediate saves expose partial state to finalize_stale_sleeps."
        )
        assert len({call[0][0] for call in set_calls}) == 2

        # Verify the watch's saved state contains ALL its stages from the payload
        state = SleepState.model_validate_json(set_calls[-1][0][1])
        assert len(state.stages) == 8


class TestHistoricalBulkUploadMerging:
    """Regression tests: consecutive payloads for the same night must produce
    a single merged DB record rather than one record per payload.

    Root cause: Apple sends one night's sleep as many small consecutive payloads
    (each ending where the next begins).  When uploaded hours after recording the
    synchronous stale-check fires on every payload (now - end_time >> 2 h),
    immediately finalizing each payload as its own separate session.

    The fix: finish_sleep() queries for an adjacent existing record and merges
    instead of always inserting.  Combined with the per-user Redis lock that
    serializes concurrent tasks, this guarantees one DB record per night
    regardless of how many payloads Apple sends.
    """

    @patch("app.integrations.celery.tasks.finalize_stale_sleep_task.finalize_stale_sleeps")
    @patch("app.services.sdk.sleep_service.event_record_service")
    @patch("app.services.sdk.sleep_service.get_redis_client")
    def test_second_payload_merges_with_adjacent_db_record(
        self,
        mock_redis_func: MagicMock,
        mock_event_service: MagicMock,
        mock_finalize: MagicMock,
        db: Session,
    ) -> None:
        """When payload B arrives after payload A has already been finalized to the
        DB, finish_sleep should find the adjacent record, delete it and create a
        new merged record covering both sessions.

        Payload A: 23:00–01:00 (light + deep)
        Payload B: 01:00–06:00 (rem + light), chains directly onto A
        """

        user_id = str(uuid4())

        payload_b = {
            "provider": "apple",
            "sdkVersion": "1.0.0",
            "syncTimestamp": "2026-03-23T08:00:01Z",
            "data": {
                "records": [],
                "workouts": [],
                "sleep": [
                    {
                        "id": "B003",
                        "stage": "rem",
                        "startDate": "2026-03-23T01:00:00Z",
                        "endDate": "2026-03-23T02:30:00Z",
                        "source": {"device_type": "watch", "device_model": "Watch7,9"},
                    },
                    {
                        "id": "B004",
                        "stage": "light",
                        "startDate": "2026-03-23T02:30:00Z",
                        "endDate": "2026-03-23T06:00:00Z",
                        "source": {"device_type": "watch", "device_model": "Watch7,9"},
                    },
                ],
            },
        }

        # Simulate the adjacent record that payload A already wrote to the DB
        mock_adjacent = MagicMock()
        mock_adjacent.id = uuid4()
        mock_adjacent.start_datetime = _dt("2026-03-22T23:00:00Z")
        mock_adjacent.end_datetime = _dt("2026-03-23T01:00:00Z")
        mock_adjacent.sleep_detail = MagicMock()
        mock_adjacent.sleep_detail.sleep_stages = [
            {"stage": "light", "start_time": "2026-03-22T23:00:00+00:00", "end_time": "2026-03-22T23:45:00+00:00"},
            {"stage": "deep", "start_time": "2026-03-22T23:45:00+00:00", "end_time": "2026-03-23T01:00:00+00:00"},
        ]

        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # No active Redis state for this user
        mock_redis_func.return_value = mock_redis

        mock_record = MagicMock()
        mock_record.id = uuid4()
        mock_event_service.create.return_value = mock_record
        # First call to find_adjacent returns the payload-A record; second call (if any) returns None
        mock_event_service.find_adjacent_sleep_record.return_value = mock_adjacent

        handle_sleep_data(db, SyncRequest.model_validate(payload_b), user_id)

        # find_adjacent_sleep_record must have been called to look for a matching record
        mock_event_service.find_adjacent_sleep_record.assert_called_once()

        # The old record must be deleted before the merged one is created
        mock_event_service.delete.assert_called_once_with(db, mock_adjacent.id)

        # A new (merged) record must be created
        mock_event_service.create.assert_called_once()
        created_record: EventRecordCreate = mock_event_service.create.call_args[0][1]

        # Merged window covers both A and B
        assert created_record.start_datetime <= _dt("2026-03-22T23:00:00Z")
        assert created_record.end_datetime >= _dt("2026-03-23T06:00:00Z")

        # Detail must contain stages from both payloads
        mock_event_service.create_detail.assert_called_once()
        created_detail: EventRecordDetailCreate = mock_event_service.create_detail.call_args[0][1]
        assert created_detail.sleep_stages is not None
        stage_types = {s.stage for s in created_detail.sleep_stages}
        assert SleepStageType.LIGHT in stage_types
        assert SleepStageType.DEEP in stage_types
        assert SleepStageType.REM in stage_types


WATCH = {"name": "Jan’s Apple Watch", "deviceModel": "Watch7,1", "deviceType": "watch"}
GOOGLE_HEALTH = {"name": "Google Health", "productType": "iPhone16,2", "deviceType": "phone"}
IPHONE = {"name": "JAN", "productType": "iPhone16,2", "deviceType": "phone"}


def _payload(*samples: tuple[str, datetime, datetime, dict]) -> SyncRequest:
    return SyncRequest.model_validate(
        {
            "provider": "apple",
            "sdkVersion": "1.0.0",
            "syncTimestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "records": [],
                "workouts": [],
                "sleep": [
                    {
                        "id": str(uuid4()),
                        "stage": stage,
                        "startDate": start.isoformat(),
                        "endDate": end.isoformat(),
                        "source": source,
                    }
                    for stage, start, end, source in samples
                ],
            },
        }
    )


def _sessions(db: Session, user_id: str) -> dict[str | None, EventRecord]:
    """The user's stored sleep sessions, by the writer they are credited to."""
    records = (
        db.query(EventRecord)
        .join(DataSource, EventRecord.data_source_id == DataSource.id)
        .filter(DataSource.user_id == user_id, EventRecord.category == "sleep")
        .all()
    )
    return {record.source_name: record for record in records}


@patch("app.integrations.celery.tasks.finalize_stale_sleep_task.finalize_stale_sleeps")
class TestSleepPerWriter:
    """Each writer's samples make sessions of their own, tracked in a Redis state per writer.

    Uses the test Redis and database rather than mocks, so states and stored sessions are real.
    """

    def test_overlapping_writers_give_one_session_each(self, mock_finalize: MagicMock, db: Session) -> None:
        """An Apple Watch and the Google Health app recording the same night are two sessions."""
        user_id = str(UserFactory().id)
        night = datetime(2026, 3, 10, 23, 0, tzinfo=timezone.utc)

        handle_sleep_data(
            db,
            _payload(
                ("deep", night, night + timedelta(hours=2), WATCH),
                ("light", night + timedelta(hours=2), night + timedelta(hours=4), WATCH),
                ("light", night - timedelta(minutes=30), night + timedelta(hours=1), GOOGLE_HEALTH),
                ("deep", night + timedelta(hours=1), night + timedelta(hours=3, minutes=30), GOOGLE_HEALTH),
            ),
            user_id,
        )

        sessions = _sessions(db, user_id)
        assert set(sessions) == {"Jan’s Apple Watch", "Google Health"}

        watch = sessions["Jan’s Apple Watch"]
        assert (watch.start_datetime, watch.end_datetime) == (night, night + timedelta(hours=4))
        assert watch.sleep_detail.sleep_deep_minutes == 120
        assert watch.sleep_detail.sleep_light_minutes == 120

        google = sessions["Google Health"]
        assert (google.start_datetime, google.end_datetime) == (
            night - timedelta(minutes=30),
            night + timedelta(hours=3, minutes=30),
        )
        assert google.sleep_detail.sleep_light_minutes == 90
        assert google.sleep_detail.sleep_deep_minutes == 150

    def test_later_payload_extends_only_its_writers_session(self, mock_finalize: MagicMock, db: Session) -> None:
        """A night in progress: each writer's next payload extends that writer's state alone."""
        user_id = str(UserFactory().id)
        # Ends within the 2-hour gap, so no state is finished yet.
        start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=150)
        watch = writer_id("Jan’s Apple Watch", "Watch7,1")
        google = writer_id("Google Health", "iPhone16,2")

        handle_sleep_data(
            db,
            _payload(
                ("light", start, start + timedelta(hours=1), WATCH),
                ("light", start, start + timedelta(hours=1), GOOGLE_HEALTH),
            ),
            user_id,
        )
        handle_sleep_data(
            db, _payload(("deep", start + timedelta(hours=1), start + timedelta(hours=2), WATCH)), user_id
        )

        watch_state = load_sleep_state(user_id, watch)
        google_state = load_sleep_state(user_id, google)
        assert watch_state is not None
        assert google_state is not None
        assert (watch_state.end_time, watch_state.deep_seconds) == (start + timedelta(hours=2), 3600)
        assert (google_state.end_time, google_state.deep_seconds) == (start + timedelta(hours=1), 0)

        handle_sleep_data(
            db, _payload(("rem", start + timedelta(hours=1), start + timedelta(minutes=150), GOOGLE_HEALTH)), user_id
        )

        watch_state = load_sleep_state(user_id, watch)
        google_state = load_sleep_state(user_id, google)
        assert watch_state is not None
        assert google_state is not None
        assert (watch_state.end_time, watch_state.rem_seconds) == (start + timedelta(hours=2), 0)
        assert (google_state.end_time, google_state.rem_seconds) == (start + timedelta(minutes=150), 5400)
        assert _sessions(db, user_id) == {}

    def test_in_bed_only_is_a_session_with_no_sleep(self, mock_finalize: MagicMock, db: Session) -> None:
        """An iPhone's bedtime window on its own is time in bed, not time asleep."""
        user_id = str(UserFactory().id)
        night = datetime(2026, 3, 10, 22, 30, tzinfo=timezone.utc)

        handle_sleep_data(db, _payload(("in_bed", night, night + timedelta(hours=8), IPHONE)), user_id)

        session = _sessions(db, user_id)["JAN"]
        assert session.sleep_detail.sleep_total_duration_minutes == 0
        assert session.sleep_detail.sleep_time_in_bed_minutes == 8 * 60

    @patch("app.services.sdk.sleep_service.event_record_service")
    def test_finish_sleep_deletes_only_its_own_state(
        self, mock_event_service: MagicMock, mock_finalize: MagicMock, db: Session
    ) -> None:
        """Finishing one writer's session, or a state saved before states were kept per writer,
        leaves every other state in place."""
        user_id = str(uuid4())
        mock_event_service.find_adjacent_sleep_record.return_value = None
        at = datetime(2026, 3, 10, 23, 0, tzinfo=timezone.utc)
        redis = get_redis_client()

        def state(source_name: str, device_model: str, writer: str | None) -> SleepState:
            return SleepState(
                uuid=str(uuid4()),
                source_name=source_name,
                device_model=device_model,
                provider="apple",
                writer=writer,
                start_time=at,
                end_time=at + timedelta(hours=1),
                last_start_timestamp=at,
                last_end_timestamp=at + timedelta(hours=1),
                light_seconds=3600,
                stages=[SleepStateStage(stage=SleepStageType.LIGHT, start_time=at, end_time=at + timedelta(hours=1))],
            )

        watch = state("Jan’s Apple Watch", "Watch7,1", writer_id("Jan’s Apple Watch", "Watch7,1"))
        google = state("Google Health", "iPhone16,2", writer_id("Google Health", "iPhone16,2"))
        save_sleep_state(user_id, watch)
        save_sleep_state(user_id, google)
        # A state saved before this change: under the user's own key, with no writer.
        legacy = state("Jan’s Apple Watch", "Watch7,1", None)
        redis.set(key(user_id), legacy.model_dump_json())

        finish_sleep(db, user_id, legacy)
        assert not redis.exists(key(user_id))
        assert redis.smembers(writers_key(user_id)) == {watch.writer, google.writer}

        finish_sleep(db, user_id, watch)
        assert load_sleep_state(user_id, watch.writer) is None
        assert load_sleep_state(user_id, google.writer) == google
        assert redis.sismember("sleep:active_users", user_id)

        finish_sleep(db, user_id, google)
        assert not redis.exists(writers_key(user_id))
        assert not redis.sismember("sleep:active_users", user_id)
