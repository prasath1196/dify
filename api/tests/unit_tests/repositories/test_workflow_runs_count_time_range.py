from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from sqlalchemy.orm import sessionmaker

from repositories.sqlalchemy_api_workflow_run_repository import DifyAPISQLAlchemyWorkflowRunRepository


def test_time_range_threshold_is_naive_utc_like_created_at():
    session = MagicMock()
    session.scalar.return_value = 0
    session_maker = MagicMock(spec=sessionmaker)
    session_maker.return_value.__enter__.return_value = session
    repository = DifyAPISQLAlchemyWorkflowRunRepository(session_maker)

    repository.get_workflow_runs_count("tenant-1", "app-1", "debugging", status="succeeded", time_range="1d")

    statement = session.scalar.call_args.args[0]
    thresholds = [value for value in statement.compile().params.values() if isinstance(value, datetime)]
    assert len(thresholds) == 1
    threshold = thresholds[0]
    assert threshold.tzinfo is None
    expected = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=1)
    assert abs((threshold - expected).total_seconds()) < 60
