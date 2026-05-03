import threading
import time
import pytest
from core.task_runner import TaskRunner, CancelledError


class MockRoot:
    """Dispatches after() callbacks immediately on the calling thread."""
    def after(self, delay, fn):
        fn()


def test_on_done_called_with_result():
    runner = TaskRunner(root=MockRoot())
    results = []

    def fn():
        return 42

    runner.run(fn, on_done=lambda r: results.append(r))
    runner._thread.join(timeout=2)

    assert results == [42]


def test_on_error_called_on_exception():
    runner = TaskRunner(root=MockRoot())
    errors = []

    def fn():
        raise ValueError("boom")

    runner.run(fn, on_error=lambda e: errors.append(str(e)))
    runner._thread.join(timeout=2)

    assert errors == ["boom"]


def test_cancel_via_event_calls_on_cancel():
    runner = TaskRunner(root=MockRoot())
    cancel_called = threading.Event()
    done_called = threading.Event()

    def fn(cancel_event=None):
        cancel_event.wait(timeout=2)

    runner.run(
        fn,
        on_done=lambda r: done_called.set(),
        on_cancel=cancel_called.set,
    )
    time.sleep(0.05)
    runner.cancel()
    runner._thread.join(timeout=2)

    assert cancel_called.is_set()
    assert not done_called.is_set()


def test_cancel_via_cancelled_error_calls_on_cancel():
    runner = TaskRunner(root=MockRoot())
    cancel_called = threading.Event()

    def fn():
        raise CancelledError()

    runner.run(fn, on_cancel=cancel_called.set)
    runner._thread.join(timeout=2)

    assert cancel_called.is_set()


def test_progress_callback_forwarded():
    runner = TaskRunner(root=MockRoot())
    progress_calls = []
    done = threading.Event()

    def fn(progress_callback=None):
        progress_callback(0.5, "halfway")
        return "done"

    runner.run(
        fn,
        on_progress=lambda p, m: progress_calls.append((p, m)),
        on_done=lambda r: done.set(),
    )
    runner._thread.join(timeout=2)

    assert (0.5, "halfway") in progress_calls


def test_is_running_true_while_alive():
    runner = TaskRunner(root=MockRoot())
    started = threading.Event()

    def fn():
        started.set()
        time.sleep(0.3)

    runner.run(fn)
    started.wait(timeout=1)
    assert runner.is_running is True
    runner._thread.join(timeout=2)
    assert runner.is_running is False


def test_run_raises_if_already_running():
    runner = TaskRunner(root=MockRoot())
    started = threading.Event()

    def fn():
        started.set()
        time.sleep(0.5)

    runner.run(fn)
    started.wait(timeout=1)

    with pytest.raises(RuntimeError, match="already running"):
        runner.run(fn)

    runner.cancel()
    runner._thread.join(timeout=2)


def test_functions_without_extra_params_run_unchanged():
    runner = TaskRunner(root=MockRoot())
    results = []

    def fn():
        return "ok"

    runner.run(fn, on_done=lambda r: results.append(r))
    runner._thread.join(timeout=2)
    assert results == ["ok"]
