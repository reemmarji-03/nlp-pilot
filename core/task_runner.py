import inspect
import threading
from typing import Any, Callable, Dict, Optional, Tuple


class CancelledError(Exception):
    """Raised by worker functions to signal voluntary cancellation."""


class TaskRunner:
    def __init__(self, root):
        self._root = root
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

    def run(
        self,
        fn: Callable,
        args: Tuple = (),
        kwargs: Dict = {},
        on_progress: Optional[Callable] = None,
        on_done: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        on_cancel: Optional[Callable] = None,
    ) -> None:
        if self.is_running:
            raise RuntimeError("A task is already running.")

        self._cancel_event = threading.Event()
        cancel_event = self._cancel_event

        def _progress_cb(percent: float, message: str) -> None:
            if on_progress:
                self._root.after(0, lambda p=percent, m=message: on_progress(p, m))

        def _worker():
            try:
                sig = inspect.signature(fn)
                extra: Dict[str, Any] = {}
                if "cancel_event" in sig.parameters:
                    extra["cancel_event"] = cancel_event
                if "progress_callback" in sig.parameters:
                    extra["progress_callback"] = _progress_cb

                result = fn(*args, **{**kwargs, **extra})

                if cancel_event.is_set():
                    if on_cancel:
                        self._root.after(0, on_cancel)
                else:
                    if on_done:
                        self._root.after(0, lambda r=result: on_done(r))

            except CancelledError:
                if on_cancel:
                    self._root.after(0, on_cancel)
            except Exception as exc:
                if on_error:
                    self._root.after(0, lambda e=exc: on_error(e))

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
