from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Callable, Self

from utilities.debug import DEBUG_ENABLED


class Timer:
    """Utility class used to time the duration of processes while debugging"""

    def __init__(self, name: str) -> None:
        """Initiliaze Timer"""

        if not isinstance(name, str):
            raise TypeError(f"Name argument must be a string ({name} was given)")

        self._name = name
        self._start_time = None
        self._stop_time = None

    def start(self) -> Self:
        """Start the Timer"""

        if self._start_time:
            raise RuntimeError(f"Timer '{self._name}' already started")

        self._start_time = perf_counter()

        print(f"Process '{self._name}' started")

        return self

    def stop(self) -> None:
        """Stop the Timer and display the elapsed time"""

        if not self._start_time:
            raise RuntimeError(f"Timer '{self._name}' has not started")

        if self._stop_time:
            raise RuntimeError(f"Timer '{self._name}' already stopped")

        self._stop_time = perf_counter()

        elapsed_time = self._stop_time - self._start_time

        if DEBUG_ENABLED:
            print(f"Time elapsed during '{self._name}': {elapsed_time:.2f}")

    @staticmethod
    def time(name: str) -> Callable:
        """Decorator factory used for timing the duration of a function"""

        if not isinstance(name, str):
            raise TypeError(f"Name argument must be a string ({name} was given)")

        def decorator(func: Callable):
            if not isinstance(func, Callable):
                raise TypeError("parameter func must be of type Callable")

            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = perf_counter()
                results = func(*args, **kwargs)
                end_time = perf_counter()

                elapsed_time = end_time - start_time

                if DEBUG_ENABLED:
                    print(f"Time elapsed during '{name}': {elapsed_time:.2f}")

                return results

            return wrapper

        return decorator


class TimeManager:
    """Context Manager used for timing the duration of a procedure"""

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Name argument must be a string ({name} was given)")

        self.name = name

    def __enter__(self):
        self._start_time = perf_counter()

        print(f"Process '{self.name}' started")

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        end_time = perf_counter()
        elapsed_time = end_time - self._start_time

        if DEBUG_ENABLED:
            if exc_type:
                exc_message = f"with exception '{exc_type.__name__}: {exc_value}'"
                print(
                    f"Time elapsed during '{self.name}' {exc_message}: {elapsed_time:.2f}"
                )
            else:
                print(f"Time elapsed during '{self.name}': {elapsed_time:.2f}")
