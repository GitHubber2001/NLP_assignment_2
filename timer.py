from time import perf_counter
from typing import Callable, Self

from debug import DEBUG_ENABLED


class Timer:
    """Utility class used to time the duration of processes while debugging"""

    def __init__(self, name: str) -> None:
        """Initiliaze Timer"""

        if not isinstance(name, str):
            raise TypeError(f"Name argument must be a string ({name} was given)")

        self._name = name
        self._start_time = None
        self._stop_time = None

    def Start(self) -> Self:
        """Start the Timer"""

        if self._start_time:
            raise RuntimeError(f"Timer '{self._name}' already started")

        self._start_time = perf_counter()

        print(f"Process '{self._name}' started")

        return self

    def Stop(self) -> None:
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
    def Time(name: str) -> Callable:
        """Decorator that times the duration of a function"""

        if not isinstance(name, str):
            raise TypeError(f"Name argument must be a string ({name} was given)")

        def TimeFunction(func: Callable):
            if not isinstance(func, Callable):
                raise TypeError("parameter func must be of type Callable")
    
            start_time = perf_counter()
            results = func()
            end_time = perf_counter()

            elapsed_time = end_time - start_time

            if DEBUG_ENABLED:
                print(f"Time elapsed during '{name}': {elapsed_time:.2f}")

            return results

        return TimeFunction
