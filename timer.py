from time import perf_counter
from typing import Self


class Timer:
    """Utility class used to time the duration of processes"""

    def __init__(self, name: str) -> None:
        """Initiliaze Timer"""

        if not isinstance(name, str):
            raise TypeError(f"Name argument must be a string ({name} was given)")

        self.__name = name
        self.__start_time = None
        self.__stop_time = None

    def Start(self) -> Self:
        """Start the Timer"""

        if self.__start_time:
            raise RuntimeError(f"Timer '{self.__name}' already started")

        self.__start_time = perf_counter()

        return self

    def Stop(self) -> None:
        """Stop the Timer"""

        if not self.__start_time:
            raise RuntimeError(f"Timer '{self.__name}' has not started")

        if self.__stop_time:
            raise RuntimeError(f"Timer '{self.__name}' already stopped")

        self.__stop_time = perf_counter()

        elapsed_time = self.__stop_time - self.__start_time

        print(f"Time elapsed during '{self.__name}': {elapsed_time:.2f}")
