import time


def format_time_elapsed(start_time: float) -> str:
    """
    Format elapsed time into 'HH:MM:SS.mmm' format.

    Args:
        start_time (float): The start time as returned by `time.time()`.

    Returns:
        str: The formatted time elapsed.
    """
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"
