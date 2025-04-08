from datetime import datetime


def get_timestamp():
    """
    Generate a timestamp string in the format 'YYYYMMDD_HHMMSS'.

    Returns:
        str: The current timestamp as a string in the format 'YYYYMMDD_HHMMSS'.

    Example:
        get_timestamp()
        '20240814_103045'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')