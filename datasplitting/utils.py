from datetime import datetime


def printElapsedTime(message, first_time):
    later_time = datetime.now()
    duration = later_time - first_time
    duration_in_s = duration.total_seconds()
    print(message, duration_in_s, ' seconds')
