class TimeTracker:
    """
    A class to manage time tracking across different segments of a route.
    """

    def __init__(self, initial_hour):
        """
        Initialize the time tracker.

        :param initial_hour: Starting hour of the journey
        """
        self.current_hour = initial_hour
        self.total_elapsed_time = 0  # Total time in minutes

    def update_time(self, segment_time):
        """
        Update the total elapsed time and current hour.

        :param segment_time: Time taken for the current segment in minutes
        :return: Updated current hour
        """
        # Update total elapsed time
        self.total_elapsed_time += segment_time

        # Calculate new current hour
        self.current_hour = (self.current_hour + int(self.total_elapsed_time // 60)) % 24

        # Adjust total elapsed time to remaining minutes
        self.total_elapsed_time %= 60

        return self.current_hour

    @property
    def total_hours(self):
        """
        Calculate total hours traveled.

        :return: Total hours as a float
        """
        return self.total_elapsed_time / 60