class RunningAverage():
    """A simple class that maintains the running average of a quantity

    This class is useful during model training to track the average of metrics
    like loss or accuracy without storing all past values in memory.

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        """
        Adds a new value to the total and increments the step count.

        Args:
            val (float): The new value to include in the running average.
        """
        self.total += val
        self.steps += 1

    def __call__(self):
        """
        Returns the current running average.

        Returns:
            float: The average of all added values.
        """
        return self.total/float(self.steps)
