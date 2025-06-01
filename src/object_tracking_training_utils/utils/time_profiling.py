import time


# TODO have a option that converts time into minutes and hours as needed.
class Timer(object):
    def __init__(self, print_at_exit=False, convert=False):
        """
        Initializes the Timer object.

        Args:
            print_at_exit (bool): If True, prints elapsed time when the context exits.
            convert (bool): If True, converts elapsed seconds into HhMmSs format.
        """
        self.print_at_exit = print_at_exit
        self.convert = convert
        self.exited = False
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """
        Starts the timer when entering the `with` block.

        Returns:
            Timer: self, so elapsed time can be accessed inside the block.
        """
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """
        Stops the timer when exiting the `with` block and optionally prints the result.
        """
        self.end_time = time.time()
        if self.print_at_exit:
            print(self)

    def reset(self):
        """
        Resets the timer start time to now.
        """
        self.start_time = time.time()

    def __repr__(self):
        """
        Returns a string representation of the Timer including elapsed time.

        Returns:
            str: Human-readable representation of the timer.
        """
        return "<{} elapsed={}>".format(self.__class__.__name__,
                                        self.elapsed)

    @property
    def elapsed(self):
        """
        Calculates the elapsed time between start and end.

        Returns:
            float or str: Time elapsed in seconds, or formatted string if convert is True.
        """
        # If called inside the context manager it calculates the partial elapsed
        # time as well.
        if not self.exited:
            self.end_time = time.time()
        elap = self.end_time - self.start_time
        if self.convert:
            sec = elap % 60
            minut = (elap // 60) % 60
            hour = elap // 3600
            return "{:.0f}h{:.0f}m{:.2f}s".format(hour, minut, sec)
        else:
            return elap


def _main():
    """
    Demonstrates the use of the Timer class.

    Runs a loop with time delays and prints the elapsed time at various points.
    """
    with Timer() as tim:
        for i in range(3):
            print(tim.elapsed)
            time.sleep(1)
            for j in range(3):
                print(tim.elapsed)
                time.sleep(0.5)


if __name__ == "__main__":
    _main()
