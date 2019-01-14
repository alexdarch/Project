import sys


class Utils(dict):
    def __getattr__(self, name):
        return self[name]

    @staticmethod
    def update_progress(tag, progress, elapsed_time):
        ''' update_progress() : Displays or updates a console progress bar. Accepts a float between 0 and 1.
            Any int will be converted to a float. A value under 0 represents a 'halt'.
            A value at 1 or bigger represents 100% '''
        barLength = 25  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "\r\n"
        block = int(round(barLength * progress))
        text = "\r{0}: [{1}] {2:.2f}%.  Iteration Time: {3:.3f}secs. {4}".format(tag, "#" * block + "-" * (barLength - block), progress * 100, elapsed_time, status)
        sys.stdout.write(text)
        sys.stdout.flush()

