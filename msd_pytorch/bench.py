import timeit
import sys
import math


def _format_time(timespan, precision=3):
    """Formats the timespan in a human readable form"""

    if timespan >= 60.0:
        # we have more than a minute, format that in a human readable form
        # Idea from http://snipplr.com/view/5713/
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            value = int(leftover / length)
            if value > 0:
                leftover = leftover % length
                time.append(u"%s%s" % (str(value), suffix))
            if leftover < 1:
                break
        return " ".join(time)

    # Unfortunately the unicode 'micro' symbol can cause problems in
    # certain terminals.
    # See bug: https://bugs.launchpad.net/ipython/+bug/348466
    # Try to prevent crashes by being more secure than it needs to
    # E.g. eclipse is able to print a µ, but has no sys.stdout.encoding set.
    units = [u"s", u"ms", u"us", "ns"]  # the save value
    if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
        try:
            u"\xb5".encode(sys.stdout.encoding)
            units = [u"s", u"ms", u"\xb5s", "ns"]
        except Exception:
            pass
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    return u"%.*g %s" % (precision, timespan * scaling[order], units[order])


class TimeitResult(object):
    """
    Object returned by the timeit magic with info about the run.

    Contains the following attributes :

    loops: (int) number of loops done per measurement
    repeat: (int) number of times the measurement has been repeated
    best: (float) best execution time / number
    all_runs: (list of float) execution time of each run (in s)

    """

    def __init__(self, name, loops, repeat, best, worst, all_runs, precision=3):
        self.name = name
        self.loops = loops
        self.repeat = repeat
        self.best = best
        self.worst = worst
        self.all_runs = all_runs
        self._precision = precision
        self.timings = [dt / self.loops for dt in all_runs]

    @property
    def average(self):
        return math.fsum(self.timings) / len(self.timings)

    @property
    def stdev(self):
        mean = self.average
        return (
            math.fsum([(x - mean) ** 2 for x in self.timings]) / len(self.timings)
        ) ** 0.5

    def __str__(self):
        pm = "+-"
        if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
            try:
                u"\xb1".encode(sys.stdout.encoding)
                pm = u"\xb1"
            except Exception:
                pass
        return u"{name:<30}{mean:<7} {pm} {std:<7} per loop (mean {pm} std. dev. of {runs} run{run_plural}, {loops} loop{loop_plural} each)".format(
            name=self.name + ":",
            pm=pm,
            runs=self.repeat,
            loops=self.loops,
            loop_plural="" if self.loops == 1 else "s",
            run_plural="" if self.repeat == 1 else "s",
            mean=_format_time(self.average, self._precision),
            std=_format_time(self.stdev, self._precision),
        )

    def _repr_pretty_(self, p, cycle):
        unic = self.__str__()
        p.text(u"<TimeitResult : " + unic + u">")


def bench(name, timer, repeat=timeit.default_repeat):
    number = 0
    # determine number so that 0.2 <= total time < 2.0
    for index in range(0, 10):
        number = 10 ** index
        time_number = timer.timeit(number)
        if time_number >= 0.2:
            break

    all_runs = timer.repeat(repeat, number)
    best = min(all_runs) / number
    worst = max(all_runs) / number
    timeit_result = TimeitResult(name, number, repeat, best, worst, all_runs)
    return timeit_result
