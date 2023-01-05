"""Tests for pipeline.py."""

from os.path import join as join_path
from datetime import datetime
from tempfile import TemporaryDirectory

from research import PipelineStep


class RangeStep(PipelineStep):
    """Pipeline step to count to a number."""

    def __init__(self, count, *args, **kwargs):
        """Construct a RangeStep.

        Arguments:
            count (int): The number to count to.
            args (obj): Positional arguments to the superclass.
            kwargs (obj): Keyword arguments to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.count = count

    def deserialize(self, fd): # noqa: D102
        return range(self.count)

    def process(self, data): # noqa: D102
        return data

    def serialize(self, data): # noqa: D102
        return (str(datum) for datum in data)


class SquareStep(PipelineStep):
    """Pipeline step to square some numbers."""

    def deserialize(self, fd): # noqa: D102
        return (int(line) for line in fd.readlines())

    def process(self, data): # noqa: D102
        return (datum**2 for datum in data)

    def serialize(self, data): # noqa: D102
        return (str(datum) for datum in data)


def test_pipeline_correctness():
    """Test pipeline caching correctness."""
    with TemporaryDirectory() as temp_dir:
        count_file = join_path(temp_dir, 'count.txt')
        square_file = join_path(temp_dir, 'square.txt')
        RangeStep(count=1000, infile=None, outfile=count_file).run()
        SquareStep(infile=count_file, outfile=square_file).run()
        with open(count_file, encoding='utf-8') as fd:
            test = all(int(line.strip()) == i for i, line in enumerate(fd.readlines()))
            message = 'Cached data is different from raw data for Count'
            assert test, message
        with open(square_file, encoding='utf-8') as fd:
            test = all(int(line.strip()) == i**2 for i, line in enumerate(fd.readlines()))
            message = 'Cached data is different from raw data for Square'
            assert test, message


def test_pipeline_speed():
    """Test pipeline caching speed."""
    with TemporaryDirectory() as temp_dir:
        count_file = join_path(temp_dir, 'count.txt')
        square_file = join_path(temp_dir, 'square.txt')
        range_step = RangeStep(count=1000000, infile=None, outfile=count_file)
        square_step = SquareStep(infile=count_file, outfile=square_file)
        before = datetime.now()
        range_step.run()
        square_step.run()
        between = datetime.now()
        range_step.run()
        square_step.run()
        after = datetime.now()
        raw_time = (between - before).total_seconds()
        cached_time = (after - between).total_seconds()
        assert cached_time < raw_time / 4, 'Cached pipeline is not four times as fast as raw pipeline'
