"""Infrastructure for building data processing pipelines."""

from os.path import exists, getmtime

class PipelineError(Exception):
    """Custom error for pipelines."""

    pass

class PipelineStep:
    """A step in a pipeline."""

    def __init__(self, infile, outfile=None):
        """Construct a pipeline.

        Arguments:
            infile (str): The input data file.
            outfile (str): The output data file. If None (default), child
                classes must override get_outfile method to dynamically
                determine outfile.
        """
        self.infile = infile
        self._outfile = outfile

    @property
    def outfile(self):
        """Get the outfile name.

        Returns:
            str: The name of the outfile.

        Raises:
            PipelineError: If neither outfile nor get_outfile are defined.
        """
        if self._outfile is not None:
            return self._outfile
        if self.get_outfile():
            return self.get_outfile()
        raise PipelineError('Outfile not specified implicitly or explicitly')

    def get_outfile(self): # pylint: disable=no-self-use
        """Get the outfile name, if determined dynamically.

        Returns:
            str: The name of the outfile.
        """
        return ""

    def deserialize(self, fd):
        """Read the infile.

        Arguments:
            fd (File): The read-only object of the infile.

        Returns:
            obj: The deserialized data.
        """
        raise NotImplementedError()

    def process(self, data):
        """Process the data.

        Arguments:
            data (obj): The deserialized data from the infile.

        Returns:
            obj: The processed data.
        """
        raise NotImplementedError()

    def serialize(self, data):
        """Serialize the processed data.

        Arguments:
            data (obj): The processed data.

        Returns:
            List[str]: The lines of strings to be saved to outfile.
        """
        raise NotImplementedError()

    def run(self):
        """Run this step in the pipeline."""
        assert self.infile is None or exists(self.infile)
        if self.outfile is not None and exists(self.outfile):
            if self.infile is not None and getmtime(self.infile) > getmtime(self.outfile):
                print(f'Warning: infile {self.infile} has a later modification time than outfile {self.outfile}')
        else:
            if self.infile is None:
                data = self.deserialize(None)
            else:
                with open(self.infile) as fd:
                    data = self.deserialize(fd)
            with open(self.outfile, 'w') as fd:
                for line in self.serialize(self.process(data)):
                    fd.write(line)
                    fd.write('\n')
