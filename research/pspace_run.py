"""Adaptor to run Permutation Spaces on a cluster."""

from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from os import environ
from os.path import dirname

from clusterun import run_cli


def import_variable(full_name):
    """Import an identifier from another module.

    Arguments:
        full_name (str): The full path to the identifier.

    Returns:
        Any: The value of the identifier in that module
    """
    if '.' in full_name:
        module_name, var_name = full_name.rsplit('.', maxsplit=1)
        module = import_module(module_name)
        return getattr(module, var_name)
    else:
        return import_module(full_name)


def get_parameters(pspace, num_cores, core):
    """Split a parameter space into a size appropriate for one core.

    Arguments:
        pspace (PermutationSpace): The space of parameters.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start.

    Returns:
        Sequence[Namespace]: The relevant section of the parameter space.
    """
    parameters = list(pspace)
    chunk_size = len(parameters) // num_cores
    remainders = len(parameters) % num_cores
    start = chunk_size * core + min(core, remainders)
    end = chunk_size * (core + 1) + min(core + 1, remainders)
    return parameters[start:end]


def run_serial(pspace_name, experiment_fn_name):
    """Run the experiment serially in the current thread

    Arguments:
        pspace_name (str): The space of parameters.
        experiment_fn_name (str): Function that runs the experiment.
    """
    pspace = import_variable(pspace_name)
    experiment_fn = import_variable(experiment_fn_name)
    size = len(pspace)
    for i, params in enumerate(pspace, start=1):
        print(f'{datetime.now().isoformat()} {i}/{size} running {params}')
        experiment_fn(params)


def generate_jobs(filepath, pspace_name, experiment_fn_name, num_cores):
    """Interactively submit jobs.

    Arguments:
        filepath (str): The path to the file to run.
        pspace_name (str): The space of parameters.
        experiment_fn_name (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
    """
    pspace = import_variable(pspace_name)
    job_name = 'pbs_' + filepath
    print(40 * '-')
    print('PARAMETERS')
    print()
    for parameter in pspace.order:
        values = pspace[parameter]
        print(f'{parameter} ({len(values)}):')
        print(f'    {", ".join(repr(param) for param in values)}')
    print()
    print(f'{len(pspace.filters)} filters; total size: {len(pspace)}')
    print()
    print(40 * '-')
    variables = [
        ('core', list(range(min(len(pspace), num_cores)))),
    ]
    commands = [
        f'cd {dirname(filepath)}',
        f'source PYTHONPATH={environ["PYTHONPATH"]}'
        ' '.join([
            f'/home/justinnhli/.venv/research/bin/python3',
            f"'{filepath}'",
            f"'{filepath}'",
            f"'{pspace_name}'",
            f"'{experiment_fn_name}'",
            f"--num-cores'{num_cores}'",
            f'--core "$core"',
        ]),
    ]
    run_cli(job_name, variables, commands, venv='research')


def run_job(pspace_name, experiment_fn_name, num_cores, core):
    """Run a single job.

    Arguments:
        pspace_name (str): The space of parameters.
        experiment_fn_name (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start.
    """
    pspace = import_variable(pspace_name)
    experiment_fn = import_variable(experiment_fn_name)
    parameter_sets = get_parameters(pspace, num_cores, core)
    for i, params in enumerate(parameter_sets):
        print(f'parameters {i} of {len(parameter_sets)}')
        experiment_fn(params)


def cluster_run(filepath, pspace, experiment_fn, num_cores=None, core=None):
    """Entry point to module.

    Arguments:
        filepath (str): The path to the file to run.
        pspace (str): The space of parameters.
        experiment_fn (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start. Defaults to None to generate jobs.
    """
    if num_cores is None or num_cores <= 0:
        run_serial(pspace, experiment_fn)
    elif core is None:
        generate_jobs(filepath, pspace, experiment_fn, num_cores)
    elif not 0 <= core <= num_cores:
        print(f'core must be between 0 and {num_cores - 1} inclusive, but got {core}')
        exit(1)
    else:
        run_job(pspace, experiment_fn, num_cores, core)


def parallel_main(filepath=None, pspace=None, experiment_fn=None, num_cores=None):
    """Command line interface to module.

    Arguments:
        filepath (str): The path to the file to run.
        pspace (str): The space of parameters.
        experiment_fn (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
    """
    arg_parser = ArgumentParser()
    arg_parser.set_defaults(filepath=filepath, pspace=pspace, experiment_fn=experiment_fn)
    if filepath is None:
        arg_parser.add_argument(
            'filepath',
            help='The path to the file to run.',
        )
    if pspace is None:
        arg_parser.add_argument(
            'pspace',
            help='The import path of the parameter space.',
        )
    if experiment_fn is None:
        arg_parser.add_argument(
            'experiment_fn',
            help='The import path of the experiment function.',
        )
    arg_parser.add_argument('--num-cores', type=int, default=num_cores)
    arg_parser.add_argument('--core', type=int)
    args = arg_parser.parse_args()
    for arg in ['filepath', 'pspace', 'experiment_fn']:
        if getattr(args, arg) is None:
            arg_parser.error('A {} must be provided as an call/function argument'.format(arg))
    cluster_run(args.filepath, args.pspace, args.experiment_fn, args.num_cores, args.core)


if __name__ == '__main__':
    parallel_main()
