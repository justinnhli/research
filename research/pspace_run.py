"""Adaptor to run Permutation Spaces on a cluster."""

import sys
from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from itertools import islice
from os import environ
from pathlib import Path

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
    return list(islice(pspace, core, None, num_cores))


def dry_run(pspace_name, experiment_fn_name, num_cores, core):
    """Print the parameter space.

    Arguments:
        pspace_name (str): The space of parameters.
        experiment_fn_name (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start.
    """
    import_variable(experiment_fn_name)
    pspace = import_variable(pspace_name)
    psubspace = get_parameters(pspace, num_cores, core)
    size = len(psubspace)
    for params in psubspace:
        print(params)
    print(f'total: {size}')


def run_serial(pspace_name, experiment_fn_name, num_cores, core):
    """Run an experiment serially in the current thread.

    Arguments:
        pspace_name (str): The space of parameters.
        experiment_fn_name (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start.
    """
    pspace = import_variable(pspace_name)
    experiment_fn = import_variable(experiment_fn_name)
    psubspace = get_parameters(pspace, num_cores, core)
    size = len(psubspace)
    for i, params in enumerate(psubspace):
        print(f'{datetime.now().isoformat()} {i}/{size} running: {params}')
        experiment_fn(params)


def generate_jobs(filepath, pspace_name, experiment_fn_name, num_cores):
    """Interactively submit jobs.

    Arguments:
        filepath (Path): The path to the file to run.
        pspace_name (str): The space of parameters.
        experiment_fn_name (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
    """
    pspace = import_variable(pspace_name)
    job_name = 'pbs_' + filepath.stem
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
        f'cd {filepath.parent}',
        f'export PYTHONPATH={environ["PYTHONPATH"]}',
        ' '.join([
            f'/home/justinnhli/.venv/research/bin/python3',
            f"'{filepath}'",
            f"'{filepath}'",
            f"'{pspace_name}'",
            f"'{experiment_fn_name}'",
            f"--num-cores '{num_cores}'",
            f'--core "$core"',
        ]),
    ]
    run_cli(job_name, variables, commands, venv='research')


def create_arg_parser(filepath=None, pspace=None, experiment_fn=None, num_cores=None):
    """Create the ArgumentParser.

    Arguments:
        filepath (Path): The path to the file to run.
        pspace (str): The space of parameters.
        experiment_fn (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.

    Returns:
        ArgumentParser: The argument parser.
    """
    arg_parser = ArgumentParser()
    arg_parser.set_defaults(filepath=filepath, pspace=pspace, experiment_fn=experiment_fn)
    arg_parser.add_argument(
        'filepath',
        type=Path,
        default=filepath,
        nargs='?',
        help='The path to the file to run.',
    )
    arg_parser.add_argument(
        'pspace',
        default=pspace,
        nargs='?',
        help='The import path of the parameter space.',
    )
    arg_parser.add_argument(
        'experiment_fn',
        default=experiment_fn,
        nargs='?',
        help='The import path of the experiment function.',
    )
    arg_parser.add_argument(
        '--num-cores', type=int, default=num_cores,
        help='Number of cores to run the job on.',
    )
    arg_parser.add_argument(
        '--core', type=int,
        help='The core to run the current job. Must be used with --num-cores.',
    )
    arg_parser.add_argument(
        '--dry-run', action='store_true',
        help='If set, print out the parameter space and exit.',
    )
    return arg_parser


def check_arguments(arg_parser, args):
    """Check arguments for errors.

    Arguments:
        arg_parser (ArgumentParser): The ArgumentParser.
        args (argparse.Namespace): The parsed arguments
    """
    for arg in ['filepath', 'pspace', 'experiment_fn']:
        if getattr(args, arg) is None:
            arg_parser.error(f'Argument --{arg} must be provided either by call or by command line')
    if args.num_cores is None:
        if args.core is not None:
            arg_parser.error('Argument --core must be used with --num-cores')
    else:
        if args.num_cores < 1:
            arg_parser.error('Argument --num-cores must be a positive integer')
        if args.core is not None and not 0 <= args.core <= args.num_cores:
            arg_parser.error(
                f'Argument --core must be between 0 and {args.num_cores - 1} inclusive, but got {args.core}'
            )


def set_arguments(args):
    """Set unset arguments intelligently.

    Arguments:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        argparse.Namespace: The parsed arguments with new values.
    """
    args.filepath = args.filepath.expanduser().resolve()
    if args.num_cores is None:
        args.num_cores = 1
        args.core = 0
    return args


def parse_arguments(cli_args, filepath=None, pspace=None, experiment_fn=None, num_cores=None):
    """Parse arguments and fill in defaults.

    Arguments:
        cli_args (Sequence[str]): The CLI arguments.
        filepath (Path): The path to the file to run.
        pspace (str): The space of parameters.
        experiment_fn (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    arg_parser = create_arg_parser(filepath, pspace, experiment_fn, num_cores)
    args = arg_parser.parse_args(cli_args)
    check_arguments(arg_parser, args)
    return set_arguments(args)


def parallel_main(filepath=None, pspace=None, experiment_fn=None, num_cores=None):
    """Command line interface to module.

    Arguments:
        filepath (Path): The path to the file to run.
        pspace (str): The space of parameters.
        experiment_fn (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
    """
    filepath = filepath.expanduser().resolve()
    args = parse_arguments(sys.argv[1:], filepath, pspace, experiment_fn, num_cores)
    if args.dry_run:
        dry_run(args.pspace, args.experiment_fn, args.num_cores, args.dry_run)
    elif args.num_cores is None or args.num_cores <= 0:
        run_serial(args.pspace, args.experiment_fn, args.num_cores, args.core)
    elif args.core is None:
        generate_jobs(args.filepath, args.pspace, args.experiment_fn, args.num_cores)
    else:
        run_serial(args.pspace, args.experiment_fn, args.num_cores, args.core)
