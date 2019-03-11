"""Adaptor to run Permutation Spaces on a cluster."""

import sys
from argparse import ArgumentParser
from datetime import datetime
from importlib import import_module
from itertools import islice
from os import environ
from pathlib import Path

from clusterun import run_cli as cluster_run_cli


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


def get_parameters(pspace, num_cores=1, core=0, skip=0):
    """Split a parameter space into a size appropriate for one core.

    Arguments:
        pspace (PermutationSpace): The space of parameters.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start.
        skip (int): The number of initial parameters to skip.

    Returns:
        Sequence[Namespace]: The relevant section of the parameter space.
    """
    return list(islice(pspace, core, None, num_cores))[skip:]


def dry_run(pspace_name, num_cores, core, skip=0):
    """Print the parameter space.

    Arguments:
        pspace_name (str): The space of parameters.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start.
        skip (int): The number of initial parameters to skip.
    """
    pspace = import_variable(pspace_name)
    psubspace = get_parameters(pspace, num_cores, core, skip)
    size = len(psubspace)
    for params in psubspace:
        print(params)
    print(f'total: {size}')


def run_serial(pspace_name, experiment_fn_name, num_cores, core, skip=0):
    """Run an experiment serially in the current thread.

    Arguments:
        pspace_name (str): The space of parameters.
        experiment_fn_name (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
        core (int): The core whose job to start.
        skip (int): The number of initial parameters to skip.
    """
    pspace = import_variable(pspace_name)
    experiment_fn = import_variable(experiment_fn_name)
    psubspace = get_parameters(pspace, num_cores, core, skip)
    size = len(psubspace)
    for i, params in enumerate(psubspace, start=1):
        print(f'{datetime.now().isoformat()} running {i}/{size}: {params}')
        experiment_fn(params)


def dispatch(filepath, pspace_name, experiment_fn_name, num_cores):
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
        f'export PYTHONPATH={environ.get("PYTHONPATH", "")}',
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
    cluster_run_cli(job_name, variables, commands, venv='research')


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
        '--skip', type=int, default=0,
        help='Skip some initial parameters. Ignored if not running serially.',
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
    arg_parser.add_argument(
        '--dispatch', type=bool, default=None,
        help=' '.join([
            'Force job to be dispatched if true, or to run serially if not.',
            'By default, will dispatch if --num-cores is set but --core is not set.',
        ]),
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
    if args.skip < 0:
        arg_parser.error('Argument --skip must be a non-negative integer')


def set_arguments(args):
    """Set unset arguments intelligently.

    Arguments:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        argparse.Namespace: The parsed arguments with new values.
    """
    args.filepath = args.filepath.expanduser().resolve()
    if args.dispatch is None:
        args.dispatch = args.num_cores is not None and args.core is None
    if args.num_cores is None:
        args.num_cores = 1
        args.core = 0
    if args.dispatch:
        args.skip = 0
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


def main(cli_args, filepath=None, pspace=None, experiment_fn=None, num_cores=None):
    """Entry point to the module.

    Arguments:
        cli_args (Sequence[str]): The CLI arguments.
        filepath (Path): The path to the file to run.
        pspace (str): The space of parameters.
        experiment_fn (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
    """
    if filepath is not None:
        filepath = filepath.expanduser().resolve()
    args = parse_arguments(cli_args, filepath, pspace, experiment_fn, num_cores)
    if args.dry_run:
        dry_run(args.pspace, args.num_cores, args.core, args.skip)
    elif args.dispatch:
        dispatch(args.filepath, args.pspace, args.experiment_fn, args.num_cores)
    else:
        run_serial(args.pspace, args.experiment_fn, args.num_cores, args.core, args.skip)


def pspace_run_cli(filepath=None, pspace=None, experiment_fn=None, num_cores=None):
    """Command line interface to the module.

    Arguments:
        filepath (Path): The path to the file to run.
        pspace (str): The space of parameters.
        experiment_fn (str): Function that runs the experiment.
        num_cores (int): The number of cores to split jobs for.
    """
    main(sys.argv[1:], filepath, pspace, experiment_fn, num_cores)
