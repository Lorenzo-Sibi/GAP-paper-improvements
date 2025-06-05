from core import console
with console.status('importing modules'):
    import torch
    import numpy as np
    from rich import box
    from rich.table import Table
    from time import time
    from typing import Annotated
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from core import globals
    from core.datasets import DatasetLoader
    from core.args.utils import print_args, create_arguments, strip_kwargs, ArgInfo
    from core.loggers import Logger
    from core.methods.node import supported_methods, NodeClassification
    from core.utils import seed_everything, confidence_interval
    from torch_geometric.data import Data


def run(seed:    Annotated[int,   ArgInfo(help='initial random seed')] = 12345,
        repeats: Annotated[int,   ArgInfo(help='number of times the experiment is repeated')] = 1,
        debug:   Annotated[bool,  ArgInfo(help='enable global debug mode')] = False,
        **kwargs
    ):

    seed_everything(seed)

    if debug:
        console.info('debug mode enabled')
        globals['debug'] = True
        console.log_level = console.DEBUG

    with console.status('loading dataset'):
        loader_args = strip_kwargs(DatasetLoader, kwargs)
        data_initial = DatasetLoader(**loader_args).load(verbose=True)

    num_classes = data_initial.y.max().item() + 1
    config = dict(**kwargs, seed=seed, repeats=repeats)
    logger_args = strip_kwargs(Logger.setup, kwargs)
    logger = Logger.setup(enabled=True, config=config, **logger_args)

    ### initiallize method ###
    Method = supported_methods[kwargs['method']]
    method_args = strip_kwargs(Method, kwargs)
    method: NodeClassification = Method(num_classes=num_classes, **method_args)

    run_metrics = {}

    ### run experiment ###
    for iteration in range(repeats):
        start_time = time()
        data = Data(**data_initial.to_dict())
        metrics = method.fit(data)
        end_time = time()
        metrics['duration'] = end_time - start_time

        ### process results ###
        for metric, value in metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

         ### print results ###
        table = Table(title=f'run {iteration + 1}', box=box.HORIZONTALS)
        table.add_column('metric')
        table.add_column('last', style="cyan")
        table.add_column('mean', style="cyan")
        table.add_row('test/acc', f'{run_metrics["test/acc"][-1]:.2f}', f'{np.mean(run_metrics["test/acc"]):.2f}')
        console.info(table)
        console.print()

        ### reset method's parameters for the next run ###
        method.reset_parameters()

    logger.enable()
    summary = {}
    
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=seed)

    table_summary = Table(title="Training Results", show_header=True)
    table_summary.add_column("Metric", style="cyan")
    table_summary.add_column("Mean", style="green")
    table_summary.add_column("Std", style="yellow")
    table_summary.add_column("CI", style="magenta")

    groups = [
        ("Epoch", "epoch"),
        ("Training", "train/acc", "train/loss"),
        ("Validation", "val/acc", "val/loss"),
        ("Testing", "test/acc"),
        ("Duration", "duration")
    ]

    for group_name, *metrics in groups:
        for metric in metrics:
            display_name = f"{group_name} {metric.split('/')[-1].capitalize()}" if '/' in metric else group_name
            mean = f"{summary[f'{metric}_mean']:.2f}"
            std = f"{summary[f'{metric}_std']:.2f}"
            ci = f"{summary[f'{metric}_ci']:.2f}"
            table_summary.add_row(display_name, mean, std, ci)

    console.info(table_summary)


def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')
    method_subparser = init_parser.add_subparsers(dest='method', required=True, title='algorithm to use')

    for method_name, method_class in supported_methods.items():
        method_parser = method_subparser.add_parser(
            name=method_name, 
            help=method_class.__doc__, 
            formatter_class=ArgumentDefaultsHelpFormatter
        )

        # dataset args
        group_dataset = method_parser.add_argument_group('dataset arguments')
        create_arguments(DatasetLoader, group_dataset)

        # method args
        group_method = method_parser.add_argument_group('method arguments')
        create_arguments(method_class, group_method)
        
        # experiment args
        group_expr = method_parser.add_argument_group('experiment arguments')
        create_arguments(run, group_expr)
        create_arguments(Logger.setup, group_expr)

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    kwargs = vars(parser.parse_args())
    print_args(kwargs, num_cols=2)

    try:
        start = time()
        run(**kwargs)
        end = time()
        console.info(f'Total running time: {(end - start):.2f} seconds.')
    except KeyboardInterrupt:
        print('\n')
        console.warning('Graceful Shutdown')
        return 1
    except RuntimeError:
        raise
    finally:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            console.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')
        else:
            console.info('No GPU memory usage statistics available (CUDA is not available)\n')
    import os
    os._exit(0)

if __name__ == '__main__':
    main()
