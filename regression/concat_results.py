import argparse
from pathlib import Path
import ast
import json
from tqdm import tqdm
import ast
from pathlib import Path


def main(args):
    with open(args.source, 'r') as f:
        source = json.load(f)
    with open(args.target, 'r') as f:
        target = json.load(f)

    # entries of source file are appended to first timestamp of target file
    target_timestamp = next(iter(target))

    # iterate over all nested keys of source file and append target file at respective dict keys
    for timestamp, results in source.items():
        for dataset, ds_res in results.items():
            for polluter, pollution_res in tqdm(ds_res.items()):
                for run_config, run_res in pollution_res.items():
                    for scenario_name, scenario_res in run_res.items():
                        if scenario_name == 'quality':
                            continue
                        for algorithm_name, algorithm_res in scenario_res.items():
                            try:
                                target[target_timestamp][dataset][polluter][run_config][scenario_name][
                                    algorithm_name] = algorithm_res
                            except KeyError:
                                continue
    # write target file
    with open(args.target, 'w') as f:
        json.dump(target, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    """
    This helper script allows concatenating several result.json files generated by the experiments. This is helpful when
    running several experiments that should be plotted together or need to be aggregated

    :param source (command line parameter): result.json file, which should be concatenated to the first timestamp value of the target file
    :param target (command line parameter): result.json file, where source content should be appended to
    """
    parser = argparse.ArgumentParser(
        'Script concatenating several result.json files.')

    parser.add_argument('--source', required=True, type=Path,
                        help='Path to the results.json source file.')
    parser.add_argument('--target', required=True, type=Path,
                        help='Path to the results.json target file.')

    main(parser.parse_args())
