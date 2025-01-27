import json
import argparse
from tqdm import tqdm
import ast
import os
from pathlib import Path

POLLUTERS_TO_DIVIDE_RESULTS = {
    "ConsistentRepresentationPolluter": 'number_of_representations', "UniquenessPolluter": 'distribution_function_name'}
NUM_SEEDS = 5


def main(args):
    with open(args.results, 'r') as f:
        all_results = json.load(f)
    new_results_name = "preprocessed_" + str(args.results)

    for timestamp, results in all_results.items():
        for dataset, ds_res in results.items():
            polluter_configs_results = {}

            for polluter, pollution_res in tqdm(ds_res.items()):
                if not polluter in POLLUTERS_TO_DIVIDE_RESULTS.keys():
                    continue

                for run_config, run_res in pollution_res.items():
                    rc = ast.literal_eval(run_config)

                    relevant_config_attribute = POLLUTERS_TO_DIVIDE_RESULTS[polluter]
                    polluter_config_name = f"{polluter}_{relevant_config_attribute}_{rc[relevant_config_attribute]}"

                    if polluter_configs_results.get(polluter_config_name) is None:
                        polluter_configs_results[polluter_config_name] = {}
                    polluter_configs_results[polluter_config_name][run_config] = run_res

            # ConsistentRepresentationPolluter has a separate polluter for baseline quality, which should be removed and merged into the other polluter configs
            cons_repr_config_names = [p_name for p_name in polluter_configs_results.keys(
            ) if "ConsistentRepresentationPolluter" in p_name]
            baseline_cons_repr_config_name = [
                p_name for p_name in cons_repr_config_names if "{}" in p_name][0]
            # there is only 1 run for the baseline polluter
            baseline_cons_repr_run_config, baseline_cons_repr_run_res = list(
                polluter_configs_results[baseline_cons_repr_config_name].items())[0]
            for p_name in cons_repr_config_names:
                temp_dict = {}
                idx_insert_baseline = len(
                    polluter_configs_results[p_name].keys()) / NUM_SEEDS
                for i, run in enumerate(polluter_configs_results[p_name].keys()):
                    if i % idx_insert_baseline == 0:
                        temp_dict[f"{baseline_cons_repr_run_config}_{i}"] = baseline_cons_repr_run_res
                    temp_dict[run] = polluter_configs_results[p_name][run]
                polluter_configs_results[p_name] = temp_dict

            del polluter_configs_results[baseline_cons_repr_config_name]

            for polluter_config_name, pollution_res in polluter_configs_results.items():
                ds_res[polluter_config_name] = pollution_res
            for polluter in POLLUTERS_TO_DIVIDE_RESULTS.keys():
                if polluter in ds_res:
                    del ds_res[polluter]

    os.makedirs(os.path.dirname(new_results_name), exist_ok=True)
    with open(new_results_name, 'w') as f:
        json.dump(all_results, f,
                  indent=4, sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Script dividing results for polluters with constants to be able to plot them individually')

    parser.add_argument('--results', required=True, type=Path,
                        help='Path to the results.json to plot from.')

    main(parser.parse_args())
