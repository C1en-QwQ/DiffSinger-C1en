import re


def parse_commandline_spk_mix(mix: str) -> dict:
    """
    Parse speaker mix info from commandline
    :param mix: Input like "opencpop" or "opencpop|qixuan" or "opencpop:0.5|qixuan:0.5"
    :return: A dict whose keys are speaker names and values are proportions
    """
    name_pattern = r'[0-9A-Za-z_-]+'
    proportion_pattern = r'\d+(\.\d+)?'
    single_pattern = rf'{name_pattern}(:{proportion_pattern})?'
    assert re.fullmatch(rf'{single_pattern}(\|{single_pattern})*', mix) is not None, f'Invalid mix pattern: {mix}'
    without_proportion = set()
    proportion_map = {}
    for component in mix.split('|'):
        # If already exists
        name_and_proportion = component.split(':')
        assert name_and_proportion[0] not in without_proportion and name_and_proportion[0] not in proportion_map, \
            f'Duplicate speaker name: {name_and_proportion[0]}'
        if ':' in component:
            proportion_map[name_and_proportion[0]] = float(name_and_proportion[1])
        else:
            without_proportion.add(name_and_proportion[0])
    sum_given_proportions = sum(proportion_map.values())
    assert sum_given_proportions < 1 or len(without_proportion) == 0, \
        'Proportion of all speakers should be specified if the sum of all given proportions are larger than 1.'
    for name in without_proportion:
        proportion_map[name] = (1 - sum_given_proportions) / len(without_proportion)
    sum_all_proportions = sum(proportion_map.values())
    assert sum_all_proportions > 0, 'Sum of all proportions should be positive.'
    for name in proportion_map:
        proportion_map[name] /= sum_all_proportions
    return proportion_map

