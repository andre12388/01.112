def estimate_emission_parameter(training_set, x, y):
    """estimate the emission parameter from the training set using Maximum
    Likelihood Estimation (MLE)"""
    count_y2x = 0
    count_y = 0

    for emission_pair in training_set:
        emission_pair = emission_pair.strip()
        if len(emission_pair) == 0:
            continue

        if emission_pair.split(' ')[-1] == y:
            count_y += 1

            if ' '.join(emission_pair.split(' ')[:-1]) == x:
                count_y2x += 1

    if count_y == 0:
        return 0.0

    return count_y2x / count_y


def estimate_smoothed_emission_parameter(training_set, x, y, k=1):
    """estimate the smoothed emission parameter from the training set using Maximum
    Likelihood Estimation (MLE)"""
    count_y2x = 0
    count_y = k

    for emission_pair in training_set:
        emission_pair = emission_pair.strip()
        if len(emission_pair) == 0:
            continue

        if emission_pair.split(' ')[-1] == y:
            count_y += 1

            if ' '.join(emission_pair.split(' ')[:-1]) == x:
                count_y2x += 1

    if x == '#UNK#':
        count_y2x = k

    return count_y2x / count_y


def get_true_param_dict(param_dict, y_count_dict, k):
    for x in param_dict.keys():
        y2x_dict = param_dict[x]

        for y in y2x_dict.keys():
            y2x_dict[y] = y2x_dict[y] / (y_count_dict[y] + k)
            param_dict['#UNK#'][y] = k / (y_count_dict[y] + k)

    return param_dict


def estimate_smoothed_emission_parameters(training_set, k=1):
    from tqdm import tqdm
    from collections import defaultdict

    param_dict = {'#UNK#': {}}
    # structure of param_dict:
    # {x: {y: count of y->x}}

    y_count_dict = defaultdict(int)
    # {y: number of occurences for this y}
    # y_count_dict.keys() -> all the tags

    with tqdm(total=len(training_set)) as pbar:
        for emission_pair in training_set:
            pbar.update(1)
            emission_pair = emission_pair.strip()
            if len(emission_pair) == 0:
                continue

            x, y = ' '.join(emission_pair.split(' ')[:-1]), \
                   emission_pair.split(' ')[-1]
            y_count_dict[y] += 1

            if x not in param_dict.keys():
                param_dict[x] = defaultdict(int)

            y2x_dict = param_dict[x]
            y2x_dict[y] += 1

    param_dict = get_true_param_dict(param_dict, y_count_dict, k)
    return param_dict


def do_simple_sequence_labelling(emission_parameters: dict,
                                 x_sequence: list):
    labelled_sequence = []

    for word in x_sequence:
        if word not in emission_parameters:
            word = '#UNK#'

        y2x_dict = emission_parameters[word]
        argmax_y = max(y2x_dict, key=lambda key: y2x_dict[key])
        labelled_sequence.append(argmax_y)

    return labelled_sequence


def label_file_sequence(emission_parameters, x_file, output_dir):
    with open(output_dir + '/dev.p2.out', 'w',encoding='utf8') as dev_out:
        for word in x_file.readlines():
            word = word.strip()

            if len(word) == 0:
                dev_out.write('\n')
                continue

            if word not in emission_parameters:
                y2x_dict = emission_parameters['#UNK#']
            else:
                y2x_dict = emission_parameters[word]

            argmax_y = max(y2x_dict, key=lambda key: y2x_dict[key])

            dev_out.write('{0} {1}\n'.format(word, argmax_y))


import json
language_list = ['CN', 'EN', 'FR', 'SG']

for language in language_list:
    with open('data/{}/train'.format(language), 'r',encoding='utf8') as train_file:
        training_set = train_file.readlines()
        print('{} training data is read'.format(language))

        e_params = estimate_smoothed_emission_parameters(training_set, k=10)
        print('{} emission parameters estimated'.format(language))

        with open('data/{}/emission_parameters.json'.format(language),
                  'w') as f:
            json.dump(e_params, f)
            print('{} emission parameters saved'.format(language))

        with open('data/{}/dev.in'.format(language), 'r',encoding='utf8') as dev_in:
            label_file_sequence(e_params, dev_in, 'data/{}'.format(language))
            print('{} sequence labelled'.format(language))