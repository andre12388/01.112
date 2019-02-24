import math


# See how to run the thing with all the files.
# The Viterbi Algorithm
# Need to fix the total number of tags that appeared
# Still need to add consideration to those sentence that have less than 3 words.
# Also those with words that are not in the training file.

def get_true_sec_ord_param_dict(param_dict):
    sumyim2im1_dict = {}
    for yim2yim1 in param_dict.keys():
        sumyim2im1_dict[yim2yim1] = sum(param_dict[yim2yim1].values())

    for yim2yim1 in param_dict.keys():
        yim2yim1yi_dict = param_dict[yim2yim1]

        for yi in yim2yim1yi_dict.keys():
            yim2yim1yi_dict[yi] = yim2yim1yi_dict[yi] / sumyim2im1_dict[yim2yim1]
    return param_dict


def sec_ord_trans_param(training_set):
    from tqdm import tqdm
    from collections import defaultdict

    count_yim2yim1yi = 0
    count_yim2yim1 = 0
    trainSetSize = len(training_set)
    i = 0
    param_dict = {}
    #     {(yim2,yim1): {yi: count of yim2yim1yi}}
    with tqdm(total=len(training_set)) as pbar:
        while i < trainSetSize:
            yi = training_set[i].strip().split(" ")[-1]
            # When pointer points at the first tag
            if i == 0:
                yim1 = "START"
                yim2 = ""
            # When pointer points at second tag
            elif i == 1:
                yim1 = training_set[i - 1].strip().split(" ")[-1]
                yim2 = "START"
            # When pointer points at any other tag
            else:
                yim1 = training_set[i - 1].strip().split(" ")[-1]
                yim2 = training_set[i - 2].strip().split(" ")[-1]

            # When pointer reaches a blank, we know we reach the end of a sequence
            if yi == "":
                yi = "STOP"
            # When pointer is at the second tag of the sequence, we want to set yim2 to START, and not when the pointer points at the first tag of the whole list.
            if i > 0 and yim2 == "":
                yim2 = "START"

            # When the pointer point to the first tag on any other sequence aside from the first sequence, set yim2 to "" and yim1 to START
            if yim1 == "":
                yim1 = "START"
                yim2 = ""
            #             if yim1 == "":
            #                 i += 1
            #                 continue

            if str((yim2, yim1)) not in param_dict.keys():
                param_dict[str((yim2, yim1))] = defaultdict(int)

            yim2yim1yi_dict = param_dict[str((yim2, yim1))]
            yim2yim1yi_dict[yi] += 1

            i += 1
            pbar.update(1)
    param_dict = get_true_sec_ord_param_dict(param_dict)
    return param_dict

import json
import numpy as np
import copy


def vitebri_tagging_sec_ord(fileList, i, j, t_params, e_params):
    # i is front pointer
    # j is back pointer
    # k is last pointer
    # listing all possible tags
    listOfTag_tparams = []
    for key in t_params:
        key_value = tuple(key[1:-1].split(','))[0].strip()
        if (key_value not in listOfTag_tparams):
            listOfTag_tparams.append(key_value)

    listOfTag_sec_ord_tparams = []
    for key in t_params:
        listOfTag_sec_ord_tparams.append(key)

    k = copy.deepcopy(j)
    j += 1
    calculation_mem = {}
    while j <= i:
        fileListWord_j = str(fileList[j])
        # check if the word has been learned or not
        if (fileListWord_j in e_params):
            pass
        else:
            fileListWord_j = "#UNK#"

        # found start
        if (not fileList[j - 1] and fileList[j]):
            tempdic = {}
            for tag in listOfTag_tparams:
                value = 0
                tag = tag.replace("'", "")
                if (tag == "STOP" or tag == "START" or tag == ""):
                    continue
                elif (tag in t_params["('', 'START')"] and tag in e_params[fileListWord_j]):  # found catch
                    temp_value = e_params[fileListWord_j][tag] * t_params["('', 'START')"][tag]
                    if (value < temp_value):
                        value = temp_value
                        tempdic[tag] = [value, 'START']
                else:
                    if (value == 0):
                        tempdic[tag] = [0, 'START']
                calculation_mem[j] = tempdic

                # found content
        if (fileList[j - 1] and fileList[j]):
            tempdic = {}
            for tag in listOfTag_tparams:
                tag = tag.replace("'", "")
                value = 0
                for tag_sec in listOfTag_sec_ord_tparams:
                    u = tuple(tag_sec[1:-1].split(','))[1].strip()
                    u = u.replace("'", "")
                    if (u not in calculation_mem[j - 1] or tag not in t_params[tag_sec]):
                        continue
                    else:

                        if (tag == "START" or tag == "STOP" or tag == ""):
                            continue
                        elif (tag not in e_params[fileListWord_j] or tag not in t_params[tag_sec] or
                              calculation_mem[j - 1][u][0] == 0):
                            if (value == 0):
                                tempdic[tag] = [0, u]
                        else:  # found catch

                            temp_value = float(calculation_mem[j - 1][u][0]) * float(t_params[tag_sec][tag]) * float(
                                e_params[fileListWord_j][tag])
                            if value < temp_value:  # finding the max value of u iteration
                                value = temp_value
                                tempdic[tag] = [value, u]
                calculation_mem[j] = tempdic

        # found stop
        if (fileList[j - 1] and not fileList[j]):
            tempdic = {}
            value = 0
            for tag_sec in listOfTag_sec_ord_tparams:
                v = tuple(tag_sec[1:-1].split(','))[1].strip()
                v = v.replace("'", "")
                if (v not in calculation_mem[j - 1]):
                    continue
                else:
                    if (tag_sec in t_params):  # found catch
                        if ("STOP" not in t_params[tag_sec]):
                            continue
                        else:
                            temp_value = calculation_mem[j - 1][v][0] * t_params[tag_sec]['STOP']
                            if value < temp_value:
                                value = temp_value
                                tempdic['STOP'] = [value, v]
                            else:
                                if (value == 0):
                                    tempdic['STOP'] = [0, v]
            calculation_mem[j] = tempdic
            # choosing the tag
            final_tag_sequence = []
            tempTag_highest = "STOP"
            while i > k:
                tag = calculation_mem[i][tempTag_highest][1]
                final_tag_sequence.insert(0, tag)
                tempTag_highest = tag
                i -= 1
            del final_tag_sequence[0]
            return final_tag_sequence

        j += 1


def viterbi_sec_ord(filename, outputDir, e_paramsDir, t_paramsDir):
    json_file = open(e_paramsDir)
    json_str = json_file.read()
    e_params = json.loads(json_str)

    json_file = open(t_paramsDir)
    json_str = json_file.read()
    t_params = json.loads(json_str)

    file = open(filename, "r+", encoding="utf8")
    content = file.readlines()
    if (content[0] != "\n"):
        file.truncate(0)
        file.seek(0, 0)
        file.write("\n")
        for line in content:
            file.write(line)
    else:
        pass
    file.close()

    file = open(filename, "r", encoding="utf8")
    fileList = np.array(file.read().splitlines())
    fileListSize = fileList.size
    i = 1
    j = 0
    wordSequence = []
    with open(outputDir + '/dev.p4.out', 'w',encoding='utf8') as dev_out:
        while i < fileListSize:
            wordSequence.append(fileList[i])
            if (not fileList[i] and fileList[i - 1]):  # found a stop
                tagSequence = vitebri_tagging_sec_ord(fileList, i, j, t_params, e_params)
                wordSequence = list(filter(None, wordSequence))  # remove empty element
                size = 0
                p = 0
                if (len(wordSequence) == len(tagSequence)):
                    size = len(wordSequence)
                else:
                    print("Size Error ... <Doesn't Match>")
                while p < size:
                    dev_out.write('{0} {1}\n'.format(wordSequence[p], tagSequence[p]))
                    p += 1
                dev_out.write("\n")
                j = i
                wordSequence = []
            else:
                pass
            i += 1


import json

language_list = ['CN', 'EN', 'FR', 'SG']
# language_list = ['EN']

for language in language_list:
    fileName = 'data/{}/train'.format(language)
    with open(fileName, 'r',encoding='utf8') as train_file:
        training_set = train_file.readlines()
        print('{} training data is read'.format(language))

        t_params = sec_ord_trans_param(training_set)
        #         pprint.pprint(t_params)
        print('{} second order transmission parameters estimated'.format(language))

        with open('data/{}/sec_ord_trans_parameters.json'.format(language),
                  'w') as f:
            json.dump(t_params, f)
            print('{} second order transmission parameters saved'.format(language))

    # running the vitebri algo
    viterbi_sec_ord("data/{}/dev.in".format(language), "data/{}".format(language),
                    'data/{}/emission_parameters.json'.format(language),
                    'data/{}/sec_ord_trans_parameters.json'.format(language))
    print('{} sequence labelled using viterbi'.format(language))