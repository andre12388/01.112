import numpy as np

def transition_param(fileList, yim1, yi):
    fileListSize = fileList.size
    i = 1
    countyim1yi = 0
    countyim1 = 0
    while i < fileListSize:
        if(not fileList[i]):
          fileListWord_i = "asdasd"
        else:
          fileListWord_i = fileList[i].split(" ")[-1]
        if(not fileList[i-1]):
          fileListWord_im1 = ""
        else:
          fileListWord_im1 = fileList[i - 1].split(" ")[-1]
        if yim1 == "START":
            if fileListWord_i == yi and fileListWord_im1 == "":
                countyim1yi+=1
            if fileListWord_im1 == "" and fileListWord_i != "":
                countyim1+=1
        elif yi == "STOP":
            if fileListWord_i == "" and fileListWord_im1 == yim1:
                countyim1yi+=1
            if fileListWord_im1 == yim1:
                countyim1+=1
        else:
            if fileListWord_i == yi and fileListWord_im1 == yim1:
                countyim1yi+=1
            if fileListWord_im1 == yim1:
                countyim1+=1
            if i==(fileListSize-1) and fileListWord_i == yim1:
                countyim1+=1
        i+=1
    return float(countyim1yi)/countyim1

def possible_state_identifier(filename):
    combination_dict = {}
    file = open(filename, "r+",encoding="utf8")
    content = file.readlines()
    if(content[0]!="\n"):
        file.truncate(0)
        file.seek(0, 0)
        file.write("\n")
        for line in content:
            file.write(line)
    else:
        pass
    file.close()

    file = open(filename, "r",encoding="utf8")
    fileList = np.array(file.read().splitlines())
    fileListSize = fileList.size
    i = 1
    while i < fileListSize:
        if(not fileList[i]):
          fileListWord_i = "STOP"
        else:
          fileListWord_i = fileList[i].split(" ")[-1]
        if(not fileList[i-1]):
          fileListWord_im1 = "START"
        else:
          fileListWord_im1 = fileList[i - 1].split(" ")[-1]
        if fileListWord_im1 not in combination_dict:
            combination_dict[fileListWord_im1] = {}
            combination_dict.get(fileListWord_im1)[fileListWord_i] = transition_param(fileList,fileListWord_im1,fileListWord_i)
        else:
            if fileListWord_i not in combination_dict.get(fileListWord_im1):
                combination_dict.get(fileListWord_im1)[fileListWord_i] = transition_param(fileList,fileListWord_im1,fileListWord_i)
            else:
                pass
        i+=1
    return combination_dict


import json
import numpy as np


def vitebri_tagging(fileList, i, j, t_params, e_params):
    # i is front pointer
    # j is back pointer
    # k is last pointer
    # listing all possible tags
    listOfTag_tparams = []
    for key in t_params:
        listOfTag_tparams.append(key)

    k = j
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
                if (tag == "STOP" or tag == "START"):
                    continue
                elif (tag in t_params['START'] and tag in e_params[fileListWord_j]):  # found catch
                    value = e_params[fileListWord_j][tag] * t_params['START'][tag]
                    tempdic[tag] = [value, 'START']
                else:
                    tempdic[tag] = [0, 'START']
                    pass
                calculation_mem[j] = tempdic

                # found content
        if (fileList[j - 1] and fileList[j]):
            tempdic = {}
            for tag in listOfTag_tparams:
                value = 0
                for u in calculation_mem[j - 1]:
                    if (tag == "START" or tag == "STOP"):
                        continue
                    elif (tag not in e_params[fileListWord_j] or tag not in t_params[u]):
                        if (value == 0):
                            tempdic[tag] = [0, u]
                    else:  # found catch
                        temp_value = float(calculation_mem[j - 1][u][0]) * float(t_params[u][tag]) * float(
                            e_params[fileListWord_j][tag])
                        if value < temp_value:  # finding the max value of u iteration
                            value = temp_value
                            tempdic[tag] = [value, u]
                calculation_mem[j] = tempdic

        # found stop
        if (fileList[j - 1] and not fileList[j]):
            tempdic = {}
            value = 0
            for v in calculation_mem[j - 1]:
                if (v in t_params):  # found catch
                    if ("STOP" not in t_params[v]):
                        continue
                    else:
                        temp_value = calculation_mem[j - 1][v][0] * t_params[v]['STOP']
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


def viterbi(filename, outputDir, e_paramsDir, t_paramsDir):
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
    with open(outputDir + '/dev.p3.out', 'w',encoding='utf8') as dev_out:
        while i < fileListSize:
            wordSequence.append(fileList[i])
            if (not fileList[i] and fileList[i - 1]):  # found a stop
                tagSequence = vitebri_tagging(fileList, i, j, t_params, e_params)
                wordSequence = list(filter(None, wordSequence))  # remove empty element
                size = 0
                k = 0
                if (len(wordSequence) == len(tagSequence)):
                    size = len(wordSequence)
                else:
                    print("Size Error ... <Doesn't Match>")
                while k < size:
                    dev_out.write('{0} {1}\n'.format(wordSequence[k], tagSequence[k]))
                    k += 1
                dev_out.write("\n")
                j = i
                wordSequence = []
            else:
                pass
            i += 1


import json

language_list = ['CN', 'EN', 'FR', 'SG']

for language in language_list:
    filename = 'data/{}/train'.format(language)
    print('{} training data is read'.format(language))
    t_params = possible_state_identifier(filename)
    print('{} transition parameters estimated'.format(language))

    # saving the t_params json
    with open('data/{}/transition_parameters.json'.format(language),
              'w') as f:
        json.dump(t_params, f)
        print('{} transition parameters saved'.format(language))

    # running the vitebri algo
    viterbi("data/{}/dev.in".format(language), "data/{}".format(language),
            'data/{}/emission_parameters.json'.format(language),
            'data/{}/transition_parameters.json'.format(language))
    print('{} sequence labelled using viterbi'.format(language))
