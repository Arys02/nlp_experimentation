import re


def parse_presto_labels(sentence, target):
    results = {
        "sentence": sentence,
        "words": sentence.split(),
        "labels": [],
        "task": ""
    }

    splited_target = target.split()

    ponctuations = r"?.!'"
    smiley_pattern = r'[:;=8][\-~oO]?[)\]\(dDpPoO/\\]'
    regex_pattern = r"{}|\w+|[{}]".format(smiley_pattern, re.escape(ponctuations))

    splited_sentence = re.findall(regex_pattern, sentence, re.UNICODE | re.IGNORECASE)
    task_content = splited_target[2:-1]

    task_content_dict = {}

    parse_target(task_content, task_content_dict, "")

    results["task"] = splited_target[0]
    results["labels"] = [0 if x not in task_content_dict.keys() else task_content_dict[x] for x in splited_sentence]

    return results


def parse_target(splited_target, task_dict, label):
    if label != "":
        label += "__"

    if len(splited_target) == 0:
        return ""
    while len(splited_target) > 0:
        if splited_target[0] == ")":  # ENDEXPR
            return splited_target[1:]
        elif len(splited_target) > 2 and splited_target[1] == "«":  ## VALUE
            splited_target = parse_value(splited_target[2:], task_dict, label + splited_target[0])
        elif len(splited_target) > 3 and splited_target[2] == "(":  # EXPR
            splited_target = parse_target(splited_target[3:], task_dict, label + splited_target[0])
        else:
            splited_target = splited_target[1:]

    return splited_target


def parse_value(target, task_dict, label):
    i = 0
    value = []
    for word in target:
        if word == "»":
            for val in value:
                task_dict[val] = label
            return target[i + 1:]
        else:
            word = re.split(r'([!.?])', word)
            for w in word:
                value.append(w)
            i += 1
    return target
