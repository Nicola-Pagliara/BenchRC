"""
This file contain methods for handling DoCRED dataset
"""
import json
import os

from Support import constant as const


def extract_entrelsen(const_path_file) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    This function open dataset in .json format and extract entity and relationship until reach a stop step.
    after create a list of entity that one will be store in .csv file from store_dataset
    :param const_path_file: file path of train, dev or test .json dataset
    :return: dictionary of all the mention's of entity and relation between them of the datasets.
    """
    # update with control on prefix path files
    with open(const_path_file, 'r') as reader:
        datas = json.load(reader)  # return list of dicts
        reader.close()

    entity_dict = {}
    relation_dict = {}
    sent_dict = {}
    # try to update with lambda functions
    for i in range(0, len(datas) - 1):
        data = datas[i]
        entity = data['vertexSet']
        relation = data['labels']
        sen = data['sents']
        entity_dict['entity #{}'.format(i)] = entity
        relation_dict['relation #{}'.format(i)] = relation
        sent_dict['sent #{}'.format(i)] = sen

    return entity_dict, relation_dict, sent_dict


def store_datasets(path_to_save, ent, rel, sen) -> None:
    """
    :param sen: dictionary of dictionaries that contains the sentences which contains entities and relations
    :param path_to_save: path of directory where the datas will be saved.
    :param ent: dictionary of dictionaries that contains entities and their mentions across various sentences.
    :param rel: dictionary of dictionaries that contains relations between a pair of entity.
    """
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    with open(path_to_save + const.PREFIX_SAVE_ENT, "w") as writer:
        entities = json.dumps(ent)
        writer.write(entities)
        writer.close()

    with open(path_to_save + const.PREFIX_SAVE_REL, "w") as writer:
        relations = json.dumps(rel)
        writer.write(relations)
        writer.close()

    with open(path_to_save + const.PREFIX_SAVE_SEN, "w") as writer:
        sentences = json.dumps(sen)
        writer.write(sentences)
        writer.close()

    return
