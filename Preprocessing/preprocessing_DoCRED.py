"""
This file contain methods for handling DoCRED dataset
"""
import json
# import os


def extract_entrel(const_path_file, stop_index) -> tuple[dict[str, str], dict[str, str]]:
    """
    This function open dataset in .json format and extract entity until reach a stop step.
    after create a list of entity that one will be store in .csv file from store_dataset
    :param stop_index: index to extract # of dictionary from the list
    :param const_path_file: file path of train, dev or test .json dataset
    :return: dictionary of all the mention's of entity of the datasets.
    """
    # update with control on prefix path files
    with open(const_path_file, 'r') as reader:
        datas = json.load(reader)  # return list of dicts
        reader.close()

    entity_dict = {}
    relation_dict = {}
    # try to update with lambda functions
    for i in range(0, stop_index):
        data = datas[i]
        entity = data['vertexSet']
        relation = data['labels']
        entity_dict['entity #{}'.format(i)] = entity
        relation_dict['relation #{}'.format(i)] = relation

    return entity_dict, relation_dict


def store_dataset(path_to_save, ent, rel) -> None:
    return
