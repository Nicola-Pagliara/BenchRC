"""
This file contain methods for handling DoCRED dataset  opz wrapping in classes
"""
import json
import os
from typing import Any

from Support import constant as const
import pandas as pd


def extract_entrelsen(const_path_file) -> tuple[dict[str, Any], dict[str, Any]]:
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
    sen_dict = {}
    # try to update with lambda functions
    for i in range(0, len(datas) - 1):
        data = datas[i]
        entity = data['vertexSet']
        relation = data['labels']
        entity_dict['entity #{}'.format(i)] = entity
        relation_dict['relation #{}'.format(i)] = relation


    return entity_dict, relation_dict


def store_datasets(path_to_save, ent, rel) -> None:
    """
    store the extracted dataset into separate json files
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

    return


def convert_to_supervised_dataset(path_saved_data):
    """
    This method take the json files and create unique csv dataset with entity pair labeled with their relation
    :param path_saved_data: contains base path for both entity with all the mentions and relationships between entity
    in bidirectional way  -> r(e1, e2) and r(e2,e1).
    :return: a dataset with all the pair entity labeled with the relations
    """
    with open(path_saved_data + const.PREFIX_SAVE_ENT, 'r') as reader:
        ent_data = json.load(reader)
        reader.close()

    with open(path_saved_data + const.PREFIX_SAVE_REL, 'r') as reader:
        rel_data = json.load(reader)
        reader.close()

    dataset = pd.DataFrame(columns=['Pair entity'])

    for i in range(0, len(ent_data)):
        entity = ent_data[i]
        dataset.append(entity['name'])

    dataset_rel = pd.DataFrame(columns=['relations'])
    for i in range(0, len(rel_data)):
        rel = rel_data[i]
        dataset_rel.append(rel['r'])

    final_dataset = dataset.append(dataset_rel)

    return final_dataset
