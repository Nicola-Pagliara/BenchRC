"""
This file contain methods for handling DoCRED dataset  opz wrapping in classes
"""
import json
import os
from typing import Any, Tuple, Dict

from Support import constant as const
import pandas as pd
import numpy as np


def extract_entrelsen(const_path_file) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        sent = data['sents']
        entity_dict['entity #{}'.format(i)] = entity
        relation_dict['relation #{}'.format(i)] = relation
        sen_dict['sent #{}'.format(i)] = sent

    return entity_dict, relation_dict, sen_dict


def store_datasets(path_to_save, ent, rel, sen) -> None:
    """
    store the extracted dataset into separate json files
    :param sen: list of sentences where add entity info
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
        sents = json.dumps(sen)
        writer.write(sents)
        writer.close()

    return


def convert_to_evidence(path_saved_data):
    """
    This method take the json file and create unique csv dataset with entity pair labeled with their relation,
    using only evidence sentences. dataset format -> evidence sent . 1 entity mentions, 2 entity mentions,
    label relation.
    :param path_saved_data: contains base path for both entity with all the mentions and relationships between entity
    in bidirectional way  -> r(e1, e2) and r(e2,e1).
    :return: a dataset with all format evidence sents plus the label relation.
    """

    with open(path_saved_data, 'r') as reader:
        datas = json.load(reader)  # return list of dicts
        reader.close()

    list_text = []
    list_label = []
    for i in range(0, len(datas)):
        data = datas[i]
        entities = data['vertexSet']
        sent = data['sents']
        relations = data['labels']
        # try use idx in labels for pairing entities es extract head and tail idxs for select th correct pair.
        for j in range(0, len(relations)):
            # add method for negative samples i. e. labels = []  and mark them as no relations
            relation = relations[j]
            head_idx = relation['h']
            tail_idx = relation['t']
            evidence_sent = relation['evidence']
            head_ent_mentions = entities[head_idx]
            tail_ent_mentions = entities[tail_idx]
            id_rel = relation['r']
            for k in range(0, len(evidence_sent)):
                # eliminate redundant mentions that will be not part of relations
                id_sent = evidence_sent[k]
                sig_sent = sent[id_sent]
                for s in range(0, len(head_ent_mentions)):
                    mention = head_ent_mentions[s]
                    sig_sent.append(mention['name'])
                for t in range(0, len(tail_ent_mentions)):
                    mentiont = tail_ent_mentions[t]
                    sig_sent.append(mentiont['name'])
                list_text.append(sig_sent)
                list_label.append(id_rel)
            """
    for i in range(0, len(list_text)):
        list_text[i].append(list_label[i])  # update all cycle with lambda function and list compression.
           
           """

    array = np.array(list_text, dtype=object)
    array = array.reshape((len(array), 1))
    array_label = np.array(list_label)
    array_label = array_label.reshape((len(array_label), 1))
    dataset = pd.DataFrame(data=array, columns=['evidence sent . head ent mentions, tail ent mentions'])
    dataset_label = pd.DataFrame(data=array_label, columns=['Relation labels'])
    final_dataset = pd.concat([dataset, dataset_label], axis=1)
    final_dataset.to_csv(const.PREPROCESS_ROOT + '/preprocess_docred.csv', index=False)

    return


def map_label(path_csv, path_rel):
    df = pd.read_csv(path_csv)
    with open(path_rel, 'r') as reader:
        info_label = json.load(reader)
        reader.close()

    df_label = df['Relation labels'].to_numpy()
    counter = 0
    for key, _ in info_label.items():
        info_label[key] = counter
        counter += 1

    for i in range(0, df_label.shape[0]):
        label = df_label[i]
        for key, _ in info_label.items():
            if key == label:
                df_label[i] = info_label[key]
            else:
                continue

    df_label = df_label.reshape((len(df_label), 1))
    maps_labels = pd.DataFrame(data=df_label, columns=['Mapped Labels'])
    mapped_dataset = pd.concat([df, maps_labels], axis=1)
    # mapped_dataset = mapped_dataset.drop(columns=mapped_dataset.columns[0], axis=1)
    # mapped_dataset = mapped_dataset.drop(columns=mapped_dataset.columns[1], axis=1)
    mapped_dataset.to_csv(const.PREPROCESS_ROOT + '/complete_docred.csv', index=False)

    return
