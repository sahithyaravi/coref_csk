from config_expansions import *

import json
import os
import string
import re
import pandas as pd
from collections import defaultdict

from joblib import Parallel, delayed
from tqdm import tqdm

from utils import load_json, save_json
import re
import pandas as pd
import spacy
import textacy

dir = '../datasets/caches'

# Configure spacy models
nlp = spacy.load('en_core_web_md')



def is_person(word):
    living_beings_vocab = ["person", "people", "man", "woman", "girl", "boy", "child"
                           "bird", "cat", "dog", "animal", "insect", "pet"]
    refdoc = nlp(" ".join(living_beings_vocab))
    tokens = [token for token in nlp(word) if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
    avg = 0
    for token2 in tokens:
        for token in refdoc:
            sim = token.similarity(token2)
            if sim == 1:
                return True
            avg += sim
    avg = avg / len(refdoc)
    if avg > 0.5:
        return True
    return False


def get_personx(input_event, use_chunk=True):
    """

    @param input_event:
    @param use_chunk:
    @return:
    """
    # print("sentence is", input_event)
    doc = nlp(input_event)
    # print(doc)
    svos = [svo for svo in textacy.extract.subject_verb_object_triples(doc)]

    if len(svos) == 0:
        if use_chunk:
            logger.info(f'No subject was found for the following sentence: "{input_event}". Using noun chunks.')
            noun_chunks = [chunk for chunk in doc.noun_chunks]

            if len(noun_chunks) > 0:
                personx = noun_chunks[0].text
                # is_named_entity = noun_chunks[0].root.pos_ == "PROP"
                return personx
            else:
                logger.info("Didn't find noun chunks either, skipping this sentence.")
                return ""
        else:
            logger.warning(
                f'No subject was found for the following sentence: "{input_event}". Skipping this sentence')
            return ""
    else:
        subj_head = svos[0][0]
        # is_named_entity = subj_head[0].root.pos_ == "PROP"
        personx = subj_head[0].text
        # " ".join([t.text for t in list(subj_head.lefts) + [subj_head] + list(subj_head.rights)])
        return personx



def test_personx():
    s1 = "A man eating a chocolate covered donut with sprinkles."
    s2 = "A desk with a laptop, monitor, keyboard and mouse."
    print(get_personx(s1))
    print(get_personx(s2))



def lexical_overlap(vocab, s1):
    if not vocab or not s1:
        return 0
    w1 = s1.split()

    for s2 in vocab:
        w2 = s2.split()
        overlap = len(set(w1) & set(w2))/(len(w1)+ 1e-8)
        if overlap > 0.7:
            return True
    return False


def convert_job(sentences, key, exp, relation):
    """

    :param sentences: actual sentence which was expanded
    :param key: index to identify sentences/expansions
    :param exp: expansions of the sentences
    :param srl: if srl should be used for generating person x
    :return:
    """
    context = []
    top_context = []
    seen = set()
    srl = False
    personx = "person"

    # if srl:
    #     personx = get_personx_srl(sentences[key])
    # else:
    #     personx = get_personx(sentences[key].replace("_", ""))  # the sentence expanded by comet
    
    if relation not in excluded_relations:
        top_context.append(relation_map[relation.lower()].replace("{0}", personx).replace("{1}", exp[0])+".")
    for beam in exp:
        source = personx
        target = beam.lstrip().translate(str.maketrans('', '', string.punctuation))
        if relation in atomic_relations and not is_person(source):
            source = "person"
        if target and target != "none" and target not in seen and not lexical_overlap(seen, target):
            sent = relation_map[relation.lower()].replace("{0}", source).replace("{1}", target)+"."
            context.append(sent.capitalize())
            seen.add(target)

    return [context, top_context]


def expansions_to_sentences(expansions, sentences, relation, parallel=False):
    print("Converting expansions to sentences:")
    all_top_contexts = {}
    keys = list(expansions.keys())
    if parallel:
        contexts, top_contexts = zip(*Parallel(n_jobs=-1)(
            delayed(convert_job)(sentences, keys[i], expansions[keys[i]], relation)for i in tqdm(range(len(keys)))))
    else:
        contexts = []
        top_contexts = []
        for i in tqdm(range(len(keys))):
            context, top_context = convert_job(sentences, keys[i], expansions[keys[i]], relation)
            contexts.append(context)
            top_contexts.append(top_context)
    all_contexts = dict(zip(keys, contexts))
    all_top_contexts = dict(zip(keys, top_contexts))

    # print(all_contexts)
    return all_contexts, all_top_contexts


if __name__ == '__main__':
    for split in ['train']:
        relations_used = expansions[split].keys()
        original_sentences = dict(zip(list(sentences[split]["combined_id"].values), list(sentences[split]["sentence"].values)))
        all_sentences = {}

        for relation_name in relations_used:
            print(f"processing relation {relation_name}")
            expansion_dict = expansions[split][relation_name]
            contexts, top_contexts = expansions_to_sentences(expansion_dict, original_sentences, relation_name, parallel=False)
            all_sentences[relation_name] = contexts
        
        dd = defaultdict(list)

        for d in list(all_sentences.values()): # you can list as many input dicts as you want here
            for key, value in d.items():
                dd[key].extend(value)
        
        save_json(f"comet/{split}_exp_sentences_ns.json", dd)
        




