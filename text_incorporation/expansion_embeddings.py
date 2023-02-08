# import hickle as hkl
import numpy as np
import pyhocon
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os
# from config_expansions import *
import sys
sys.path.append('../coref')
import pandas as pd
from model_utils import pad_and_read_bert
from utils import save_pkl_dump, load_json
from random import sample

# These csv files should contain a column called predictions with before and after inferences as follows:
# `Before:Lindsay lohan may have decided to change their life from bad to good. She decided to change her life and seek help for her addiction. She is assessed by the staff at betty ford. She is welcomed by staff members at betty ford.
#  After:She is treated for her addiction.She ends up in the hospital. She no longer has a problem. She engaged in further treatment during her stay at betty ford.She attended daily group therapy meetings.`
#

# Choose whether to embed GPT3 or COMET
commonsense_model = 'gpt3'

def lexical_overlap(vocab, s1, threshold=0.75):
    if not vocab or not s1:
        return 0
    w1 = s1.split()

    for s2 in vocab:
        w2 = s2.split()
        overlap = len(set(w1) & set(w2)) / (len(w1)+1e-8)
        if overlap > threshold:
            return True
    return False



def comet_to_roberta_embeddings(bert_tokenizer, bert_model, comet_inferences_root="comet", embedding_mode="ind", max_inferences=5):
    """

    @param bert_tokenizer:
    @type bert_tokenizer:
    @param bert_model:
    @type bert_model:
    @param gpt3_inferences_root:
    @type gpt3_inferences_root:
    @param embedding_mode:
    @type embedding_mode:
    @return:
    @rtype:
    """
    start_end_embeddings = {}
    continuous_embeddings = {}
    widths = {}
    paths = {
        'train':'../datasets/coref/sentence_ecb_corpus_train.csv',
        'dev': '../datasets/coref/sentence_ecb_corpus_dev.csv',
        'test': '../datasets/coref/sentence_ecb_corpus_test.csv'
    }
    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(paths[split])
        expansions = load_json(f"{comet_inferences_root}/expansions_{split}.json")
        for index, row in  tqdm(df.iterrows(), total=df.shape[0]):
            idx = row["combined_id"]
            key = idx
            sentence = row["sentence"]

            # remove duplicate inferences
            before_infs = list(set([text_processing(e).replace(".", "")+"." for e in expansions["isBefore"][idx]]))
            after_infs = list(set([text_processing(e).replace(".", "")+"." for e in expansions["isAfter"][idx]]))

            # remove inferences with more than 70% overlap
            before_array = []
            after_array = []
            seen = set()
            for inf in before_infs:
                if inf not in seen and not lexical_overlap(seen, inf):
                    before_array.append(inf)
                    seen.add(inf)

            for inf in after_infs:
                if inf not in seen and not lexical_overlap(seen, inf):
                    after_array.append(inf)
                    seen.add(inf)


            # Sort the inferences
            before_array = sorted(before_array[:max_inferences], reverse=False)
            after_array = sorted(after_array[:max_inferences], reverse=False)


            before_array = ["Before, "+ inf.lower() for inf in before_array]
            after_array = ["After, "+ inf.lower() for inf in after_array]
            if not before_array:
                before_array = [row["sentence"]]
            if not after_array:
                after_array = [row["sentence"]]

            print("=======================")
            print(before_array)
            print(after_array)
            print(len(before_array), len(after_array))
            print("\n")

            before_token_ids = [bert_tokenizer.encode(inf) for inf in before_array]
            after_token_ids = [bert_tokenizer.encode(inf) for inf in after_array]
            max_len = max([len(d) for d in (before_token_ids + after_token_ids)])
            before_embeddings, before_lengths = pad_and_read_bert(before_token_ids, bert_model, max_len)
            after_embeddings, after_lengths = pad_and_read_bert(after_token_ids, bert_model, max_len)
            before_embeddings = F.pad(before_embeddings, pad=(0, 0, 0, 0, 0, max_inferences - before_embeddings.shape[0]))
            before_lengths = np.pad(before_lengths, (0, max_inferences - before_lengths.shape[0]), 'constant', constant_values=(0))
            after_embeddings = F.pad(after_embeddings, pad=(0, 0, 0, 0, 0, max_inferences - after_embeddings.shape[0]))
            after_lengths = np.pad(after_lengths, (0, max_inferences - after_lengths.shape[0]), 'constant', constant_values=(0))
            # Stack before and after
            embeddings = torch.cat((before_embeddings, after_embeddings), axis=0)
            lengths = np.concatenate((before_lengths, after_lengths), axis=0)
            print(embeddings.shape, lengths.shape)
            # if torch.equal(before_embeddings, after_embeddings):
            #     print("SAME!")
            starts = embeddings[:, 0, :]
            ends = embeddings[:, -1, :]
            start_ends = torch.hstack((starts, ends))
            start_end_embeddings[key] = start_ends.cpu().detach().numpy()
            # print(start_ends.shape)
            continuous_embeddings[key] = embeddings.cpu().detach().numpy()
            widths[key] = lengths
            torch.cuda.empty_cache()
        save_pkl_dump(f"comet/comet_{embedding_mode}_{split}_startend", start_end_embeddings)
        save_pkl_dump(f"comet/comet_{embedding_mode}_{split}_widths", widths)
        save_pkl_dump(f"comet/comet_{embedding_mode}_{split}_cont", continuous_embeddings)
        print(f"Done {split}")

def gpt3_roberta_embeddings(bert_tokenizer, bert_model, gpt3_inferences_root="gpt3", embedding_mode="ind", max_inferences=5):
    """

    @param bert_tokenizer:
    @type bert_tokenizer:
    @param bert_model:
    @type bert_model:
    @param gpt3_inferences_root:
    @type gpt3_inferences_root:
    @param embedding_mode:
    @type embedding_mode:
    @return:
    @rtype:
    """
    start_end_embeddings = {}
    continuous_embeddings = {}
    widths = {}
    # inference names = output_train.csv, output_dev.csv etc placed under gpt3 folder.
    for split in ['train', 'dev', 'test']:
        df = pd.read_csv(f'{gpt3_inferences_root}/output_{split}.csv')
        df.fillna('', inplace=True)
        print(f"Processing {df.shape[0]} of examples in {split}")
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            key = (row["combined_id"], row["event"])
            all_expansions = row["predictions"]
            inferences = all_expansions.split("After:")

            # Clean up each inference
            for i in range(len(inferences)):
                inferences[i] = text_processing(inferences[i])
            
            # Either before or after are missing
            if len(inferences) != 2:
                inferences.extend([row["sentence"]]*(2-len(inferences)))
                print(key)

            # Before and after
            before_array = [inf.lstrip() + "." for inf in inferences[0].split(".") if len(inf.split()) > 3]
            after_array = [inf.lstrip() + "." for inf in inferences[1].split(".") if len(inf.split()) > 3]
            
            if not before_array:
                before_array = [row["sentence"]]
            if not after_array:
                after_array = [row["sentence"]]

            # Sort the inferences
            before_array = sorted(before_array[:max_inferences], reverse=False)
            after_array = sorted(after_array[:max_inferences], reverse=False)

            # # Prepend Before and After
            before_array = ["Before, "+ inf for inf in before_array]
            after_array = ["After, "+ inf for inf in after_array]

            print("=======================")
            print(before_array)
            print(after_array)
            print(len(before_array), len(after_array))
            print("\n")

            # Tokenize and embed before and after
            if embedding_mode == "condensed":
                # Before and After are condensed into one paragraph
                before_condensed = " ".join(before_array).lstrip(". ") + "."
                after_condensed = " ".join(after_array).lstrip(". ") + "."
                before_token_ids = [bert_tokenizer.encode(before_condensed)]
                after_token_ids = [bert_tokenizer.encode(after_condensed)]
                max_len = max([len(d) for d in (before_token_ids + after_token_ids)])
                before_embeddings, before_lengths = pad_and_read_bert(before_token_ids, bert_model, max_length=max_len)
                after_embeddings, after_lengths = pad_and_read_bert(after_token_ids, bert_model, max_length=max_len)
            else:
                # Before has K inferences, After has K inferences separately - useful for attending to most important inferences
                before_token_ids = [bert_tokenizer.encode(inf) for inf in before_array]
                after_token_ids = [bert_tokenizer.encode(inf) for inf in after_array]
                max_len = max([len(d) for d in (before_token_ids + after_token_ids)])
                before_embeddings, before_lengths = pad_and_read_bert(before_token_ids, bert_model, max_len)
                after_embeddings, after_lengths = pad_and_read_bert(after_token_ids, bert_model, max_len)
                before_embeddings = F.pad(before_embeddings, pad=(0, 0, 0, 0, 0, max_inferences - before_embeddings.shape[0]))
                before_lengths = np.pad(before_lengths, (0, max_inferences - before_lengths.shape[0]), 'constant', constant_values=(0))
                after_embeddings = F.pad(after_embeddings, pad=(0, 0, 0, 0, 0, max_inferences - after_embeddings.shape[0]))
                after_lengths = np.pad(after_lengths, (0, max_inferences - after_lengths.shape[0]), 'constant', constant_values=(0))
        

            # Stack before and after
            embeddings = torch.cat((before_embeddings, after_embeddings), axis=0)
            lengths = np.concatenate((before_lengths, after_lengths), axis=0)
            starts = embeddings[:, 0, :]
            ends = embeddings[:, -1, :]
            start_ends = torch.hstack((starts, ends))
            start_end_embeddings[key] = start_ends.cpu().detach().numpy()
            continuous_embeddings[key] = embeddings.cpu().detach().numpy()
            widths[key] = lengths
            # print(start_ends.shape)
            torch.cuda.empty_cache()

        save_pkl_dump(f"{gpt3_inferences_root}/gpt3_{embedding_mode}_{split}_startend", start_end_embeddings)
        save_pkl_dump(f"{gpt3_inferences_root}/gpt3_{embedding_mode}_{split}_widths", widths)
        save_pkl_dump(f"{gpt3_inferences_root}/gpt3_{embedding_mode}_{split}_cont", continuous_embeddings)
        print(f"Done {split}")
    # hkl.dump(start_end_embeddings, f"gpt3/{split}_e_startend_ns.hkl", mode='w')
    # hkl.dump(widths, f"gpt3/{split}_e_widths_ns.hkl", mode='w')


def roberta_sentence_embeddings(bert_tokenizer, bert_model):
    start_end_embeddings = {}
    continuous_embeddings = {}
    widths = {}
    for split in ['train', 'val']:
        print("Saved embeddings, saving original sentences")
        original_sentences = dict(
            zip(list(sentences[split]["combined_id"].values), list(sentences[split]["sentence"].values)))
        for key, expansions in tqdm(original_sentences.items()):
            sentence = original_sentences[key]
            token_ids = [bert_tokenizer.encode(sentence)]
            # Find all embeddings of the inferences and find start_end_embeddings
            embeddings, lengths = pad_and_read_bert(token_ids, bert_model)
            starts = embeddings[:, 0, :]
            ends = embeddings[:, -1, :]
            start_ends = torch.hstack((starts, ends))
            start_end_embeddings[key] = start_ends.cpu().detach().numpy()
            continuous_embeddings[key] = embeddings.cpu().detach().numpy()
            widths[key] = lengths
        save_pkl_dump(f"comet/{split}_sent_startend", start_end_embeddings)
        save_pkl_dump(f"comet/{split}_sent_cont", continuous_embeddings)
        save_pkl_dump(f"comet/{split}_sent_widths", widths)


def text_processing(inference):
    inference = inference.replace("After:", "")
    inference = inference.replace("Before:", "")
    inference = inference.replace("Before this,", "")
    inference = inference.replace("After this,", "")
    inference = inference.replace("Before", "")
    inference = inference.replace("After", "")
    inference = inference.replace(",", "")
    inference = inference.replace(",", "")
    inference = inference.replace("\n", "")
    inference = inference.strip()
    return inference


if __name__ == '__main__':
    # Check GPU
    if torch.cuda.is_available():
        print("### USING GPU:0")
        device = 'cuda'
    else:
        print("### USING CPU")
        device = 'cpu'

    # Load model based on configuration of pairwise scorer
    bert_tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    bert_model = AutoModel.from_pretrained("roberta-large").to(device)
    bert_model.eval()
    ids = bert_tokenizer.encode('granola bars')
    embeddings, lengths = pad_and_read_bert([ids], bert_model)
    print("Sample roberta embeddings size", embeddings.size())

    if commonsense_model == "comet":
        comet_to_roberta_embeddings(bert_tokenizer, bert_model)
    elif commonsense_model == "gpt3":
        # config = pyhocon.ConfigFactory.parse_file('configs/config_pairwise.json')
        # You can embed individually with "ind" and as one single before or after vector with condensed
        gpt3_roberta_embeddings(bert_tokenizer, bert_model, embedding_mode="ind", max_inferences=5,gpt3_inferences_root='gpt3_fs')
    else:
        raise ValueError("commonsense_model should be one of comet or gpt3")
