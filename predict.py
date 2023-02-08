import argparse
from itertools import product

import pyhocon
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoTokenizer, AutoModel

from conll import write_output_file
from model_utils import *
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder, SimpleFusionLayer
from train_pairwise_scorer import get_all_candidate_spans
from utils import *


def init_models(config, device):
    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                      "span_repr_{}".format(config['model_num'])),
                                         map_location=device))
    span_repr.eval()
    span_scorer = SpanScorer(config).to(device)
    span_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                        "span_scorer_{}".format(config['model_num'])),
                                           map_location=device))
    span_scorer.eval()
    pairwise_scorer = SimplePairWiseClassifier(config).to(device)
    pairwise_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                            "pairwise_scorer_{}".format(config['model_num'])),
                                               map_location=device))
    pairwise_scorer.eval()
    fusion_layer = SimpleFusionLayer(config).to(device)
    if config.include_text:
        fusion_layer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                             "fusion_model_{}".format(config['model_num'])),
                                                map_location=device))
    fusion_layer.eval()

    return span_repr, span_scorer, pairwise_scorer, fusion_layer


def is_included(docs, starts, ends, i1, i2):
    doc1, start1, end1 = docs[i1], starts[i1], ends[i1]
    doc2, start2, end2 = docs[i2], starts[i2], ends[i2]

    if doc1 == doc2 and (start1 >= start2 and end1 <= end2):
        return True
    return False


def remove_nested_mentions(cluster_ids, doc_ids, starts, ends):
    # nested_mentions = collections.defaultdict(list)
    # for i, x in range(len(cluster_ids)):
    #     nested_mentions[x].append(i)

    doc_ids = np.asarray(doc_ids)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    new_cluster_ids, new_docs_ids, new_starts, new_ends = [], [], [], []

    for cluster, idx in cluster_ids.items():
        docs = doc_ids[idx]
        start = starts[idx]
        end = ends[idx]

        for i in range(len(idx)):
            indicator = [is_included(docs, start, end, i, j) for j in range(len(idx))]
            if sum(indicator) > 1:
                continue

            new_cluster_ids.append(cluster)
            new_docs_ids.append(docs[i])
            new_starts.append(start[i])
            new_ends.append(end[i])

    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(new_cluster_ids):
        clusters[cluster_id].append(i)

    return clusters, new_docs_ids, new_starts, new_ends


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_clustering.json')
    parser.add_argument("--dev_best_name", type=str, default="")
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)
    if args.dev_best_name:
        mention_type = config["mention_type"]
        # ('dev_events_model_9_average_0.75_topic_level.conll', 68.22460756651444)
        conll_file = args.dev_best_name.split(".conll")[0]
        # dev_events_model_9_average_0.75_topic_level
        splits = conll_file.split("average")
        model_num = int(splits[0].replace("_", "")[-1])
        thresh = splits[1].replace("_", "")  # 0.75topiclevel
        thresh = float(thresh.replace("topiclevel", ""))
        config["model_num"] = model_num
        config["threshold"] = thresh
        config["split"] = "test"

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['save_path'])
    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'

    # Load models and init clustering
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr, span_scorer, pairwise_scorer, fusion_model = init_models(config, device)
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=config['threshold'])

    # Load data
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    data = create_corpus(config, bert_tokenizer, config.split, is_training=False)
    # Additional embeddings for commonsense
    graph_embeddings_train, graph_embeddings_dev = None, None
    expansions_train, expansions_val = None, None
    expansion_embeddings_train, expansion_embeddings_val = None, None

    if config.include_graph:
        # Graph embeddings for commonsense
        graph_embeddings_dev = load_stored_embeddings(config, split=config.split)

    if config.include_text:
        # Text embeddings for commonsense
        expansions_val, expansion_embeddings_val = load_text_embeddings(config, split=config.split)
        # config = assign_sizes(config)

    doc_ids, sentence_ids, starts, ends = [], [], [], []
    all_topic_predicted_clusters = []
    max_cluster_id = 0

    # Go through each topic

    for topic_num, topic in enumerate(data.topic_list):
        print('Processing topic {}'.format(topic))
        docs_embeddings, docs_length = pad_and_read_bert(data.topics_bert_tokens[topic_num], bert_model)
        topic_spans = get_all_candidate_spans(
            config, bert_model, span_repr, span_scorer, data, topic_num, expansions_val,
            expansion_embeddings_val)
        span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
            config, data, topic_num, docs_embeddings, docs_length)

        doc_id, sentence_id, start, end = span_meta_data
        start_end_embeddings, continuous_embeddings, width = span_embeddings
        width = width.to(device)

        labels = data.get_candidate_labels(doc_id, start, end)

        if config['use_gold_mentions']:
            span_indices = labels.nonzero().squeeze(1)
        else:
            with torch.no_grad():
                span_emb = span_repr(start_end_embeddings, continuous_embeddings, width)
                span_scores = span_scorer(span_emb)

            if config.exact:
                span_indices = torch.where(span_scores > 0)[0]
            else:
                k = int(config['top_k'] * num_of_tokens)
                _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)

            # span_indices, _ = torch.sort(span_indices)

        number_of_mentions = len(span_indices)
        start_end_embeddings = start_end_embeddings[span_indices]
        continuous_embeddings = [continuous_embeddings[i] for i in span_indices]
        width = width[span_indices]
        torch.cuda.empty_cache()

        # Prepare all the pairs for the distance matrix
        first, second = zip(*list(product(range(len(span_indices)), repeat=2)))
        first = torch.tensor(first)
        second = torch.tensor(second)

        torch.cuda.empty_cache()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(first), 1000):
                end_max = i + 1000
                first_idx, second_idx = first[i:end_max], second[i:end_max]
                g1 = span_repr(start_end_embeddings[first_idx],
                               [continuous_embeddings[k] for k in first_idx],
                               width[first_idx])
                g2 = span_repr(start_end_embeddings[second_idx],
                               [continuous_embeddings[k] for k in second_idx],
                               width[second_idx])
                # calculate the keys to look up graph embeddings for this batch
                combined_ids1 = [topic_spans.combined_ids[k]
                                 for k in first_idx]
                combined_ids2 = [topic_spans.combined_ids[k]
                                 for k in second_idx]

                knowledge_embeddings = topic_spans.knowledge_start_end_embeddings, topic_spans.knowledge_continuous_embeddings, topic_spans.knowledge_width
                e1, e2 = None, None  # expansion embeddings
                if config.include_text:
                    if True:  # used to be attention based
                        # If knowledge embeddings need to be represented similar to spans i.e with attention
                        e1, e2 = get_expansion_with_attention(span_repr, knowledge_embeddings, first_idx,
                                                              second_idx, device, config)
                    else:
                        e1 = torch.stack([topic_spans.knowledge_start_end_embeddings[k] for k in first_idx]).to(device)
                        e2 = torch.stack([topic_spans.knowledge_start_end_embeddings[k] for k in second_idx]).to(device)

                g1_final, g2_final, attn_weights = final_vectors(combined_ids1, combined_ids2, config, g1, g2,
                                                                 graph_embeddings_dev, e1, e2, fusion_model)

                scores = pairwise_scorer(g1_final, g2_final)

                torch.cuda.empty_cache()
                if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
                    g1_score = span_scorer(g1)
                    g2_score = span_scorer(g2)
                    scores += g1_score + g2_score

                scores = torch.sigmoid(scores)
                all_scores.extend(scores.detach().cpu().squeeze(1))
                torch.cuda.empty_cache()

        all_scores = torch.stack(all_scores)

        # Affinity score to distance score
        pairwise_distances = 1 - all_scores.view(number_of_mentions, number_of_mentions).numpy()

        if len(pairwise_distances) > 1:
            predicted = clustering.fit(pairwise_distances)
            predicted_clusters = predicted.labels_ + max_cluster_id
        else:
            predicted_clusters = np.array([0] * len(pairwise_distances)) + max_cluster_id

        max_cluster_id = max(predicted_clusters) + 1

        doc_ids.extend(doc_id[span_indices.cpu()])
        sentence_ids.extend(sentence_id[span_indices].tolist())
        starts.extend(start[span_indices].tolist())
        ends.extend(end[span_indices].tolist())
        all_topic_predicted_clusters.extend(predicted_clusters)
        torch.cuda.empty_cache()

    all_clusters = {}
    for i, cluster_id in enumerate(all_topic_predicted_clusters):
        if cluster_id not in all_clusters:
            all_clusters[cluster_id] = []
        all_clusters[cluster_id].append(i)

    if not config['use_gold_mentions']:
        all_clusters, doc_ids, starts, ends = remove_nested_mentions(all_clusters, doc_ids, starts, ends)

    if not config['keep_singletons']:
        all_clusters = {cluster_id: mentions for cluster_id, mentions in all_clusters.items() if len(mentions) > 1}

    print('Saving conll file...')
    doc_name = '{}_{}_{}_{}_model_{}'.format(
        config['split'], config['mention_type'], config['linkage_type'], config['threshold'], config['model_num'])

    write_output_file(data.documents, all_clusters, doc_ids, starts, ends, config['save_path'], doc_name,
                      topic_level=config.topic_level, corpus_level=not config.topic_level)
    print(doc_name + "_topic_level.conll")
