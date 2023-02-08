import argparse
import collections
import gc
import time
from itertools import combinations

import pyhocon
import wandb
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from evaluator import Evaluation
from model_utils import *
from models import SpanEmbedder, SpanScorer, SimplePairWiseClassifier, SimpleFusionLayer
from spans import TopicSpans
from utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gc.collect()
torch.cuda.empty_cache()


def combine_ids(dids, sids):
    """
    combine documentid , sentenceid into documentid_sentenceid
    @param dids: list of document ids
    @param sids: list of sentence ids
    @return:
    """
    underscores = ['_'] * len(dids)
    dids = list(map(''.join, zip(dids, underscores)))
    combined_ids = list(map(''.join, zip(dids, sids)))
    return combined_ids


def train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings,
                              first, second, labels, batch_size, criterion, optimizer, combined_indices,
                              graph_embeddings, text_knowledge_embeddings, fusion_model=None):
    """
    Training of the pairwise classifier for classifying if two mentions are corefering
    It receives all the span embedidngs, expansion embeddings (optional) , divides into batches and trains the classifier.
    Optionally, it also trains the fusion model for fusing the span and knowledge embeddings.
    @param config:
    @type config:
    @param pairwise_model:
    @type pairwise_model:
    @param span_repr:
    @type span_repr:
    @param span_scorer:
    @type span_scorer:
    @param span_embeddings:
    @type span_embeddings:
    @param first:
    @type first:
    @param second:
    @type second:
    @param labels:
    @type labels:
    @param batch_size:
    @type batch_size:
    @param criterion:
    @type criterion:
    @param optimizer:
    @type optimizer:
    @param combined_indices:
    @type combined_indices:
    @param graph_embeddings:
    @type graph_embeddings:
    @param text_knowledge_embeddings:
    @type text_knowledge_embeddings:
    @param fusion_model:
    @type fusion_model:
    @return:
    @rtype:
    """
    accumulate_loss = 0
    start_end_embeddings, continuous_embeddings, width = span_embeddings
    knowledge_start_end_embeddings, knowledge_continuous_embeddings, knowledge_width = text_knowledge_embeddings
    device = start_end_embeddings.device
    labels = labels.to(device)
    # width = width.to(device)

    idx = shuffle(list(range(len(first))), random_state=config.random_seed)
    for i in range(0, len(first), batch_size):
        indices = idx[i:i + batch_size]
        batch_first, batch_second = first[indices], second[indices]
        batch_labels = labels[indices].to(torch.float)
        optimizer.zero_grad()

        # the look up keys are combined ids we calculated earlier of the form docid_sentenceid
        combined1 = [combined_indices[k] for k in batch_first]
        combined2 = [combined_indices[k] for k in batch_second]

        g1 = span_repr(start_end_embeddings[batch_first],
                       [continuous_embeddings[k] for k in batch_first], width[batch_first])
        g2 = span_repr(start_end_embeddings[batch_second],
                       [continuous_embeddings[k] for k in batch_second], width[batch_second])

        e1, e2 = None, None

        if config.include_text:
            if True:
                # If knowledge embeddings need to be represented similar to spans i.e with attention
                # span_repr, knowledge_embs, batch_first, batch_second, device, config
                e1, e2 = get_expansion_with_attention(span_repr, text_knowledge_embeddings, batch_first, batch_second,
                                                      device, config)
            else:
                # Represent only with start-end embeddings
                e1 = torch.stack([knowledge_start_end_embeddings[k] for k in batch_first]).to(device)
                e2 = torch.stack([knowledge_start_end_embeddings[k] for k in batch_second]).to(device)

        g1_final, g2_final, attention_weights = final_vectors(combined1, combined2, config, g1, g2, graph_embeddings,
                                                              e1, e2,
                                                              fusion_model)
        scores = pairwise_model(g1_final, g2_final)
        # print(scores.squeeze(1))

        if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions'] and not config[
            'exclude_span_repr']:
            print("Span scoring")
            g1_score = span_scorer(g1)
            g2_score = span_scorer(g2)
            scores += g1_score + g2_score

        loss = criterion(scores.squeeze(1), batch_labels)
        accumulate_loss += loss.item()
        loss.backward()
        optimizer.step()

    return accumulate_loss


def get_all_candidate_spans(config, bert_model, span_repr, span_scorer, data, topic_num,
                            expansions=None, expansion_embeddings=None):
    """
    Get all candidate spans in a topic.
    If gold mentions are used, it is the gold mention spans.
    If not, it is the spans chosen based on scores from the span scorer.
    @param config:
    @type config:
    @param bert_model:
    @type bert_model:
    @param span_repr:
    @type span_repr:
    @param span_scorer:
    @type span_scorer:
    @param data:
    @type data:
    @param topic_num:
    @type topic_num:
    @param expansions:
    @type expansions:
    @param expansion_embeddings:
    @type expansion_embeddings:
    @return:
    @rtype:
    """
    docs_embeddings, docs_length = pad_and_read_bert(
        data.topics_bert_tokens[topic_num], bert_model)
    topic_spans = TopicSpans(config, data, topic_num,
                             docs_embeddings, docs_length, is_training=True)

    topic_spans.set_span_labels()
    span_emb = None

    # Pruning the spans according to gold mentions or spans with highest scores
    if config['use_gold_mentions']:
        span_indices = torch.nonzero(topic_spans.labels).squeeze(1)
    else:
        with torch.no_grad():
            span_emb = span_repr(topic_spans.start_end_embeddings,
                                 topic_spans.continuous_embeddings,
                                 topic_spans.width)
            span_scores = span_scorer(span_emb)

        if config.exact:
            span_indices = torch.where(span_scores > 0)[0]
        else:
            k = int(config['top_k'] * topic_spans.num_tokens)
            _, span_indices = torch.topk(
                span_scores.squeeze(1), k, sorted=False)

    span_indices = span_indices.cpu()
    topic_spans.prune_spans(span_indices)

    d_ids = topic_spans.doc_ids.tolist()
    s_ids = topic_spans.sentence_id.squeeze().numpy().astype(str).tolist()

    # combined_ids holds the keys for looking up commonsense embeddings
    topic_spans.combined_ids = combine_ids(d_ids, s_ids)

    if config.include_text:
        if span_emb is not None:
            span_emb_final = span_emb[span_indices]
        else:
            span_emb_final = None
        topic_spans.knowledge_start_end_embeddings, topic_spans.knowledge_continuous_embeddings, topic_spans.knowledge_width, topic_spans.knowledge_text = get_span_specific_embeddings(
            topic_spans, span_repr, expansions, expansion_embeddings,
            span_emb_final, config)

    return topic_spans


def get_pairwise_labels(labels, is_training):
    """
    What are the labels for each pair of spans? Are the pair corefering or not?
    These labels are obtained using the cluster IDS and are required for the pairwise classifier
    @param labels:
    @type labels:
    @param is_training:
    @type is_training:
    @return:
    @rtype:
    """
    first, second = zip(*list(combinations(range(len(labels)), 2)))
    first = torch.tensor(first)
    second = torch.tensor(second)
    pairwise_labels = (labels[first] != 0) & (labels[second] != 0) & \
                      (labels[first] == labels[second])

    if is_training:
        positives = torch.nonzero(pairwise_labels == 1).squeeze()
        positive_ratio = len(positives) / len(first)
        negatives = torch.nonzero(pairwise_labels != 1).squeeze()
        rands = torch.rand(len(negatives))
        rands = (rands < positive_ratio * 20).to(torch.long)
        sampled_negatives = negatives[torch.nonzero(rands).squeeze()]
        new_first = torch.cat((first[positives], first[sampled_negatives]))
        new_second = torch.cat((second[positives], second[sampled_negatives]))
        new_labels = torch.cat(
            (pairwise_labels[positives], pairwise_labels[sampled_negatives]))
        first, second, pairwise_labels = new_first, new_second, new_labels

    pairwise_labels = pairwise_labels.to(torch.long).to(device)

    if config['loss'] == 'hinge':
        pairwise_labels = torch.where(
            pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device))
    else:
        pairwise_labels = torch.where(
            pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))

    return first, second, pairwise_labels


if __name__ == '__main__':
    # Time one whole run
    start = time.time()

    # Config parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/config_pairwise.json')
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.parse_file(args.config)
    fix_seed(config)

    # Logging
    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['model_path'])

    # Weights and biases
    os.environ["WANDB_SILENT"] = "true"
    wandb.login(key='')
    wandb.init(
        project="coref-pairwise",
        # notes="baselin",

    )
    wandb.run.name = f'{config["log_path"].replace("logs/", "")}'

    # GPU number

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    # logger.info('Using device {}'.format(str(config["gpu_num"])))

    # init train and dev set
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    training_set = create_corpus(config, bert_tokenizer, 'train')
    dev_set = create_corpus(config, bert_tokenizer, 'dev')

    # Additional embeddings for commonsense
    graph_embeddings_train, graph_embeddings_dev = None, None
    expansions_train, expansions_val = None, None
    expansion_embeddings_train, expansion_embeddings_val = None, None

    if config.include_graph:
        # Graph embeddings for commonsense
        graph_embeddings_train = load_stored_embeddings(config, split='train')
        graph_embeddings_dev = load_stored_embeddings(config, split='val')

    if config.include_text:
        # Text embeddings for commonsense
        expansions_train, expansion_embeddings_train = load_text_embeddings(config, split='train')
        expansions_val, expansion_embeddings_val = load_text_embeddings(config, split='dev')
        # config = assign_sizes(config)

    # Model initiation
    logger.info('Init models')
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr = SpanEmbedder(config, device).to(device)
    span_scorer = SpanScorer(config).to(device)

    if config['training_method'] in ('pipeline', 'continue') and not config['use_gold_mentions']:
        span_repr.load_state_dict(torch.load(
            config['span_repr_path'], map_location=device))
        span_scorer.load_state_dict(torch.load(
            config['span_scorer_path'], map_location=device))

    span_repr.eval()
    span_scorer.eval()

    # Pairwise classifier model
    pairwise_model = SimplePairWiseClassifier(config).to(device)

    # Fusion models to fuse text embeddings and span representation
    fusion_model = SimpleFusionLayer(config).to(device)

    # Optimizer and loss function
    models = [pairwise_model, fusion_model]
    if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
        models.append(span_repr)
        models.append(span_scorer)

    optimizer = get_optimizer(config, models)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = get_loss_function(config)

    logger.info('Number of parameters of mention extractor: {}'.format(
        count_parameters(span_repr) + count_parameters(span_scorer)))
    logger.info('Number of parameters of the pairwise classifier: {}'.format(
        count_parameters(pairwise_model)))
    logger.info('Number of topics: {}'.format(len(training_set.topic_list)))

    f1 = []
    best_f1 = 0
    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))
        pairwise_model.train()
        fusion_model.train()
        if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
            span_repr.train()
            span_scorer.train()
        accumulate_loss = 0
        list_of_topics = shuffle(list(range(len(training_set.topic_list))))
        total_number_of_pairs = 0
        for topic_num in tqdm(list_of_topics):
            topic = training_set.topic_list[topic_num]
            topic_spans = get_all_candidate_spans(config, bert_model, span_repr, span_scorer,
                                                  training_set, topic_num, expansions_train,
                                                  expansion_embeddings_train)
            first, second, pairwise_labels = get_pairwise_labels(topic_spans.labels, is_training=config['neg_samp'])
            span_embeddings = topic_spans.start_end_embeddings, topic_spans.continuous_embeddings, topic_spans.width
            knowledge_embeddings = topic_spans.knowledge_start_end_embeddings, topic_spans.knowledge_continuous_embeddings, topic_spans.knowledge_width
            loss = train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings, first,
                                             second, pairwise_labels, config['batch_size'], criterion, optimizer,
                                             topic_spans.combined_ids, graph_embeddings_train, knowledge_embeddings,
                                             fusion_model
                                             )

            accumulate_loss += loss
            total_number_of_pairs += len(first)
            torch.cuda.empty_cache()

        logger.info('Number of training pairs: {}'.format(total_number_of_pairs))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))
        wandb.log({'train loss': accumulate_loss})
        scheduler.step()

        logger.info('Evaluate on the dev set')
        span_repr.eval()
        span_scorer.eval()
        pairwise_model.eval()
        fusion_model.eval()
        accumul_val_loss = 0
        all_scores, all_labels = [], []
        count = collections.defaultdict(set)
        # Additional lists for debugging later
        # PER TOPIC SPANS, EXPANSIONS, KEYS
        all_spans, all_span_expansions = [], []
        all_lookups = []
        # PAIRS
        all_pairs1, all_pairs2 = [], []  # pairs of combined_ids
        all_s1, all_s2 = [], []  # pairs of spans
        all_k1, all_k2 = [], []  # pairs of expansions
        all_b1, all_b2, all_a1, all_a2 = [], [], [], []  # pairs of attention weights

        for topic_num, topic in enumerate(tqdm(dev_set.topic_list)):
            topic_spans = get_all_candidate_spans(
                config, bert_model, span_repr, span_scorer, dev_set, topic_num, expansions_val,
                expansion_embeddings_val)
            first, second, pairwise_labels = get_pairwise_labels(
                topic_spans.labels, is_training=False)
            span_embeddings = topic_spans.start_end_embeddings, topic_spans.continuous_embeddings, \
                              topic_spans.width
            topic_spans.width = topic_spans.width.to(device)
            with torch.no_grad():
                for i in range(0, len(first), 1000):
                    end_max = i + 1000
                    first_idx, second_idx = first[i:end_max], second[i:end_max]
                    batch_labels = pairwise_labels[i:end_max]
                    g1 = span_repr(topic_spans.start_end_embeddings[first_idx],
                                   [topic_spans.continuous_embeddings[k]
                                    for k in first_idx],
                                   topic_spans.width[first_idx])
                    g2 = span_repr(topic_spans.start_end_embeddings[second_idx],
                                   [topic_spans.continuous_embeddings[k]
                                    for k in second_idx],
                                   topic_spans.width[second_idx])
                    # calculate the keys to look up graph embeddings for this batch
                    combined_ids1 = [topic_spans.combined_ids[k]
                                     for k in first_idx]
                    combined_ids2 = [topic_spans.combined_ids[k]
                                     for k in second_idx]

                    knowledge_embeddings = topic_spans.knowledge_start_end_embeddings, topic_spans.knowledge_continuous_embeddings, topic_spans.knowledge_width
                    e1, e2 = None, None  # expansion embeddings
                    if config.include_text:
                        if True:
                            # If knowledge embeddings need to be represented similar to spans i.e with attention
                            e1, e2 = get_expansion_with_attention(span_repr, knowledge_embeddings, first_idx,
                                                                  second_idx, device, config)
                        else:
                            e1 = torch.stack([topic_spans.knowledge_start_end_embeddings[k] for k in first_idx]).to(
                                device)
                            e2 = torch.stack([topic_spans.knowledge_start_end_embeddings[k] for k in second_idx]).to(
                                device)

                    g1_final, g2_final, attn_weights = final_vectors(combined_ids1, combined_ids2, config, g1, g2,
                                                                     graph_embeddings_dev, e1, e2, fusion_model)

                    scores = pairwise_model(g1_final, g2_final)

                    loss = criterion(scores.squeeze(1), batch_labels.to(torch.float))
                    accumul_val_loss += loss.item()

                    if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
                        g1_score = span_scorer(g1)
                        g2_score = span_scorer(g2)
                        scores += g1_score + g2_score

                    ############DEBUG#######
                    if config.include_text and config.fusion != "concat" and config.fusion != "random":
                        # print(attn_weights[0].shape)
                        all_b1.extend(attn_weights[0].tolist())
                        all_a1.extend(attn_weights[1].tolist())
                        all_b2.extend(attn_weights[2].tolist())
                        all_a2.extend(attn_weights[3].tolist())
                    # counting = (
                    #     list(zip(combined_ids1, combined_ids2, batch_labels.cpu().detach().numpy())))
                    # for c1, c2, l in counting:
                    #     count[(c1, c2)].add(l)
                    all_pairs1.extend(combined_ids1)
                    all_pairs2.extend(combined_ids2)
                    all_scores.extend(scores.squeeze(1))
                    all_labels.extend(batch_labels.to(torch.int))
                    ###########DEBUG###########
                    torch.cuda.empty_cache()

            #############
            # Save topic-wise information for debugging
            span1 = [topic_spans.span_texts[k] for k in first]
            span2 = [topic_spans.span_texts[k] for k in second]
            all_s1.extend(span1)
            all_s2.extend(span2)
            if config.include_text:
                k1 = [topic_spans.knowledge_text[k] for k in first]
                k2 = [topic_spans.knowledge_text[k] for k in second]
                all_k1.extend(k1)
                all_k2.extend(k2)

            if config.include_text:
                all_span_expansions.extend(topic_spans.knowledge_text)

            all_spans.extend(topic_spans.span_texts)
            all_lookups.extend(topic_spans.combined_ids)
            # Plot the cosine similarity of embeddings for this topic based on labels
            # if config['plot_cosine'] and (epoch == config["epochs"] - 1):
            #     plot_this_batch(span1, span2, c1, c2, pairwise_labels.to(torch.float))
            ###########

        ### Save epoch wise info - models, errors, metrics
        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)
        strict_preds = (all_scores > 0).to(torch.int)
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len(torch.nonzero(all_labels == 1)),
                                                             len(all_labels)))
        eval = Evaluation(strict_preds, all_labels.to(device))

        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                        eval.get_precision(), eval.get_f1()))
        f1.append(eval.get_f1())
        wandb.log({"val loss": accumul_val_loss})
        wandb.log({"f1": eval.get_f1()})
        torch.save(span_repr.state_dict(), os.path.join(
            config['model_path'], 'span_repr_{}'.format(epoch)))
        torch.save(span_scorer.state_dict(), os.path.join(
            config['model_path'], 'span_scorer_{}'.format(epoch)))
        torch.save(pairwise_model.state_dict(), os.path.join(
            config['model_path'], 'pairwise_scorer_{}'.format(epoch)))
        torch.save(fusion_model.state_dict(), os.path.join(
            config['model_path'], 'fusion_model_{}'.format(epoch)))

        # Document wrong predictions of the best model 
        cur_f1 = eval.get_f1()
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            save_error_examples(config, strict_preds, all_labels, all_s1, all_s2, all_pairs1, all_pairs2, all_k1,
                                all_k2)

        # Document attention weights and commonsense expansions
        save_attention_weights(config, all_a1, all_a2, all_b1, all_b2, all_pairs1, all_pairs2, all_s1, all_s2)
        save_span_expansions(config, all_span_expansions, all_spans, all_lookups)

    end = time.time()
    logger.info('Time taken: {}'.format(end - start))
    logger.info('Best F1:{}'.format(round(best_f1, 4)))
