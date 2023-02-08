import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)


def prepare_mask(t, min_nonzeros=50):
    """
    Prepare which key elements need to be masked out or ignored during attention computation.
    Out of K inferences, the inferences with lowest non-zero elements will have the mask/ignore flag set to True.
    The # of non_zero elements required to be considered for attention is configurable with min_nonzeros
    """
    t = t.reshape(t.shape[1], t.shape[0], -1)
    # using torch parallel

    # non_zero_elements = torch.count_nonzero(t, dim=2)
    # mask = (non_zero_elements < min_nonzeros).fill_diagonal_(False, wrap=True)
    # print(mask)
    # # to avoid all True rows which result in NAN in softmax of the attention layer
    # # print(mask)
    # # all_zeros_indices = torch.all(mask,dim=1)
    # # # print(all_zeros_indices)
    # # mask[all_zeros_indices] = mask[all_zeros_indices].fill_diagonal_(False)
    # # print(mask)
    # return mask

    # Iterative method:
    mask = []
    for i in range(t.shape[0]):
        current_row = t[i].squeeze(0)
        non_zero_elements = torch.count_nonzero(current_row, dim=1)
        ignore_elements = (non_zero_elements < min_nonzeros)
        if torch.all(ignore_elements):  # if we ignored all elements in the row, just make first element 1 to avoid NAN
            ignore_elements[0] = False
        mask.append(ignore_elements)
    return torch.stack(mask)


class SpanEmbedder(nn.Module):
    def __init__(self, config, device):
        super(SpanEmbedder, self).__init__()
        self.bert_hidden_size = config.bert_hidden_size
        self.with_width_embedding = config.with_mention_width
        self.use_head_attention = config.with_head_attention
        self.device = device
        self.dropout = config.dropout
        self.padded_vector = torch.zeros(self.bert_hidden_size, device=device)
        self.self_attention_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.bert_hidden_size, config.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config.hidden_layer, 1)
        )
        self.self_attention_layer.apply(init_weights)
        self.width_feature = nn.Embedding(5, config.embedding_dimension)

    def pad_continous_embeddings(self, continuous_embeddings):
        max_length = max(len(v) for v in continuous_embeddings)
        padded_tokens_embeddings = torch.stack(
            [torch.cat((emb, self.padded_vector.repeat(max_length - len(emb), 1)))
             for emb in continuous_embeddings]
        )
        masks = torch.stack(
            [torch.cat(
                (torch.ones(len(emb), device=self.device),
                 torch.zeros(max_length - len(emb), device=self.device)))
                for emb in continuous_embeddings]
        )
        return padded_tokens_embeddings, masks

    def forward(self, start_end, continuous_embeddings, width):
        vector = start_end
        # print([v.size() for v in vector])
        if self.use_head_attention:
            padded_tokens_embeddings, masks = self.pad_continous_embeddings(continuous_embeddings)
            attention_scores = self.self_attention_layer(padded_tokens_embeddings).squeeze(-1)
            attention_scores *= masks
            attention_scores = torch.where(attention_scores != 0, attention_scores,
                                           torch.tensor(-9e9, device=self.device))
            attention_scores = F.softmax(attention_scores, dim=1)
            weighted_sum = (attention_scores.unsqueeze(-1) * padded_tokens_embeddings).sum(dim=1)
            vector = torch.cat((vector, weighted_sum), dim=1)

        if self.with_width_embedding:
            width = torch.clamp(width, max=4)
            width_embedding = self.width_feature(width)
            vector = torch.cat((vector, width_embedding), dim=1)

        return vector


class SpanScorer(nn.Module):
    def __init__(self, config):
        super(SpanScorer, self).__init__()
        self.input_layer = config.bert_hidden_size * 3
        if config.with_mention_width:
            self.input_layer += config.embedding_dimension
        self.mlp = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_layer, config.hidden_layer),
            # nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(config.hidden_layer, 1)
        )
        self.mlp.apply(init_weights)

    def forward(self, span_embedding):
        return self.mlp(span_embedding)


class SimplePairWiseClassifier(nn.Module):
    def __init__(self, config):
        super(SimplePairWiseClassifier, self).__init__()
        self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2

        if config.with_mention_width:
            self.input_layer += config.embedding_dimension

        if config.include_graph:
            # set knowledge_embedding_dimension=1024,relations_per_sentence=0 when using COMET sentence embeddings
            if config.exclude_span_repr:
                # exclude span representation, use only the knowledge embedding
                self.input_layer = 0
            self.input_layer += config.knowledge_embedding_dimension * (config.relations_per_sentence + 1)

        if config.include_text:
            if config.exclude_span_repr:
                # exclude span representation, use only the knowledge embedding
                self.input_layer = 0
            # configure the # of expansions uding n_inferences and embedding dimension = 2048(start-end)/3092(attention)
            self.input_layer += config.expansion_dimension * (config.relations_per_sentence)
        print("INPUT of pairwise classifier", self.input_layer)
        self.input_layer *= 3
        self.hidden_layer = config.hidden_layer
        self.pairwise_mlp = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.Dropout(config.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )
        self.pairwise_mlp.apply(init_weights)

    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))


class SimpleFusionLayer(nn.Module):
    def __init__(self, config):
        self.num_heads = 1
        self.embed_dim = 3092
        super(SimpleFusionLayer, self).__init__()
        self.input_layer = config.bert_hidden_size * 3 if config.with_head_attention else config.bert_hidden_size * 2

        if config.with_mention_width:
            self.input_layer += config.embedding_dimension

        self.final_layer = self.input_layer
        self.dim_layer = nn.Linear(self.embed_dim, 1024)
        if config.fusion == "inter_intra":
            fusion_input = int(config.n_inferences) + 2
        else:
            fusion_input = int(config.n_inferences / 2) + 2
        self.input_layer *= fusion_input
        if config.fusion == "linear":
            self.fusion = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Linear(self.embed_dim, self.final_layer),
                nn.ReLU(),
            )
        else:
            self.fusion = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=0.1)
        self.fusion.apply(init_weights)
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

    def forward(self, first, second, config):
        if config.fusion == "linear":
            return self.fusion(torch.cat((first, second), dim=1))
        else:
            # print("first, second", first.shape, second.shape)
            # Sequence length * Batch size * embedding dimension
            """
             If specified, a mask of shape (N, S) indicating which elements within key to ignore for the purpose of attention (i.e. treat as “padding”). 
            """

            query = first.reshape(int(first.shape[1] / self.embed_dim), first.shape[0], -1)
            key = second.reshape(int(second.shape[1] / self.embed_dim), second.shape[0], -1)
            value = second.reshape(int(second.shape[1] / self.embed_dim), second.shape[0], -1)
            query = self.norm(query)
            key = self.norm(key)
            value = self.norm(value)
            key_padding_mask = prepare_mask(key)
            # print(mask_zeros.shape)
            attn_output, attn_output_weights = self.fusion(query, key, value, key_padding_mask=key_padding_mask)
            attn_weights = attn_output_weights.reshape(first.shape[0], -1).cpu().detach().numpy()
            attn_weights = np.around(attn_weights, 4)
            if config.reduce_attention_output:
                # reduce the attention output to 1024 dimensions
                attn_output = self.dim_layer(attn_output)
            return attn_output.squeeze(0).reshape(first.shape[0], -1), attn_weights
