# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/models.ipynb (unless otherwise specified).

__all__ = ['TokenEmbeddings', 'PositionalEncoding', 'TransformerEmbeddings', 'PositionwiseFeedForwardNetwork',
           'ScaledDotProductAttention', 'MultiHeadAttention', 'LayerNormalization', 'EncoderLayer', 'Encoder',
           'DecoderLayer', 'Decoder', 'TransformerOutputLayer', 'Transformer']

# Cell
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

# Cell
class TokenEmbeddings(nn.Module):
    """

    Chapter 3.4 - we use learned embeddings to convert the input tokens
    and output tokens to vectors of dimension dmodel. [...]

    """

    def __init__(self, vocab_size, model_dim, multiply_dim_sqrt=False):
        super().__init__()

        self.model_dim_sqrt = math.sqrt(model_dim)
        self.multiply_dim_sqrt = multiply_dim_sqrt

        self.embeddings = nn.Embedding(vocab_size, model_dim)

    def forward(self, input_ids):
        """ Chapter 3.4 - In the embedding layers, we multiply those weights by √dmodel """

        # NOTE:
        # In my experiments I found that multiplying the weights by √dmodel
        # gives worst performance, that's why this is optional

        if self.multiply_dim_sqrt:
            return self.embeddings(input_ids) * self.model_dim_sqrt

        return self.embeddings(input_ids)

# Cell
class PositionalEncoding(nn.Module):

    """
    Chapter 3.5 - The positional encodings have the same dimension dmodel
    as the embeddings, so that the two can be summed.
    In this work, we use sine and cosine functions of different frequencies:
        PE(pos, 2i) = sin(pos / (10000 ** (2i/dmodel)))
        PE(pos, 2(i+1)) = cos(pos / (10000 ** (2i/dmodel)))
    """

    # NOTE:
    #    pos - token position in sequence
    #    i - embedding column mapping

    def __init__(self, model_dim, maximum_sequence_length, learnable=False):
        super().__init__()

        self.model_dim = model_dim
        self.learnable = learnable
        self.maximum_sequence_length = maximum_sequence_length

        if self.learnable:
            """
            Chapter 3.5 - We also experimented with using learned
            positional embeddings [9] instead, and found that the two
            versions produced nearly identical results
            """
            self.positional_encoding_table = nn.Embedding(maximum_sequence_length, model_dim)

        else:
            positions_in_sequence = torch.arange(0, maximum_sequence_length).unsqueeze(1)

            i = torch.arange(0, model_dim // 2, 1)
            frequencies = torch.pow(10000, (2 * i) / model_dim)

            positional_encoding_table = torch.zeros(self.maximum_sequence_length, self.model_dim)
            positional_encoding_table[:, 0::2] = torch.sin(positions_in_sequence / frequencies)
            positional_encoding_table[:, 1::2] = torch.cos(positions_in_sequence / frequencies)

            self.register_buffer('positional_encoding_table', positional_encoding_table)

    def forward(self, input_ids):
        # NOTE:
        # get batch sequence length from input_ids
        sequence_length = input_ids.shape[1]

        if self.learnable:
            device = next(self.parameters()).device
            positions = torch.arange(0, sequence_length).to(device)
            return self.positional_encoding_table(positions)

        return self.positional_encoding_table[:sequence_length]

# Cell
class TransformerEmbeddings(nn.Module):
    """
    Chapter 3.5 - To this end, we add "positional encodings" to the input embeddings
    at the bottoms of the encoder and decoder stacks. The positional encodings have
    the same dimension dmodel as the embeddings, so that the two can be summed

    Chapter 5.4 - In addition, we apply dropout to the sums of the embeddings and the
    positional encodings in both the encoder and decoder stacks. For the base model,
    we use a rate of Pdrop = 0.1
    """

    def __init__(self, model_dim, vocab_size, maximum_sequence_length, learnable_positional_endocings=False, multiply_token_emb_by_dim_sqrt=False):
        super().__init__()

        self.token_embeddings = TokenEmbeddings(vocab_size, model_dim, multiply_token_emb_by_dim_sqrt)
        self.positional_encodings = PositionalEncoding(model_dim, maximum_sequence_length, learnable=learnable_positional_endocings)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        return self.dropout(self.token_embeddings(input_ids) * self.positional_encodings(input_ids))

# Cell
class PositionwiseFeedForwardNetwork(nn.Module):

    """
    Chapter 3.3 - This consists of two linear transformations with
    a ReLU activation in between. [...] The dimensionality of input
    and output is dmodel = 512, and the inner-layer has dimensionality
    dff = 2048.
    """

    def __init__(self, model_dim, inner_layer_dim=2048):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, inner_layer_dim)
        self.relu = nn.ReLU()
        self.linear_2  = nn.Linear(inner_layer_dim, model_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x

# Cell
class ScaledDotProductAttention(nn.Module):

    """
    Chapter 3.2.1 - The input consists of queries and keys
    of dimension dk, and values of dimension dv.
    """

    def __init__(self, key_dim):
        super().__init__()
        self.key_dim = key_dim

    def forward(self, query, key, value, mask=None):

        """ Chapter 3.2.1 - We compute the dot products of the query with all keys """
        attention_coefficients = torch.matmul(query, key.transpose(1, 2))

        """
        Chapter 3.2.1 - divide each by √dk [...]
        We suspect that for large values of dk, the dot products grow large
        in magnitude, pushing the softmax function into regions where it has
        extremely small gradients 4. To counteract this effect, we scale the
        dot products by
        """
        attention_coefficients = attention_coefficients / math.sqrt(self.key_dim)

        """
        Chapter 3.2.3 - We need to prevent leftward information flow in the decoder to
        preserve the auto-regressive property. We implement this inside of scaled
        dot-product attention by masking out (setting to −∞) all values in the input
        of the softmax which correspond to illegal connections.
        """

        if mask is not None:
            attention_coefficients = attention_coefficients.masked_fill(mask == 0, -float("inf"))

        """ Chapter 3.2.1 - and apply a softmax function to obtain the weights on the values """
        output = torch.softmax(attention_coefficients, dim=-1) @ value

        return output

# Cell
class MultiHeadAttention(nn.Module):

    """
    Chapter 3.2.2 - we found it beneficial to linearly project the queries,
    keys and values h times with different, learned linear projections to dk,
    dk and dv dimensions, respectively.

    Chapter 3.2.2 - In this work we employ h = 8 parallel attention layers, or heads.
    For each of these we use dk = dv = dmodel/h = 64
    """

    def __init__(self, model_dim, num_of_heads=8):

        super().__init__()
        assert model_dim % num_of_heads == 0, "model_dim should be divisible by num_of_heads"

        self.num_of_heads = num_of_heads
        self.head_dim = int(model_dim / num_of_heads)

        self.attention = ScaledDotProductAttention(self.head_dim)

        """
        Chapter 3.2.2 - MultiHead(Q, K, V) = Concat(head_1, ..., head_n)Wo
        where head_i = Attention(Q * Wq_i, K * Wk_i, V * Wv_i)
        """

        # NOTE 1:
        # I'm setting bias=False, because in the paper there is no
        # mention about bias. There are only matrices Wq, Wk, Wv and Wo.

        # NOTE 2:
        # nn.ParameterList() and nn.Parameter() can be used instead of nn.ModuleList and nn.Linear()
        # and perform multiplication of created nn.Parameter() and inputs (x @ W)

        # NOTE 3:
        # I'm using nn.ModuleList() with num_heads x nn.Linear() to follow
        # the paper. You can use nn.Linear(model_dim, model_dim) and then reshape
        # the output to get (num_heads x model_dim / num_heads) this will be more efficient,
        # but i want to make the impolementation more clear to understand.

        """
        Chapter 3.2.2 - Where the projections are parameter matrices
            Wq_i ∈ R(dmodel×dk),
            Wk_i ∈ R(dmodel×dk),
            Wv_i ∈ R(dmodel×dv)
            and Wo ∈ R(h*dv×dmodel) .
        """

        self.all_Wq = nn.ModuleList([nn.Linear(model_dim, self.head_dim, bias=False) for _ in range(self.num_of_heads)])
        self.all_Wk = nn.ModuleList([nn.Linear(model_dim, self.head_dim, bias=False) for _ in range(self.num_of_heads)])
        self.all_Wv = nn.ModuleList([nn.Linear(model_dim, self.head_dim, bias=False) for _ in range(self.num_of_heads)])

        self.output_projection = nn.Linear(model_dim, model_dim, bias=False)


    def forward(self, query, key, value, mask=None):

        """
        Chapter 3.2.2 - On each of these projected versions of
        queries, keys and values we then perform the attention
        function in parallel, yielding dv -dimensional output values.
        These are concatenated and once again projected, resulting in
        the final values
        """

        output = torch.cat(
            [
                self.attention(
                    Wq_i(query), Wk_i(key), Wv_i(value), mask
                )
                for Wq_i, Wk_i, Wv_i in zip(
                    self.all_Wq,
                    self.all_Wk,
                    self.all_Wv,
                )
            ],
            dim=-1,
        )


        output = self.output_projection(output)

        return output

# Cell
class LayerNormalization(nn.Module):

    """
    Chapter 3.1 - We employ a residual connection around each of the two sub-layers,
    followed by layer normalization. That is, the output of each sub-layer is
    LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented
    by the sub-layer itself. To facilitate these residual connections, all
    sub-layers in the model, as well as the embedding layers, produce
    outputs of dimension dmodel = 512.
    """

    def __init__(self, model_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, residual_input, sublayer_output):
        # NOTE:
        # residual_input - this are the hidden states berfore the sublayer
        # sublayer_output - this are the hidden states after the sublayer,
        # where sublayers are: MultiHeadAttention or PositionwiseFeedForwardNetwork

        return self.layer_norm(residual_input + sublayer_output)

# Cell
class EncoderLayer(nn.Module):
    """
    Each layer has two sub-layers. The first is a
    multi-head self-attention mechanism, and the
    second is a simple, position-wise fully connected
    feed-forward network.
    """

    def __init__(self, model_dim, num_of_heads, ffn_inner_dim=2048):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(model_dim, num_of_heads)
        self.feed_forward_network = PositionwiseFeedForwardNetwork(model_dim, inner_layer_dim=ffn_inner_dim)

        self.multihead_attention_normalization = LayerNormalization(model_dim)
        self.feedforward_network_normalization = LayerNormalization(model_dim)

    def forward(self, encoder_input, mask):
        """
        Chapter 3.1 - We employ a residual connection around each of
        the two sub-layers, followed by layer normalization
        """

        # NOTE 1:
        # encoder_input - this is the summation of input embeddings with positional encodings

        multihead_attention_output = self.multihead_attention(encoder_input, encoder_input, encoder_input, mask)

        # NOTE 2:
        # residual connection of MultiHeadAttention input (encoder_input)
        # and MultiHeadAttention output
        multihead_attention_norm_output = self.multihead_attention_normalization(
            residual_input=encoder_input,
            sublayer_output=multihead_attention_output
        )

        feedforward_network_output = self.feed_forward_network(multihead_attention_norm_output)
        # NOTE 3:
        # residual connection of PositionwiseFeedForwardNetwork input (multihead_attention_norm_output)
        # and PositionwiseFeedForwardNetwork output
        encoder_output = self.feedforward_network_normalization(
            residual_input=multihead_attention_norm_output,
            sublayer_output=feedforward_network_output
        )

        return encoder_output

# Cell
class Encoder(nn.Module):
    """
    The encoder is composed of a stack of N = 6 identical layers.
    """

    def __init__(self, N=6, model_dim=512, num_of_heads=8, ffn_inner_dim=2048):
        super().__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_of_heads, ffn_inner_dim=ffn_inner_dim) for _ in range(N)])

    def forward(self, encoder_input, mask=None):

        """
        Chapter 3.2.3 - In a self-attention layer all of the keys, values
        and queries come from the same place, in this case, the output
        of the previous layer in the encoder.
        """

        # NOTE 1:
        # renaming only to get better intuition
        encoder_layer_input = encoder_input

        for encoder_layer in self.encoder_layers:
            encoder_layer_input = encoder_layer(encoder_layer_input, mask)
        else:
            # NOTE 2:
            # same as NOTE 1
            encoder_output = encoder_layer_input

        return encoder_output


# Cell
class DecoderLayer(nn.Module):
    """
    Chapter 3.1 - In addition to the two sub-layers in each encoder layer,
    the decoder inserts a third sub-layer, which performs multi-head
    attention over the output of the encoder stack
    """

    def __init__(self, model_dim, num_of_heads, ffn_inner_dim=2048):
        super().__init__()
        self.masked_multihead_attention = MultiHeadAttention(model_dim, num_of_heads)
        self.multihead_attention = MultiHeadAttention(model_dim, num_of_heads)
        self.feed_forward_network = PositionwiseFeedForwardNetwork(model_dim, inner_layer_dim=ffn_inner_dim)

        self.masked_multihead_attention_normalization = LayerNormalization(model_dim)
        self.multihead_attention_normalization = LayerNormalization(model_dim)
        self.feed_forward_network_normalization = LayerNormalization(model_dim)

    def forward(self, decoder_input, encoder_output, mask):

        """
        Chapter 3.2.3 - In "encoder-decoder attention" layers, the queries come
        from the previous decoder layer, and the memory keys and values come
        from the output of the encoder
        """

        # NOTE 1:
        # decoder_input - this is summation of output embeddings with positional encodings.
        masked_multihead_attention_output = self.masked_multihead_attention(decoder_input, decoder_input, decoder_input, mask)

        # NOTE 2:
        # residual connection of MaskedMultiHeadAttention input (decoder_input)
        # and MaskedMultiHeadAttention output
        masked_multihead_attention_norm_output = self.masked_multihead_attention_normalization(
            residual_input=decoder_input,
            sublayer_output=masked_multihead_attention_output
        )

        multihead_attention_output = self.multihead_attention(masked_multihead_attention_norm_output, encoder_output, encoder_output)

        # NOTE 3:
        # residual connection of MultiHeadAttention input (masked_multihead_attention_norm_output)
        # and MultiHeadAttention output
        multihead_attention_norm_output = self.multihead_attention_normalization(
            residual_input=masked_multihead_attention_norm_output,
            sublayer_output=multihead_attention_output
        )

        feedforward_network_output = self.feed_forward_network(multihead_attention_norm_output)

        # NOTE 4:
        # residual connection of PositionwiseFeedForwardNetwork input (multihead_attention_norm_output)
        # and MultiHeadAttention output
        decoder_output = self.feed_forward_network_normalization(
            residual_input=multihead_attention_norm_output,
            sublayer_output=feedforward_network_output
        )

        return decoder_output

# Cell
class Decoder(nn.Module):
    """
    The decoder is also composed of a stack of N = 6 identical layers.
    """

    def __init__(self, N=6, model_dim=512, num_of_heads=8, maximum_sequence_length=128, ffn_inner_dim=2048):
        super().__init__()

        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_of_heads, ffn_inner_dim=ffn_inner_dim) for _ in range(N)])
        mask = torch.tril(torch.ones(maximum_sequence_length, maximum_sequence_length)).bool()
        self.register_buffer("mask", mask)

    def _get_decoder_mask(self, padding_mask, sequence_length):
        return (padding_mask & self.mask[:sequence_length, :sequence_length]).int()

    def forward(self, decoder_input, encoder_output, mask=None):
        sequence_length = decoder_input.shape[1]

        # NOTE 1:
        # Connect decoder mask with padding mask
        if mask is not None:
            mask = self._get_decoder_mask(mask, sequence_length)

        # NOTE 2:
        # renaming only to get better intuition
        decoder_layer_input = decoder_input

        for decoder_layer in self.decoder_layers:
            decoder_layer_input = decoder_layer(decoder_layer_input, encoder_output, mask)
        else:
            # NOTE 2:
            # same as NOTE 1
            decoder_output = decoder_layer_input

        return decoder_output

# Cell
class TransformerOutputLayer(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        if self.training:
            return self.linear(x)

        return self.linear(x)

# Cell
class Transformer(nn.Module):
    def __init__(
        self,
        model_dim,
        src_vocab_size,
        trg_vocab_size,
        ffn_inner_dim=2048,
        N_encoder_layers=6,
        N_decoder_layers=6,
        num_of_attention_heads=8,
        maximum_sequence_length=128,
        learnable_positional_endocings=False
    ):

        super().__init__()

        self.encoder_input_embeddings = TransformerEmbeddings(
            model_dim,
            vocab_size=src_vocab_size,
            maximum_sequence_length=maximum_sequence_length,
            learnable_positional_endocings=learnable_positional_endocings
        )

        self.decoder_input_embeddings = TransformerEmbeddings(
            model_dim,
            vocab_size=trg_vocab_size,
            maximum_sequence_length=maximum_sequence_length,
            learnable_positional_endocings=learnable_positional_endocings
        )

        self.encoder = Encoder(
            N=N_encoder_layers,
            model_dim=model_dim,
            num_of_heads=num_of_attention_heads,
            ffn_inner_dim=ffn_inner_dim
        )

        self.decoder = Decoder(
            N=N_decoder_layers,
            model_dim=model_dim,
            num_of_heads=num_of_attention_heads,
            ffn_inner_dim=ffn_inner_dim
        )

        self.transformer_output_layer = TransformerOutputLayer(model_dim, trg_vocab_size)


    def forward(self, input_ids, target_ids, input_padding_mask, target_padding_mask):

        batch_size = input_ids.shape[0]

        input_mask = input_padding_mask.view(batch_size, 1, -1)
        target_mask = target_padding_mask.view(batch_size, 1, -1)

        encoder_input_embeddings = self.encoder_input_embeddings(input_ids)
        decoder_input_embeddings = self.decoder_input_embeddings(target_ids)

        encoder_output = self.encoder(encoder_input_embeddings, input_mask)
        decoder_output = self.decoder(decoder_input_embeddings, encoder_output, target_mask)

        transformer_output = self.transformer_output_layer(decoder_output)

        return transformer_output

    def forward_encoder(self, input_ids, input_padding_mask=None):
        # This function returns the encoder output without the decoder
        batch_size = input_ids.shape[0]

        input_mask = input_padding_mask if input_padding_mask is None else input_padding_mask.view(batch_size, 1, -1)

        encoder_input_embeddings = self.encoder_input_embeddings(input_ids)

        encoder_output = self.encoder(encoder_input_embeddings, input_mask)

        return encoder_output

    def forward_decoder_and_output_layer(self, encoder_output, decoder_input_ids, decoder_input_padding_mask=None):
        # This function returns the decoder output with the output layer
        decoder_input_embeddings = self.decoder_input_embeddings(decoder_input_ids)

        decoder_output = self.decoder(decoder_input_embeddings, encoder_output, decoder_input_padding_mask)

        transformer_output = self.transformer_output_layer(decoder_output)

        return transformer_output