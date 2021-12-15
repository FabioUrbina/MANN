import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
import vanilla_encoder_decoder as vn
import class_model as cm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=3, drop_prob=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device))


class LuongDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=3, drop_prob=0.1):
        super(LuongDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # The Attention Mechanism is defined in a separate class
        self.attention = attention

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lin_cast = nn.Linear(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers)
        self.classifier = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embed input words
        # embedded = self.embedding(inputs).view(1, 1, -1)
        embedded = self.lin_cast(inputs)
        embedded = self.dropout(embedded)

        # Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(lstm_out, encoder_outputs)
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        # Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector), -1)
        # Pass concatenated vector through Linear layer acting as a Classifier
        # output = F.log_softmax(self.classifier(output[0]), dim=1)
        output = self.classifier(output[0])

        return output, hidden, attn_weights

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device))


class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.empty(1, hidden_size, device=device))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)


class EncoderLSTM_b(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, drop_prob=0.2):
        super(EncoderLSTM_b, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=True, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device))


class LuongDecoder_b(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=4, drop_prob=0.2):
        super(LuongDecoder_b, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # The Attention Mechanism is defined in a separate class
        self.attention = attention

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lin_cast = nn.Linear(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size * 2, self.n_layers)
        self.classifier = nn.Linear(self.hidden_size * 4, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embed input words
        # embedded = self.embedding(inputs).view(1, 1, -1)
        embedded = self.lin_cast(inputs)
        embedded = self.dropout(embedded)

        # Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(lstm_out, encoder_outputs)
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        # Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector), -1)
        # Pass concatenated vector through Linear layer acting as a Classifier
        # output = F.log_softmax(self.classifier(output[0]), dim=1)
        output = self.classifier(output[0])

        return output, hidden, attn_weights

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device))


class Attention_b(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention_b, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.empty(1, hidden_size, device=device))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)

### more laters!!!
class EncoderLSTM_bb(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=3, drop_prob=0.2):
        super(EncoderLSTM_bb, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=True, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device))


class LuongDecoder_bb(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=6, drop_prob=0.2):
        super(LuongDecoder_bb, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # The Attention Mechanism is defined in a separate class
        self.attention = attention

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lin_cast = nn.Linear(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size * 2, self.n_layers)
        self.classifier = nn.Linear(self.hidden_size * 4, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embed input words
        # embedded = self.embedding(inputs).view(1, 1, -1)
        embedded = self.lin_cast(inputs)
        embedded = self.dropout(embedded)

        # Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(lstm_out, encoder_outputs)
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        # Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector), -1)
        # Pass concatenated vector through Linear layer acting as a Classifier
        # output = F.log_softmax(self.classifier(output[0]), dim=1)
        output = self.classifier(output[0])

        return output, hidden, attn_weights

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device))


class Attention_bb(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention_bb, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.empty(1, hidden_size, device=device))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)


##### Here, we have both encoder and decoder as BIDIRECTIONAL!!!
class EncoderLSTM_bibi(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=4, drop_prob=0.2):
        super(EncoderLSTM_bibi, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional= True, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device))


class LuongDecoder_bibi(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=4, drop_prob=0.2):
        super(LuongDecoder_bibi, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # The Attention Mechanism is defined in a separate class
        self.attention = attention

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lin_cast = nn.Linear(self.output_size, self.hidden_size*2)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size, self.n_layers, bidirectional=True)
        self.classifier = nn.Linear(self.hidden_size * 4, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embed input words
        # embedded = self.embedding(inputs).view(1, 1, -1)
        embedded = self.lin_cast(inputs)
        embedded = self.dropout(embedded)

        # Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(lstm_out, encoder_outputs)
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        # Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector), -1)
        # Pass concatenated vector through Linear layer acting as a Classifier
        # output = F.log_softmax(self.classifier(output[0]), dim=1)
        output = self.classifier(output[0])

        return output, hidden, attn_weights

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers*2, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers*2, batch_size, self.hidden_size, device=device))


class Attention_bibi(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention_bibi, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.empty(1, hidden_size, device=device))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)

### 512, 3 layers, but 64 is batch size
class EncoderLSTM_b64(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, drop_prob=0.2):
        super(EncoderLSTM_b64, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=True, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, hidden):
        # Embed input words
        embedded = self.embedding(inputs)
        # Pass the embedded word vectors into LSTM and return all outputs
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device),
                torch.zeros(self.n_layers * 2, batch_size, self.hidden_size, device=device))


class LuongDecoder_b64(nn.Module):
    def __init__(self, hidden_size, output_size, attention, n_layers=4, drop_prob=0.2):
        super(LuongDecoder_b64, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # The Attention Mechanism is defined in a separate class
        self.attention = attention

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.lin_cast = nn.Linear(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.drop_prob)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size * 2, self.n_layers)
        self.classifier = nn.Linear(self.hidden_size * 4, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        # Embed input words
        # embedded = self.embedding(inputs).view(1, 1, -1)
        embedded = self.lin_cast(inputs)
        embedded = self.dropout(embedded)

        # Passing previous output word (embedded) and hidden state into LSTM cell
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(lstm_out, encoder_outputs)
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores.view(1, -1), dim=1)

        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs)

        # Concatenating output from LSTM with context vector
        output = torch.cat((lstm_out, context_vector), -1)
        # Pass concatenated vector through Linear layer acting as a Classifier
        # output = F.log_softmax(self.classifier(output[0]), dim=1)
        output = self.classifier(output[0])

        return output, hidden, attn_weights

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size * 2, device=device))


class Attention_b64(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention_b64, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.empty(1, hidden_size, device=device))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1, -1, 1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1, -1, 1)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)


def evaluate(x, y, encoder, decoder, max_length, criterion):
    # create lists to hold the predicted and actual values
    pred_li = []
    actual_li = []
    attn_li = []
    batch_loss = 0
    tar_len = []
    batch_loss = 0

    for j in range(len(x)):

        # set the loss to 0
        loss = 0

        # get the input and target tensors
        input_tensor = torch.tensor(x[j], dtype=torch.long, device=device).unsqueeze(0)
        target_tensor = torch.tensor(y[j], dtype=torch.float, device=device).view(-1, 1)
        input_length = input_tensor.size(1)
        target_length = target_tensor.size(0)

        # zero the encoder weights
        encoder_hidden = encoder.init_hidden()

        # forward pass of the encoder: run through the whole sequence
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        # now that we have the encoded hidden state, we'll pass an input of 0 for first context
        decoder_input = torch.tensor([[0.0]], dtype=torch.float, device=device)
        # print(decoder_input)
        decoder_input = decoder_input.unsqueeze(1)
        decoder_hidden = encoder_hidden

        pred_temp = []
        actual_temp = []
        attn_temp = []
        tar_len.append(input_length)
        # Use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decoder_output.detach()  # detach from history as input
            decoder_input = decoder_input.unsqueeze(0)
            pred_temp.append(decoder_input.data.cpu().numpy()[0])
            actual_temp.append(target_tensor[di].data.cpu().numpy()[0])
            attn_li.append(decoder_attention.data.cpu().numpy()[0])
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(1))

        # get the batch loss
        batch_loss += loss / target_length

        # append to separate dataframes
        pred_li.append(pred_temp)
        actual_li.append(actual_temp)
        # attn_li.append(attn_li)
    eval_loss = batch_loss / len(x)
    pred_df = pd.DataFrame(pred_li)
    act_df = pd.DataFrame(actual_li)
    attn_df = pd.DataFrame(attn_li)
    tar_df = pd.DataFrame(tar_len)
    return eval_loss, pred_df, act_df, attn_df, tar_df


def evaluate_b(x, y, encoder, decoder, max_length, criterion):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # create lists to hold the predicted and actual values
        pred_li = []
        actual_li = []
        attn_li = []
        batch_loss = 0
        tar_len = []
        batch_loss = 0

        for j in range(len(x)):

            # set the loss to 0
            loss = 0

            # get the input and target tensors

            input_tensor = torch.tensor(x[j], dtype=torch.long, device=device).unsqueeze(0)
            target_tensor = torch.tensor(y[j], dtype=torch.float, device=device).view(-1, 1)
            input_length = input_tensor.size(1)
            target_length = target_tensor.size(0)

            # zero the encoder weights
            encoder_hidden = encoder.init_hidden()

            # forward pass of the encoder: run through the whole sequence
            encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

            # now that we have the encoded hidden state, we'll pass an input of 0 for first context
            decoder_input = torch.tensor([[0.0]], dtype=torch.float, device=device)
            # print(decoder_input)
            decoder_input = decoder_input.unsqueeze(1)
            decoder_hidden = encoder_hidden
            decoder_hidden = decoder.init_hidden()
            pred_temp = []
            actual_temp = []
            attn_temp = []
            tar_len.append(input_length)
            # Use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output.detach()  # detach from history as input
                decoder_input = decoder_input.unsqueeze(0)
                pred_temp.append(decoder_input.data.cpu().numpy()[0])
                actual_temp.append(target_tensor[di].data.cpu().numpy()[0])
                attn_li.append(decoder_attention.data.cpu().numpy()[0])
                loss += criterion(decoder_output, target_tensor[di].unsqueeze(1))

            # get the batch loss
            batch_loss += loss / target_length

            # append to separate dataframes
            pred_li.append(pred_temp)
            actual_li.append(actual_temp)
            # attn_li.append(attn_li)
        eval_loss = batch_loss / len(x)
        pred_df = pd.DataFrame(pred_li)
        act_df = pd.DataFrame(actual_li)
        attn_df = pd.DataFrame(attn_li)
        tar_df = pd.DataFrame(tar_len)

    return eval_loss, pred_df, act_df, attn_df, tar_df


def evaluate_only_b(x, y, encoder, decoder):

    # turn on eval mode
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # create lists to hold the predicted and actual values
        pred_li = []
        actual_li = []
        attn_li = []
        #batch_loss = 0
        tar_len = []
        #batch_loss = 0

        for j in range(len(x)):

            # print out the current evaluated target
            #loss = 0
            print('Evaluating {0} out of {1}.'.format(j, len(x)))

            # get the input and target tensors

            input_tensor = torch.tensor(x[j], dtype=torch.long, device=device).unsqueeze(0)
            target_tensor = torch.tensor(y[j], dtype=torch.float, device=device).view(-1, 1)
            input_length = input_tensor.size(1)
            target_length = target_tensor.size(0)

            # zero the encoder weights
            encoder_hidden = encoder.init_hidden()

            # forward pass of the encoder: run through the whole sequence
            encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

            # now that we have the encoded hidden state, we'll pass an input of 0 for first context
            decoder_input = torch.tensor([[0.0]], dtype=torch.float, device=device)
            # print(decoder_input)
            decoder_input = decoder_input.unsqueeze(1)
            #decoder_hidden = encoder_hidden
            decoder_hidden = decoder.init_hidden()
            pred_temp = []
            actual_temp = []
            # attn_temp = []
            tar_len.append(input_length)
            # Use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output.detach()  # detach from history as input
                decoder_input = decoder_input.unsqueeze(0)
                pred_temp.append(decoder_input.data.cpu().numpy()[0])
                actual_temp.append(target_tensor[di].data.cpu().numpy()[0])
                attn_li.append(decoder_attention.data.cpu().numpy()[0])
                #loss += criterion(decoder_output, target_tensor[di].unsqueeze(1))

            # get the batch loss
            #batch_loss += loss / target_length

            # append to separate dataframes
            pred_li.append(pred_temp)
            actual_li.append(actual_temp)
            # attn_li.append(attn_li)
        #eval_loss = batch_loss / len(x)
        pred_df = pd.DataFrame(pred_li)
        act_df = pd.DataFrame(actual_li)
        attn_df = pd.DataFrame(attn_li)
        tar_df = pd.DataFrame(tar_len)
    return pred_df, act_df, attn_df, tar_df


def predict_only_b(x, encoder, decoder):

    # turn on eval mode
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # create lists to hold the predicted and actual values
        pred_li = []
        attn_li = []
        #batch_loss = 0
        tar_len = []
        #batch_loss = 0

        for j in range(len(x)):

            # print out the current evaluated target
            #loss = 0
            print('Evaluating {0} out of {1}.'.format(j, len(x)))

            # get the input and target tensors

            input_tensor = torch.tensor(x[j], dtype=torch.long, device=device).unsqueeze(0)
            input_length = input_tensor.size(1)
            target_length = 181
            # zero the encoder weights
            encoder_hidden = encoder.init_hidden()

            # forward pass of the encoder: run through the whole sequence
            encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

            # now that we have the encoded hidden state, we'll pass an input of 0 for first context
            decoder_input = torch.tensor([[0.0]], dtype=torch.float, device=device)
            # print(decoder_input)
            decoder_input = decoder_input.unsqueeze(1)
            #decoder_hidden = encoder_hidden
            decoder_hidden = decoder.init_hidden()
            pred_temp = []
            # attn_temp = []
            tar_len.append(input_length)
            # Use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = decoder_output.detach()  # detach from history as input
                decoder_input = decoder_input.unsqueeze(0)
                pred_temp.append(decoder_input.data.cpu().numpy()[0])
                attn_li.append(decoder_attention.data.cpu().numpy()[0])

            # get the batch loss
            #batch_loss += loss / target_length

            # append to separate dataframes
            pred_li.append(pred_temp)
            # attn_li.append(attn_li)
        #eval_loss = batch_loss / len(x)
        pred_df = pd.DataFrame(pred_li)
        attn_df = pd.DataFrame(attn_li)
        tar_df = pd.DataFrame(tar_len)
    return pred_df, attn_df, tar_df
