import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import re
from sklearn.utils import shuffle
import models as lm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random


def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string


def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('<eos>')
    return tokenized


def encode(self, char_list):
    """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
    smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
    for i, char in enumerate(char_list):
        smiles_matrix[i] = self.vocab[char]
    return smiles_matrix


def decode(self, matrix):
    """Takes an array of indices and returns the corresponding SMILES"""
    chars = []
    for i in matrix:
        if i == self.vocab['EOS']: break
        chars.append(self.reversed_vocab[i])
    smiles = "".join(chars)
    smiles = smiles.replace("L", "Cl").replace("R", "Br")
    return smiles


def construct_vocabulary(smiles_list):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = []
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                if char not in add_chars:
                    add_chars.append(char)
            else:
                chars = [unit for unit in char]
                for n in chars:
                    if n not in add_chars:
                        add_chars.append(n)
                # [add_chars.add(unit) for unit in chars]
    print("Number of characters: {}".format(len(add_chars)))
    # with open('Data/Voc', 'w') as f:
    # for char in add_chars:
    # f.write(char + "\n")
    return add_chars


# read in the SMILES data
y_t = pd.read_csv('./Data/train_Y_microsource_2236.csv')
y_t = y_t.sort_values(by=['Molecule_ID'])
y_t = y_t.drop(['Molecule_ID'], axis=1)
Y_train = y_t.values.tolist()

x_test = pd.read_csv('./Data/val_X_smiles_clean_microsource_2236.csv')
x_test = x_test.sort_values(by=['Molecule_ID'])
x_test = x_test.drop(['Molecule_ID'], axis=1)

y_test_raw = pd.read_csv('./Data/val_Y_microsource_2236.csv')
y_test_raw = y_test_raw.sort_values(by=['Molecule_ID'])
y_test_temp = y_test_raw.drop(['Molecule_ID'], axis=1)
Y_test = y_test_temp.to_numpy(dtype=np.float32)

x_t = pd.read_csv('./Data/train_X_smiles_clean_microsource_2236.csv')
x_t = x_t.sort_values(by=['Molecule_ID'])
x_t = x_t.drop(['Molecule_ID'], axis=1)

# extract the smiles list
smiles_list = x_t['SMILES'].values.tolist()
smiles_list_test = x_test['SMILES'].values.tolist()

# replace halogens
test_re = [replace_halogen(x) for x in smiles_list]
test_re_test = [replace_halogen(x) for x in smiles_list_test]

# construct a vocabulary for the training set
# combine the dataframes to create the vocab
full_list_for_vocab = test_re + test_re_test
vocab = construct_vocabulary(full_list_for_vocab)


# Create an integer code for each vocab
vocabTable, index = {}, 2  # start indexing from 2
vocabTable['<eos>'] = 0  # add an end of sequence token
vocabTable['<sos>'] = 1  # start of sequence token; for the decoder

for token in vocab:
    if token not in vocabTable:
        vocabTable[token] = index
        index += 1
vocab_size_for_input = len(vocabTable) + 1

# save the full vocabulary
with open('Data/Vocab.txt', 'w') as filehandle:
    for listitem in vocabTable:
        filehandle.write('%s\n' % listitem)

# tokenize each SMILES string
token_X = [tokenize(x) for x in test_re]
token_X_test = [tokenize(x) for x in test_re_test]

# encode each SMILES string
encode_sequence_train = []
for x in token_X:
    encode_sequence_train.append([vocabTable[word] for word in x])

encode_sequence_test = []
for x in token_X_test:
    encode_sequence_test.append([vocabTable[word] for word in x])

# prep the val set
x_test, y_test = encode_sequence_test, Y_test

# pad the sequences to get them ready for input
pad_length = 190  # Maximum sequence input length; 190 is a good max for now.

####### TRAINING TIME #########
hidden_size = 512
output_size = 181
learning_rate = 0.0001
n_epochs = 116
batch_size = 64  # mini batch size? Try smaller next!
max_length = 190  # max length of the input sequence
clip = 5
teacher_forcing_ratio = 1
curriculum_rate = 2  # number of epochs before decreasing probability of teacher forcing
curriculum_drop = 0.1  # how much to decrease the probability of teacher forcing
min_prob_teacher = 0.0  # when to stop decreasing the teacher forcing
patience = 5  # patience for early stopping

# first specify the models
encoder = lm.EncoderLSTM_bb(vocab_size_for_input, hidden_size).to(device)
attn = lm.Attention_bb(hidden_size, "dot").to(device)
decoder = lm.LuongDecoder_bb(hidden_size, 1, attn).to(device)


# set the encoder and decoder optimizers to SGD
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-4)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-4)

# set the loss function
criterion = nn.L1Loss()

# setup for early stopping
prev_eval_loss = 999  # start with a ridiculous loss
num_not_imp = 0  # counter for patience

# for each epoch
for epoch in range(n_epochs):

    # make sure model is in training mode after eval
    encoder.train()
    attn.train()
    decoder.train()

    # keep running epoch loss
    epoch_loss = 0
    # shuffle the data
    x, y = shuffle(encode_sequence_train, Y_train)

    # for each batch
    for i in range(0, len(x), batch_size):

        # set batch loss
        batch_loss = 0

        # get the batch data
        batch_x, batch_y = x[i:i + batch_size], y[i:i + batch_size]

        # for each datapoint:
        for j in range(len(batch_x)):

            # zero the gradients and loss
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()


            # set the loss to 0
            loss = 0

            # get the input and target tensors
            input_tensor = torch.tensor(batch_x[j], dtype=torch.long, device=device).unsqueeze(0)
            target_tensor = torch.tensor(batch_y[j], dtype=torch.float, device=device).view(-1, 1)
            input_length = input_tensor.size(1)
            target_length = target_tensor.size(0)

            # zero the encoder weights
            encoder_hidden = encoder.init_hidden()

            # forward pass of the encoder: run through the whole sequence
            encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

            # now that we have the encoded hidden state, we'll pass an input of 0 for first context
            decoder_input = torch.tensor([[0.0]], dtype=torch.float, device=device)
            decoder_input = decoder_input.unsqueeze(1)
            decoder_hidden = encoder_hidden
            decoder_hidden = decoder.init_hidden()

            # use teacher forcing 50% of the time

            # use_teacher_forcing = False
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    #print(decoder_output.data.cpu().numpy())
                    #print(target_tensor[di].data.cpu().numpy())
                    #print(decoder_attention.data.cpu().numpy())
                    loss += criterion(decoder_output, target_tensor[di].unsqueeze(1))
                    decoder_input = (target_tensor[di].unsqueeze(1)).unsqueeze(1)  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    decoder_input = decoder_output.detach()  # detach from history as input
                    decoder_input = decoder_input.unsqueeze(0)
                    loss += criterion(decoder_output, target_tensor[di].unsqueeze(1))

            # get the batch loss
            batch_loss += loss/target_length
            # compute the gradient, update the parameters
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

            encoder_optimizer.step()
            decoder_optimizer.step()

            # print something?
            # if j % 10 == 0:
            # print("{0}/{1} complete. Loss: {2}".format(j, batch_size, loss))
        print("Batch loss: {0}".format(batch_loss / len(batch_x)))
        epoch_loss += batch_loss
    print('Epoch: {0}: epoch loss: {1} '.format(epoch, epoch_loss / len(x)))
    eval_loss, __, __, __, __ = lm.evaluate_b(x_test, y_test, encoder, decoder, max_length, criterion)
    print('Validation Loss: {0}'.format(eval_loss))

    # set up curriculum learning to go down every 5 epochs
    if epoch != 0 and epoch % curriculum_rate == 0 and teacher_forcing_ratio >= min_prob_teacher:
        teacher_forcing_ratio -= curriculum_drop
        print('Teacher forcing probability is now {0}'.format(teacher_forcing_ratio))

    # save every 10 epochs
    if epoch % 5 == 0:
        torch.save(encoder.state_dict(), "./Models/encoder_luong_bi_115_{0}_6lay_MS.pth".format(epoch))
        torch.save(decoder.state_dict(), "./Models/decoder_attn_luong_bi_115_{0}_6lay_MS.pth".format(epoch))
        torch.save(attn.state_dict(), "./Models/attention_attn_luong_bi_115_{0}_6laMS.pth".format(epoch))

    # early stopping: If evaluation loss hasn't improved in n epoch, stop early.
    if eval_loss > prev_eval_loss:
        num_not_imp += 1
    else:
        num_not_imp = 0
    if num_not_imp >= patience:
        break

    prev_eval_loss = eval_loss

# save the final model(s)
torch.save(encoder.state_dict(), "./Models/encoder_luong_bi_final_model_115_6lay.pth")
torch.save(decoder.state_dict(), "./Models/decoder_attn_luong_bi_final_model_115_6lay.pth")
torch.save(attn.state_dict(), "./Models/attention_attn_luong_bi_final_model_115_6lay.pth")

#torch.save(encoder.state_dict(), "./Models/dummy_11.pth")
#torch.save(decoder.state_dict(), "./Models/dummy_22.pth")
#torch.save(attn.state_dict(), "./Models/dummy_33.pth")

# evaluate?????, or save the final output on validation
eval_loss, predictions, actual, attn_data, target_le_dat = lm.evaluate_b(x_test, y_test, encoder, decoder, max_length,
                                                             criterion)

# save the predictions and attn.....I think
predictions.to_csv('./Outputs/luong_predictions_bi_final_model_115_6lay.csv')
actual.to_csv('./Outputs/luong_actual_bi_final_model_115_6lay.csv')
attn_data.to_csv('./Outputs/luong_attention_bi_final_model_115_6lay.csv')
target_le_dat.to_csv('./Outputs/luong_lengths_bi_final_model_115_6lay.csv')
