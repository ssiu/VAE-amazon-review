"""
VAE model
"""
import torch
import torch.nn as nn
from embeddings import ModelEmbeddings
from torch.nn.utils.rnn import pack_padded_sequence


class Vae(torch.nn.Module):
    def __init__(self, dim_embedding, dim_hidden, dim_latent, vocab, dropout_prob, device: torch.device):
        super(Vae, self).__init__()
        """
        @param dim_embedding: dimension of word embeddings
        @param dim_hidden: dimension of hidden space of encoder LSTM
        @param dim_latent: dimension of latent space, also the hidden space of decoder LSTMCell
        @param sents_len: length of padded sentences        
        @param dropout_prob: dropout probability
        """
        self.padding_idx = vocab.vocab_entry['<pad>']
        self.sos_idx = vocab.vocab_entry['<s>']
        self.eos_idx = vocab.vocab_entry['</s>']
        self.model_embeddings = ModelEmbeddings(embed_size=dim_embedding, vocab=vocab, padding_idx=self.padding_idx)
        self.dim_embedding = dim_embedding
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.vocab = vocab
        self.device = device
        self.dropout_prob = dropout_prob
        # encoder layers
        self.enc_LSTM = nn.LSTM(input_size=dim_embedding, hidden_size=dim_hidden)
        self.enc_linear_mu = nn.Linear(dim_hidden, dim_latent)
        self.enc_linear_log_sd = nn.Linear(dim_hidden, dim_latent)
        # decoder layers
        self.dec_LSTM = nn.LSTMCell(input_size=dim_embedding, hidden_size=dim_latent)
        self.dec_output = nn.Linear(dim_latent, len(vocab.vocab_entry))
        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        # nn.init.xavier_uniform_(self.enc_layer1.weight)

    def encoder(self, sents_embed, sents_len):
        """
        @param sents_embed : tensor of size (sents_len, batch_size, dim_embedding), ordered by sents_len in decreasing order
        @param sents_len: tensor of size (batch_size), list of sentence lengths
        @returns mu: tensor of size (batch_size, dim_latent)
        @returns log_sd: tensor of size (batch_size, dim_latent)
        """
        # sents_t is tensor of size (sents_len, batch_size, dim_embedding)
        pack_sents_embed = pack_padded_sequence(input=sents_embed, lengths=sents_len, enforce_sorted=False)
        # h,c are tensors of size (1, batch_size, dim_hidden)
        output, (h, c) = self.enc_LSTM(pack_sents_embed)
        # h,c is a tensor of size (batch_size, dim_hidden)
        h = torch.squeeze(h, dim=0)
        mu = self.enc_linear_mu(h)
        log_sd = self.enc_linear_log_sd(h)
        return mu, log_sd

    def sampling(self, mu, log_sd) -> torch.tensor:
        """
        @param mu: tensor of size (batch_size, dim_latent)
        @param log_sd: tensor of size (batch_size,  dim_latent)
        @returns: tensor of size (batch_size, dim_latent)
        """
        (batch_size, dim_latent) = mu.size()
        epsilon = torch.randn(batch_size, dim_latent, device=self.device)
        z = mu + epsilon * torch.exp(log_sd)
        return z

    # This computes the LSTMCell at every timestep
    def step(self, word, h_prev, c_prev):
        """
        @param word: tensor of size (batch_size, dim_embedding)
        @param h_prev: (batch_size, dim_latent)
        @param c_prev: (batch_size, dim_latent)
        @return words_hat: tensor of size (batch, vocab_size)
        """
        (h_next, c_next) = self.dec_LSTM(word, (h_prev, c_prev))
        words_hat = self.dec_output(h_next)
        return words_hat, h_next, c_next

    # This takes in a sents and compute cross entropy
    def forward(self, sents, sents_len) :
        """
        @param sents: tensor of size (sents_len, batch_size)
        @return sents_hat: tensor of size (sents_len, batch_size, vocab_size)
        """
        # sents_embed is tensor of size (sents_len, batch_size, dim_embedding)
        sents_embed = self.model_embeddings.embeddings(sents)
        mu, log_sd = self.encoder(sents_embed, sents_len)
        z = self.sampling(mu, log_sd)
        (batch_size, dim_latent) = z.size()
        h = z
        c = torch.zeros((batch_size, dim_latent), device=self.device)
        # sos is tensor of size (batch_size, dim_embeddings), containing embedding of '<s>', the first word for the decoder
        sos = torch.ones(batch_size, dtype=torch.long, device=self.device) * self.sos_idx
        sos_embed = self.model_embeddings.embeddings(sos)
        # append the first words_hat
        words_hat, h, c = self.step(sos_embed, h, c)
        sents_hat = words_hat.unsqueeze(dim=0)
        for words in sents_embed[:-1]:
            words_hat, h, c = self.step(words, h, c)
            sents_hat = torch.cat((sents_hat, words_hat.unsqueeze(dim=0)), dim=0)
        return sents_hat, mu, log_sd

    def decoder(self, batch_size, beam_width, max_len):
        """
        @param z: tensor of size (batch_size, dim_latent)
        @param beam_width: beam width for beam search
        @param max_len: integer for maximum length of sentences
        @returns: tensor of size (batch_size, beam_width, sents_len)
        """
        z = torch.randn((batch_size, self.dim_latent), device=self.device )
        word_ids = self.beam_search(z, beam_width, max_len)
        sents = []
        # batch is List[List[str]]
        for batch in word_ids:
            batch_w = []
            # beam is List[str]
            for beam in batch:
                beam_w = []
                for w_id in beam:
                    if w_id is self.vocab.vocab_entry['</s>']:
                        break
                    beam_w.append(self.vocab.vocab_entry.id2word[w_id])
                batch_w.append(beam_w)
            sents.append(batch_w)
        return sents

    def beam_search(self, z_batch, beam_width, max_timestep):
        """
        @param z: tensor of size (batch_size, dim_latent)
        @param beam_width: search beam width
        @returns completed_hypothesis: List of size (batch_size, beam_width, max_timestep)
        """
        sm = nn.LogSoftmax()
        (batch_size, dim_latent) = z_batch.size()
        h_batch = z_batch
        c_batch = torch.zeros((batch_size, dim_latent), device=self.device)
        # completed_hypothesis is of size (batch_size, beam_width, max_timestep)
        completed_hypothesis = []
        for h, c in zip(h_batch, c_batch):
            # h,c are tensors of size (1,dim_latent)
            h = h.unsqueeze(dim=0)
            c = c.unsqueeze(dim=0)
            hypothesis = [([self.vocab.vocab_entry['<s>']], 0, h, c)]
            t = 0
            while t < max_timestep:
                potential_hypothesis = []
                for potential_sent, potential_loss, h_prev, c_prev in hypothesis:
                    prev_word = potential_sent[-1]
                    # prev_word is tensor of size (1, dim_embedding)
                    prev_word = self.model_embeddings.embeddings(torch.tensor([prev_word], device=self.device))
                    # next_word_values is a tensor of size (1, vocab_size)
                    # next_h, next_c are tensors of size (1, dim_latent)
                    next_word_values, next_h, next_c = self.step(prev_word, h_prev, c_prev)
                    # next_word_distribution is tensor of size (vocab_size)
                    next_word_distribution = sm(next_word_values.squeeze(dim=0))
                    # values, indices are tensors of size (beam_width)
                    values, indices = next_word_distribution.topk(k=beam_width)
                    for v, i in zip(values.tolist(), indices.tolist()):
                        potential_hypothesis.append((potential_sent+[i], potential_loss + v, next_h, next_c))
                # get the top beam_width
                hypothesis = sorted(potential_hypothesis, key=lambda tup: tup[1], reverse=True)[10:beam_width+10]
                # remove hypothesis that has eos character
                remove_hypothesis = []
                for hyp_check, loss_check, h_check, c_check in hypothesis:
                    if hyp_check[-1] is self.vocab.vocab_entry['</s>']:
                        completed_hypothesis.append([hyp_check])
                        remove_hypothesis.append((hyp_check, loss_check, h_check, c_check))
                for hyp_rm, loss_rm, h_rm, c_rm in remove_hypothesis:
                    hypothesis.remove((hyp_rm, loss_rm, h_rm, c_rm))
                t += 1

            completed_hypothesis.append([hyp for hyp, loss, h, c in hypothesis])
        return completed_hypothesis

