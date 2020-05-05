import torch
import torch.nn.functional as F
import utils
import clean
import os.path
from vocab import Vocab
from model import Vae
import torch.optim as optim

dim_embedding = 128
dim_hidden = 32
dim_latent = 32
dropout_prob = 0.5
batch_size = 16
keep_rate = 0.5 # word dropout probability

def loss_function(sents_hat, sents, mu, log_sd, padding_idx, ann_x):
    """
    Args
        sents_hat (tensor) : size (sents_len, batch_size, vocab_size)
        sents     (tensor) : size (sents_len, batch_size)
        mu        (tensor) : size (batch_size, dim_latent)
        log_sd    (tensor) : size (batch_size)
        ann_x     (tensor) : argument for KL annealing
        reduction:
    Returns
        loss (tensor) : tensor of size (batch_size)
    """
    batch_size = sents.size(1)
    # sents list of list of integers where each integer corresponds to a word in the vocab dictionary
    Cross_Entropy_loss = F.cross_entropy(sents_hat.permute(1, 2, 0), sents.permute(1, 0), ignore_index=padding_idx, reduction='sum')/batch_size
    KL_loss = utils.annealing(ann_x)*0.5*torch.sum(torch.exp(2*log_sd)+mu*mu-1-2*log_sd, dim=1)
    print('CE is {}, KL is {}, weight is {}'.format(Cross_Entropy_loss, KL_loss, utils.annealing(ann_x)))
    return Cross_Entropy_loss + KL_loss


def train(epoch, vae, optimizer, dataset, device, keep_rate):
    vae.train()
    total_loss = 0
    data_loader = utils.batch_loader(dataset=dataset, batch_size=batch_size, shuffle=True)
    dataset_size = len(dataset)
    #batch_data is List[List[str]]
    for batch_idx, batch_data in data_loader:
        # sents is tensor of size (sents_len, batch_size)
        sents = vae.vocab.vocab_entry.to_input_tensor(batch_data, device=device)
        batch_data_dropout = utils.word_dropout(batch_data, keep_rate, '<unk>')
        sents_dropout = vae.vocab.vocab_entry.to_input_tensor(batch_data_dropout, device=device)
        sents_len = torch.tensor([len(sent) for sent in batch_data], device=device)
        optimizer.zero_grad()
        sents_hat, mu, log_sd = vae.forward(sents_dropout, sents_len)
        x = epoch+min(dataset_size, (batch_idx+1)*batch_size)/dataset_size
        loss = loss_function(sents_hat, sents, mu, log_sd, vae.padding_idx, x)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, min(dataset_size, (batch_idx+1)*batch_size), dataset_size, 100.*min(1, (batch_idx+1)*batch_size/dataset_size), loss.item()))

    #print('Average loss: {:.06f}'.format(total_loss/len(dataset)))


def main(task):
    if task is 'clean_dataset':
        dataset = clean.clean_corpus('./datasets/Video_Games_5.json.gz')
        sents_len = utils.get_sents_len(dataset)
        utils.save_dataset(dataset, './raw_dataset.json')
        utils.save_dataset(sents_len, './raw_dataset.json')
    elif task is 'build_vocab':
        dataset = utils.load_dataset('./raw_dataset.json')
        # build vocab, remove sentences that are too long or contains words that are not in vocab
        truncated_dataset = [sent for sent in dataset if len(sent) < 300]
        vocab = Vocab.build(truncated_dataset, 50000, 10)
        vocab.save('vocab.json')
        clean_dataset = utils.remove_unk(truncated_dataset, vocab)
        utils.save_dataset(clean_dataset, './clean_dataset.json')
        print('total number of reviews is {}'.format(len(clean_dataset)))
    elif task is 'train_model':
        # Load vocab
        vocab = Vocab.load('./vocab.json')
        dataset = utils.load_dataset('./clean_dataset.json')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.isfile('./vae.pt'):
            vae = torch.load('./vae.pt')
            start_epoch = utils.load_dataset('./epoch.json')[0]
        else:
            vae = Vae(dim_embedding=dim_embedding, dim_hidden=dim_hidden, dim_latent=dim_latent, vocab=vocab, dropout_prob=dropout_prob, device=device)
            start_epoch = 1
        vae.to(device)
        optimizer = optim.Adam(vae.parameters())
        for epoch in range(start_epoch, 10):
            train(epoch, vae, optimizer, dataset, device, keep_rate)
            torch.save(vae, './vae.pt')
            utils.save_dataset([epoch], './epoch.json')

    elif task is 'continue_train_model':
        vae = torch.load('./vae.pt')
        dataset = utils.load_dataset('./clean_dataset.json')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae.to(device)
        optimizer = optim.Adam(vae.parameters())
        for epoch in range(9,10):
            train(epoch, vae, optimizer, dataset, device, keep_rate)
            torch.save(vae, './vae.pt')
            utils.save_dataset([epoch], './epoch.json')

    elif task is 'load_model':
        vae = torch.load('./vae.pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae.to(device)
        vae.eval()
        # sents is List[List[List[str]]]
        sents = vae.decoder(batch_size=100, beam_width=20, max_len=20)
        # batch is List[List[str]]
        utils.save_dataset(sents, './generated_sents.json')
        """
        for batch in sents:
            # beams is List[str]
            for beam in batch:
                print(beam, '\n')
        """
    elif task is 'test':
        utils.get_review_scores('./datasets/Video_Games.json.gz')

    elif task is 'detect':
        vae = torch.load('./vae.pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae.to(device)
        vae.eval()
        dataset = utils.load_dataset('./clean_dataset.json')
        dataset_size = len(dataset)
        data_loader = utils.batch_loader(dataset=dataset, batch_size=batch_size, shuffle=True)
        sents_loss = []
        for batch_idx, batch_data in data_loader:
            # sents is tensor of size (sents_len, batch_size)
            sents = vae.vocab.vocab_entry.to_input_tensor(batch_data, device=device)
            sents_len = torch.tensor([len(sent) for sent in batch_data], device=device)
            sents_hat, mu, log_sd = vae.forward(sents, sents_len)
            Cross_Entropy_loss = F.cross_entropy(sents_hat.permute(1, 2, 0), sents.permute(1, 0), ignore_index=0, reduction='none').sum(dim=1)
            KL_loss = 10*0.5*torch.sum(torch.exp(2*log_sd)+mu*mu-1-2*log_sd, dim=1)
            loss = (Cross_Entropy_loss + KL_loss).tolist()
            print('[{}/{} ({:.0f}%)]\n'.format(min(dataset_size, (batch_idx+1)*batch_size), dataset_size, 100.*min(1, (batch_idx+1)*batch_size/dataset_size)))
            for sent, loss in zip(batch_data, loss):
                sents_loss.append((sent, loss/len(sent)))
        sents_anomaly = sorted(sents_loss, key=lambda t: t[1], reverse=True)
        utils.save_dataset(sents_anomaly, './anomaly.json')
    else:
        print('Invalid task')


#main('get_lengths')
#main('build_vocab')
#main('build_dataset')
#main('train_model')
#main('continue_train_model')
#main('detect')
#main('normalise_anomaly')
#main('detect')
main('test')
