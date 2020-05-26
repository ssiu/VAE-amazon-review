import torch
import torch.nn.functional as F
import utils
import clean
import os.path
from vocab import Vocab
from model import Vae
import torch.optim as optim


def loss_function(sents_hat, sents, mu, log_sd, padding_idx, weight):
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
    KL_loss = weight*0.5*torch.sum(torch.exp(2*log_sd)+mu*mu-1-2*log_sd, dim=1).mean()
    #print('CE is {}, KL is {}, weight is {}'.format(Cross_Entropy_loss, KL_loss, weight))
    return Cross_Entropy_loss + KL_loss


def train(epoch, vae, optimizer, dataset, batch_size, device, keep_rate, a, b):
    KLweight = utils.KLweight(a, b)
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
        loss = loss_function(sents_hat, sents, mu, log_sd, vae.padding_idx, KLweight.weight(x))
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, min(dataset_size, (batch_idx+1)*batch_size), dataset_size, 100.*min(1, (batch_idx+1)*batch_size/dataset_size), loss.item()))

    #print('Average loss: {:.06f}'.format(total_loss/len(dataset)))


def detect(vae, dataset, sample_size, device):
    """
    Compute the anomaly score for a single review, 
    see algorithm 4 in https://pdfs.semanticscholar.org/0611/46b1d7938d7a8dae70e3531a00fceb3c78e8.pdf
    Args
        vae: trained vae model
        data: tensor of size (1, )
        sample_size: number of z~N(\mu,\sigma) to sample
        device: 
    Returns
        Anomaly score: List[List[str]] where reviews are sorted by reconstruction loss

    """
    data_loader = utils.batch_loader(dataset=dataset, batch_size=1, shuffle=True)
    sents_recon_loss = []
    for batch_idx, batch_data in data_loader:
        # sents is tensor of size (sents_len, batch_size)
        sent = vae.vocab.vocab_entry.to_input_tensor(batch_data, device=device)
        wun = torch.ones([1, sample_size], device=device)
        sents = torch.mm(sent.float(), wun).long()
        sents_len = torch.tensor([len(sent)]*sample_size, device=device)
        sents_hat, mu, log_sd = vae.forward(sents, sents_len)
        loss = loss_function(sents_hat, sents, mu, log_sd, vae.padding_idx, 1)
        sents_recon_loss.append((batch_data, loss.item()/len(sent)))
        if batch_idx % 100 == 0:
            print('Progress: [{}/{} ({:.0f}%)]'.format(batch_idx, len(dataset), 100*batch_idx/len(dataset)))

    sents_recon_loss.sort(key=lambda x:x[1], reverse=True)
    return sents_recon_loss


def main(task, config):
    if task is 'split_dataset':
        """
        Split dataset based on review score 
        """
        review_scores, review_data = utils.get_review_scores_and_data(config['split_dataset']['load_path'])
        utils.save_dataset(review_scores, config['split_dataset']['save_path_scores'])
        for i in range(5):
            score = str(i+1)
            save_path = config['split_dataset']['save_path_data'] + score + '.json'
            utils.save_dataset(review_data[i], save_path)
    elif task is 'clean_dataset':
        dataset = clean.clean_corpus(config['clean_dataset']['load_path'])
        sents_len = utils.get_sents_len(dataset)
        utils.save_dataset(dataset, config['clean_dataset']['save_path_data'])
        utils.save_dataset(sents_len, config['clean_dataset']['save_path_seq_len'])
    elif task is 'build_vocab':
        dataset = utils.load_dataset(config['build_vocab']['load_path'])
        # build vocab, remove sentences that are too long or contains words that are not in vocab
        truncated_dataset = [sent for sent in dataset if len(sent) < config['build_vocab']['max_seq_len']]
        vocab = Vocab.build(truncated_dataset, config['build_vocab']['max_vocab'], config['build_vocab']['min_word_freq'], config['build_vocab']['save_path_vocab_dist'])
        vocab.save(config['build_vocab']['save_path_vocab'])
        train_dataset = utils.remove_unk(truncated_dataset, vocab)
        utils.save_dataset(train_dataset, config['build_vocab']['save_path_data'])
        print('total number of reviews is {}'.format(len(train_dataset)))
    elif task is 'train_model':
        load_path_vocab = config['train_model']['load_path_vocab']
        load_path_data = config['train_model']['load_path_data']
        save_path = config['train_model']['save_path']
        start_epoch = config['train_model']['start_epoch']
        max_epoch = config['train_model']['max_epoch']
        dim_embedding = config['train_model']['model']['dim_embedding']
        dim_hidden = config['train_model']['model']['dim_hidden']
        dim_latent = config['train_model']['model']['dim_latent']
        dropout_prob = config['train_model']['model']['dropout_prob']
        batch_size = config['train_model']['model']['batch_size']
        keep_rate = config['train_model']['model']['keep_rate']
        a = config['train_model']['model']['a']
        b = config['train_model']['model']['b']

        vocab = Vocab.load(load_path_vocab)
        dataset = utils.load_dataset(load_path_data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.isfile(save_path):
            vae = torch.load(save_path)
        else:
            vae = Vae(dim_embedding=dim_embedding, dim_hidden=dim_hidden, dim_latent=dim_latent, vocab=vocab, dropout_prob=dropout_prob, device=device)
        vae.to(device)
        optimizer = optim.Adam(vae.parameters())
        for epoch in range(start_epoch, max_epoch):
            train(epoch, vae, optimizer, dataset, batch_size, device, keep_rate, a, b)
            torch.save(vae, save_path)
            utils.set_epoch(epoch)
    elif task is 'detect_anomaly':
        load_path_data = config['detect_anomaly']['load_path_data']
        load_path_model = config['detect_anomaly']['load_path_model']
        save_path = config['detect_anomaly']['save_path']
        sample_size = config['detect_anomaly']['sample_size']
        vae = torch.load(load_path_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae.to(device)
        vae.eval()
        dataset = utils.load_dataset(load_path_data)
        sents_recon_loss = detect(vae=vae, dataset=dataset, sample_size=sample_size, device=device)
        utils.save_dataset(sents_recon_loss, save_path)
    elif task is 'test':
        test = []
        dataset = utils.load_dataset('./files/Fashion/anomaly.json')
        for sent in dataset:
            if len(sent[0][0])>5 and len(sent[0][0])<50:
                test.append(sent)
        print(len(test))
        utils.save_dataset(test, './files/Fashion/test.json')
    else:
        print('Invalid task')


config = utils.read_config('./config.yml')

#main('split_dataset', config)
#main('clean_dataset', config)
#main('build_vocab', config)
#main('train_model', config)
#main('detect_anomaly', config)
main('test', config)
