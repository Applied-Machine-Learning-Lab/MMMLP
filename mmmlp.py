import torch
from torch import nn
from functools import partial
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, FeatureSeqEmbLayer, VanillaAttention
from recbole.model.loss import BPRLoss
from einops.layers.torch import Rearrange, Reduce
import pickle
import numpy as np
from time import time 

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

class MLP1(nn.Module):

    def __init__(self):
        super(MLP1, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(300, 150),
            nn.LeakyReLU(inplace=True),
            nn.Linear(150, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 64),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x
        

class MLP2(nn.Module):

    def __init__(self):
        super(MLP2, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 150),
            nn.LeakyReLU(inplace=True),
            nn.Linear(150, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 128),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x


class MLP3(nn.Module):

    def __init__(self):
        super(MLP3, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 150),
            nn.LeakyReLU(inplace=True),
            nn.Linear(150, 100),
            nn.LeakyReLU(inplace=True),
            nn.Linear(100, 1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x

class MMMLP(SequentialRecommender):
    r"""
    MMMLP is similar with the mlpmixer implemented in RecBole, which uses three different mlpmxier to
    encode items and features respectively and concatenates the three subparts' outputs as the final output.

    """

    def __init__(self, config, dataset):
        super(MMMLP, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
    
        self.device = config['device']
        expansion_factor = 4
        chan_first = partial(nn.Conv1d, kernel_size = 1)
        chan_last = nn.Linear
        self.concat_layer_f = nn.Linear(self.hidden_size * 3, self.hidden_size)

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

       # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)


        self.tokenMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.t_tokenMixer = PreNormResidual(self.hidden_size, FeedForward(self.max_seq_length, expansion_factor, self.hidden_dropout_prob, chan_first))
        self.t_channelMixer = PreNormResidual(self.hidden_size, FeedForward(self.hidden_size, expansion_factor, self.hidden_dropout_prob))
        self.t_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.LayerNormFeature = nn.LayerNorm(3*self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.batch_size = config['train_batch_size']
        self.batch_num = max(self.batch_size // 4, 1)
        self.dataset = dataset
        self.model =  self.vMixer(image_size = (300,300),channels = 3,patch_size = 50,dim = 128,depth = 4)
        device = torch.device("cuda:0")
        self.model = self.model.to(device)
        #Text dict
        F=open('D:/BaiduNetdiskDownload/New_dict/1m_text.pkl','rb')
        self.content=pickle.load(F)
        
        #Image dict
        F1=open('D:/BaiduNetdiskDownload/New_dict/1m_image.pkl','rb')
        self.content1=pickle.load(F1)

        
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def vMixer(self,image_size, channels, patch_size, dim, depth, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
        image_h, image_w = pair(image_size)
        assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_h // patch_size) * (image_w // patch_size)
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        return nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear((patch_size ** 2) * channels, dim),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            )for _ in range(depth)],
            nn.LayerNorm(dim),
            Reduce('b n c -> b c', 'mean')
        )


    def vMLPMixer(self, item_seq):
        iid_series = self.dataset.id2token(self.dataset.iid_field, item_seq.cpu())
        rows, cols = iid_series.shape
        a = []
        for i in np.nditer(iid_series):
            if i != np.array('[PAD]'):
                externalid = str(i)
                img_tensor = self.content1[externalid] # numpy数组格式为（H,W,C）,tensor数据格式是torch(C,H,W)
                
                
            else:
                img_tensor = torch.zeros([1,128]).cuda()

            
            a.append(img_tensor)

        imixer_output = torch.cat(a)
        imixer_output = torch.reshape(imixer_output,(rows,50,128))
        imixer_output = self.LayerNorm(imixer_output)
        for _ in range(4):
            ioutput = self.tokenMixer(imixer_output)
            ioutput = self.channelMixer(ioutput)
        
        return ioutput

    def tMLPMixer(self,item_seq):
        iid_series = self.dataset.id2token(self.dataset.iid_field, item_seq.cpu()) 
        rows, cols = iid_series.shape
        b = []
        for i in np.nditer(iid_series):
            if i != np.array('[PAD]'):
                externalid = str(i)

                text_tensor = self.content[externalid] # numpy数组格式为（H,W,C）,tensor数据格式是torch(C,H,W)
                
            else:
                text_tensor = torch.zeros([1,128]).cuda()
        
            
            b.append(text_tensor)

        tmixer_output = torch.cat(b)
        tmixer_output= torch.reshape(tmixer_output,(rows,50,128))
        tmixer_output = self.t_LayerNorm(tmixer_output)
        for _ in range(4):
            toutput = self.t_tokenMixer(tmixer_output)
            toutput = self.t_channelMixer(toutput)

        return toutput


    def forward(self, item_seq, item_seq_len):
        iid_series = self.dataset.id2token(self.dataset.iid_field, item_seq.cpu())
        
        item_emb = self.item_embedding(item_seq)
        mixer_output = self.LayerNorm(item_emb)
               
        for _ in range(self.n_layers):
            mixer_output = self.tokenMixer(mixer_output)
            mixer_output = self.channelMixer(mixer_output)

        fusemixer_output  = torch.cat((mixer_output,self.tMLPMixer(item_seq)),-1)
        fusemixer_output  = torch.cat((fusemixer_output,self.vMLPMixer(item_seq)),-1)#(Batch seq n*emb_size)
        
        fusemixer_output = self.LayerNormFeature(fusemixer_output)
        fusemixer_output = self.concat_layer_f(fusemixer_output)# [B H]
        seq_output = self.gather_indexes(fusemixer_output, item_seq_len - 1)
        seq_output = self.LayerNorm(seq_output)
        return seq_output



    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores