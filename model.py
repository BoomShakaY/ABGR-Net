# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_ as clip_grad_norm
import numpy as np
from collections import OrderedDict
from graph import GraphTripleConv, GraphTripleConvNet, FusionLayer



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def parse_glove_with_split(vocab, file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print(len(lines))
    embed_dim =0 
    word2vec = dict()
    for line in lines:
        line = line.strip('\n').split(' ')
        word = line[0]
        vec = [float(line[i]) for i in range(1,len(line))]
        vec = np.array(vec, dtype = float)
        embed_dim = vec.shape[0]
        word2vec[word] = vec
    idx2vec = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    
    for i in range(len(vocab)):
        word = vocab.idx2word[int(i)]
        if word in word2vec:
            idx2vec[i] = word2vec[word]
        elif word =='<start>':
            idx2vec[i] = np.random.uniform(-0.1, 0.1, embed_dim)
        elif word == '<end>':
            idx2vec[i] = np.zeros(embed_dim, dtype= np.float32)
        elif '_' in word:
            word = word.split('_')
            tmp = np.zeros((len(word), embed_dim), dtype=np.float32)
            for k,w in enumerate(word):
                if w in word2vec:
                    tmp[k] = word2vec[w]
            idx2vec[i] = np.mean(tmp, axis=0)
        elif '-' in word:
            word = word.split('-')
            tmp = np.zeros((len(word), embed_dim), dtype=np.float32)
            for k,w in enumerate(word):
                if w in word2vec:
                    tmp[k] = word2vec[w]
            idx2vec[i] = np.mean(tmp, axis=0)

        else:
            idx2vec[i] = np.random.uniform(-0.25, 0.25, embed_dim)

    return idx2vec


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, obj_nums, captions, cap_lens, opt):
    """
    Images: (n_image, n_objs, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_n_word = captions.size(1)

    ## padding caption use 0
    # for i in range(n_caption):
    #     n_word = cap_lens[i]
    #     captions[i,n_word:, :] = torch.zeros(max_n_word-n_word, captions.size(2), dtype= captions.dtype).cuda()

    cap_lens = torch.tensor(cap_lens, dtype=captions.dtype)
    cap_lens = Variable(cap_lens).cuda()
    captions = torch.transpose(captions, 1, 2)
    for i in range(n_image):
        n_obj = obj_nums[i]
        img_i = images[i, : n_obj, :].unsqueeze(0).contiguous()
        # --> (n_caption , n_region ,d)
        img_i_expand = img_i.repeat(n_caption, 1, 1)
        # --> (n_caption, d, max_n_word)
        dot = torch.bmm(img_i_expand, captions)
        dot = dot.max(dim=1, keepdim=True)[0].squeeze()
        dot = dot.sum(dim=1, keepdim=True)
        cap_lens = cap_lens.contiguous().view(-1, 1)
        if dot.shape != cap_lens.shape:
            print("Conflict dot.shape = ", dot.shape)
            print("Conflict cap_lens.shape", cap_lens.shape)

        dot = dot/cap_lens
        dot = torch.transpose(dot, 0, 1)
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 0)
    
    return similarities


def xattn_score_i2t(images, obj_nums, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    obj_nums:(n_obj) list of obj num per image
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_n_obj = images.size(1)

    obj_nums = torch.tensor(obj_nums, dtype=images.dtype)
    obj_nums = Variable(obj_nums).cuda()
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        cap_i_expand = cap_i_expand.contiguous()
        cap_i_expand = torch.transpose(cap_i_expand, 1,2)
        dot = torch.bmm(images, cap_i_expand)
        dot = dot.max(dim=2, keepdim=True)[0].squeeze()
        dot = dot.sum(dim=1, keepdim=True)
        obj_nums = obj_nums.contiguous().view(-1, 1)
        dot = dot/obj_nums
        similarities.append(dot)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, im_l, s, s_l, pred, pred_l, cap_o_pred, cap_o_l, c_r_pred, c_r_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            if self.opt.xattn_score == 'normal':
                scoresImg = xattn_score_t2i(im, im_l, s, s_l, self.opt)
                scoresTxt = xattn_score_t2i(pred, pred_l, c_r_pred, c_r_l, self.opt)
            scores = scoresImg + self.opt.predicate_score_rate * scoresTxt
        elif self.opt.cross_attn == 'i2t':
            if self.opt.xattn_score == 'normal':
                scoresImg = xattn_score_i2t(im, im_l, s, s_l, self.opt)
                scoresTxt = xattn_score_i2t(pred, pred_l, c_r_pred, c_r_l, self.opt)
            scores = scoresImg + self.opt.predicate_score_rate * scoresTxt
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class EncoderImageSg(nn.Module):
    def __init__(self, img_dim, gconv_dim, word_dim, obj_vocab, rel_vocab, 
    			 glove=None, no_imgnorm=True, is_fusion=True,
                 fusion_activation='relu', fusion_method='concatenate', 
                 gconv_hidden_dim=1024, gconv_num_layers=5, alpha=1.0,
                 mlp_normalization='none',activation=None,
               **kwargs):
        super(EncoderImageSg, self).__init__()

        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.glove = glove
        self.no_imgnorm = no_imgnorm
        self.obj_vocab = obj_vocab
        self.rel_vocab = rel_vocab
        
        # Embedding layers -- word2vec
        self.obj_embed = nn.Embedding(len(obj_vocab), word_dim)
        self.rel_embed = nn.Embedding(len(rel_vocab), word_dim)
 
        # Multi-modal fusion Layer
        if is_fusion:
            self.obj_fusion = FusionLayer(img_dim, word_dim, fusion_activation, fusion_method)
            self.rel_fusion = FusionLayer(img_dim, word_dim, fusion_activation, fusion_method)
        else:
            self.obj_fusion = None
            self.rel_fusion = None
        # GCN Net
        if gconv_num_layers == 0:
            self.gconv = nn.Linear(img_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
            'input_dim': img_dim,
            'output_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'alpha':alpha,
            'mlp_normalization': mlp_normalization,
            'activation': activation,
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
            'input_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'alpha':alpha,
            'num_layers': gconv_num_layers - 1,
            'mlp_normalization': mlp_normalization,
            'activation': activation
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        self.init_weights()

    def init_weights(self):
        if self.glove is None:
            self.obj_embed.weight.data.uniform_(-0.1, 0.1)
            self.rel_embed.weight.data.uniform_(-0.1, 0.1)
        else:
            idx2vec = parse_glove_with_split(self.obj_vocab, self.glove)
            self.obj_embed.weight.data.copy_(torch.from_numpy(idx2vec))

            idx2vec2 = parse_glove_with_split(self.rel_vocab, self.glove)
            self.rel_embed.weight.data.copy_(torch.from_numpy(idx2vec2))
        


    def forward(self, obj_embs, obj_nums, pred_embs, pred_nums, rels, objs):
        O, T = obj_embs.size(0), rels.size(0)
        s, p, o = rels.chunk(3, dim=1)           # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)

        obj_label_embs = self.obj_embed(objs)
        if self.obj_fusion is not None:
            obj_vecs = self.obj_fusion(obj_embs, obj_label_embs)
        else:
            obj_vecs = obj_embs
        
        pred_label_embs = self.rel_embed(p)
        if self.rel_fusion is not None:
            pred_vecs = self.rel_fusion(pred_embs, pred_label_embs)
        else:
            pred_vecs = pred_embs
        

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        if not self.no_imgnorm:
            obj_vecs = l2norm(obj_vecs, dim=-1)
            pred_vecs = l2norm(pred_vecs, dim=-1)

        return obj_vecs, pred_vecs

class EncoderTextGCN(nn.Module):
    def __init__(self, vocab_size, gconv_dim, word_dim, cap_obj_vocab, cap_rel_vocab, embed_size, num_layers,
                 vocab=None, glove=None, no_imgnorm=True,
                 use_bi_gru=False, no_txtnorm=False,
                 gconv_hidden_dim=1024, gconv_num_layers=5, alpha=1.0,
                 mlp_normalization='none', activation=None,
                 **kwargs):
        super(EncoderTextGCN, self).__init__()

        self.glove = glove
        self.vocab = vocab
        self.embed_size = embed_size
        self.obj_vocab = cap_obj_vocab
        self.rel_vocab = cap_rel_vocab
        self.no_imgnorm = no_imgnorm
        self.no_txtnorm = no_txtnorm

        self.word_embed = nn.Embedding(vocab_size, word_dim)
        self.obj_embed = nn.Embedding(len(cap_obj_vocab), word_dim)
        self.rel_embed = nn.Embedding(len(cap_rel_vocab), word_dim)

        self.use_bi_gru = use_bi_gru
        self.wrnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        if gconv_num_layers == 0:
            self.gconv = nn.Linear(word_dim, gconv_dim)
        elif gconv_num_layers > 0:
            gconv_kwargs = {
                'input_dim': word_dim,
                'output_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'alpha': alpha,
                'mlp_normalization': mlp_normalization,
                'activation': activation
            }
            self.gconv = GraphTripleConv(**gconv_kwargs)

        self.gconv_net = None
        if gconv_num_layers > 1:
            gconv_kwargs = {
                'input_dim': gconv_dim,
                'hidden_dim': gconv_hidden_dim,
                'alpha': alpha,
                'num_layers': gconv_num_layers-1,
                'mlp_normalization': mlp_normalization,
                'activation': activation
            }
            self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        self.init_weights()

    def init_weights(self):
        self.rel_embed.weight.data.uniform_(-0.1, 0.1)

        if self.glove is None:
            self.word_embed.weight.data.uniform_(-0.1, 0.1)
            self.obj_embed.weight.data.uniform_(-0.1, 0.1)
            self.rel_embed.weight.data.uniform_(-0.1, 0.1)
        else:
            idx2vec = parse_glove_with_split(self.vocab, self.glove)
            self.word_embed.weight.data.copy_(torch.from_numpy(idx2vec))
            idx2vec2 = parse_glove_with_split(self.obj_vocab, self.glove)
            self.obj_embed.weight.data.copy_(torch.from_numpy(idx2vec2))
            idx2vec3 = parse_glove_with_split(self.rel_vocab, self.glove)
            self.rel_embed.weight.data.copy_(torch.from_numpy(idx2vec3))


    def forward(self, x, lengths, cap_obj_nums, cap_pred_nums, cap_obj_list, cap_rel_list):
        # Embed word ids to vectors
        total_length = x.size(1)
        x = self.word_embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.wrnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padden = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padden
        cap_len = cap_len.cuda()

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2)/2] + cap_emb[:, :, cap_emb.size(2)/2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        O, I = cap_obj_list.size(0), cap_obj_list.size(0)
        s, p, o = cap_rel_list.chunk(3, dim=1)
        s, p, o = [item.squeeze(1) for item in [s, p, o]]
        edges = torch.stack([s, o], dim=1)
        cap_object = self.obj_embed(cap_obj_list)
        cap_predicate = self.rel_embed(p)
        obj_vecs = cap_object
        pred_vecs = cap_predicate

        if isinstance(self.gconv, nn.Linear):
            obj_vecs = self.gconv(obj_vecs)
        else:
            obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)

        if self.gconv_net is not None:
            obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        if not self.no_imgnorm:
            obj_vecs = l2norm(obj_vecs, dim=-1)
            pred_vecs = l2norm(pred_vecs, dim=-1)

        return cap_emb, cap_len, obj_vecs, pred_vecs
                 
class ABGR(object):
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImageSg(opt.img_dim, opt.embed_size, opt.word_dim, opt.obj_vocab, opt.rel_vocab, 
                                      glove = opt.glove, no_imgnorm = opt.no_imgnorm, is_fusion = opt.is_fusion, 
                                      fusion_activation = opt.fusion_activation, fusion_method = opt.fusion_method, 
                                      gconv_hidden_dim = 1024, gconv_num_layers = opt.gcn_num_layers,
                                      alpha = opt.alpha,
                                      activation = opt.activation)

        self.txt_enc = EncoderTextGCN(opt.vocab_size, opt.embed_size, opt.word_dim, opt.cap_obj_vocab, opt.cap_rel_vocab,
                                      opt.embed_size, opt.num_layers, vocab=opt.vocab,
                                      glove=opt.glove, no_imgnorm=opt.no_imgnorm,
                                      use_bi_gru=opt.bi_gru,
                                      no_txtnorm=opt.no_imgnorm,
                                      gconv_hidden_dim=1024, gconv_num_layers=opt.gcn_num_layers,
                                      alpha=opt.alpha,
                                      activation=opt.activation)


        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())


        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()


    def forward_emb(self, images, obj_nums, captions, lengths, 
    	            image_predicates, pred_nums, rels, objs, 
    	            cap_obj_nums, cap_pred_nums, cap_obj_list, cap_rel_list,
    	            max_obj_n=36, max_pred_n = 25, volatile=False):
        
        images = Variable(images, volatile=volatile)
        image_predicates = Variable(image_predicates, volatile=volatile)
        captions = Variable(captions, volatile=volatile)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            image_predicates = image_predicates.cuda()
            rels = rels.cuda()
            objs = objs.cuda()
            cap_obj_list = cap_obj_list.cuda()
            cap_rel_list = cap_rel_list.cuda()


        img_vecs, pred_vecs = self.img_enc(images, obj_nums, image_predicates, pred_nums, rels, objs)
        cap_emb, cap_lens, cap_obj_vecs, cap_pred_vecs = self.txt_enc(captions, lengths, cap_obj_nums, cap_pred_nums, cap_obj_list, cap_rel_list)
        max_cap_obj_n = max(cap_obj_nums)
        max_cap_pred_n = max(cap_pred_nums)

        img_emb = torch.zeros(len(obj_nums), max_obj_n, img_vecs.shape[1]).cuda()
        pred_emb = torch.zeros(len(pred_nums), max_pred_n, pred_vecs.shape[1]).cuda()

        cap_obj_emb = torch.zeros(len(cap_obj_nums), max_cap_obj_n, cap_obj_vecs.shape[1]).cuda()
        cap_pred_emb = torch.zeros(len(cap_pred_nums), max_cap_pred_n, cap_pred_vecs.shape[1]).cuda()

        obj_offset = 0
        for i, obj_num in enumerate(obj_nums):
            img_emb[i][: obj_num] = img_vecs[obj_offset: obj_offset+obj_num]
            obj_offset+= obj_num
        
        pred_offset = 0
        for i, pred_num in enumerate(pred_nums):
            pred_emb[i][: pred_num] = pred_vecs[pred_offset: pred_offset+pred_num]
            pred_offset+= pred_num

        cap_obj_offset = 0
        for i, cap_obj_num in enumerate(cap_obj_nums):
            cap_obj_emb[i][:cap_obj_num] = cap_obj_vecs[cap_obj_offset: cap_obj_offset+cap_obj_num]
            cap_obj_offset += cap_obj_num

        cap_pred_offset = 0
        for i, cap_pred_num in enumerate(cap_pred_nums):
            cap_pred_emb[i][:cap_pred_num] = cap_pred_vecs[cap_pred_offset: cap_pred_offset+cap_pred_num]
            # if cap_pred_emb[i][0][0] == 0:
            #     print("The index = ", i, "The cap_pred_num=", cap_obj_num)
            #     print("cap_pred_vecs = ", cap_pred_vecs)
            pred_offset += cap_pred_num


        return img_emb, cap_emb, cap_lens, pred_emb, cap_obj_emb, cap_pred_emb

    def forward_loss(self, img_emb, obj_nums, cap_emb, cap_len, pred_emb, pred_nums, cap_obj_emb, cap_obj_nums, cap_pred_emb, cap_pred_nums):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, obj_nums, cap_emb, cap_len, pred_emb, pred_nums, cap_obj_emb, cap_obj_nums, cap_pred_emb, cap_pred_nums)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, obj_nums, captions, lengths, ids, image_predicates, pred_nums, rels, objs, cap_objs, cap_obj_nums, cap_rels, cap_rel_nums):
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens, pred_emb, cap_obj_emb, cap_pred_emb = \
                     self.forward_emb(images, obj_nums, captions, lengths, image_predicates, pred_nums, rels, objs, cap_obj_nums, cap_rel_nums, cap_objs, cap_rels)

        self.optimizer.zero_grad()

        loss = self.forward_loss(img_emb, obj_nums, cap_emb, cap_lens, pred_emb, pred_nums, cap_obj_emb, cap_obj_nums, cap_pred_emb, cap_rel_nums)
    
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()