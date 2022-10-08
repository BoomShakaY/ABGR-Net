# encoding: utf-8

"""Evaluation"""

from __future__ import print_function
import os

import sys
from data import get_test_loader
import time
import numpy as np
import torch
from collections import OrderedDict
import time
from torch.autograd import Variable

from model import ABGR, xattn_score_t2i, xattn_score_i2t
from vocab import Vocabulary, deserialize_vocab, deserialize_vocab_from_list


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data_with_predicate(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader` """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()
    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    obj_nums = None
    pred_embs = None
    pred_nums = None
    cap_obj_embs = None
    cap_obj_nums = None
    cap_pred_embs =None
    cap_rel_nums = None

    max_n_word = 0
    max_n_cap_rel =0
    max_n_cap_obj = 0

    for i, (images, obj_num, captions, lengths, ids, image_predicates, pred_num, 
            rels, objs, cap_objs, cap_obj_num, cap_rels, cap_rel_num) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
        max_n_cap_obj = max(max_n_cap_obj, max(cap_obj_num)+1)
        max_n_cap_rel = max(max_n_cap_rel, max(cap_rel_num)+1)

    for i, (images, obj_num, captions, lengths, ids, image_predicates, pred_num, rels, objs,
            cap_objs, cap_obj_num, cap_rels, cap_rel_num) in enumerate(data_loader):
        
        # make sure val logger is used
        model.logger = val_logger
        img_emb, cap_emb, cap_len, pred_emb, cap_obj_emb, cap_pred_emb = \
            model.forward_emb(images, obj_num, captions, lengths, image_predicates,
                              pred_num, rels, objs, cap_obj_num, cap_rel_num, cap_objs, cap_rels)

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                pred_embs = np.zeros((len(data_loader.dataset), pred_emb.size(1), pred_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                pred_embs = np.zeros((len(data_loader.dataset), pred_emb.size(1)))
            if cap_obj_emb.dim() == 3:
                cap_obj_embs = np.zeros((len(data_loader.dataset), max_n_cap_obj, cap_obj_emb.size(2)))
                cap_pred_embs = np.zeros((len(data_loader.dataset), max_n_cap_rel, cap_pred_emb.size(2)))
            else:
                cap_obj_embs = np.zeros((len(data_loader.dataset), cap_obj_emb.size(1)))
                cap_pred_embs = np.zeros((len(data_loader.dataset), cap_pred_emb.size(1)))

            obj_nums = [0] * len(data_loader.dataset)
            pred_nums = [0] * len(data_loader.dataset)
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
            cap_obj_nums = [0] * len(data_loader.dataset)
            cap_rel_nums = [0] * len(data_loader.dataset)   


        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        pred_embs[ids] = pred_emb.data.cpu().numpy().copy()
        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
        cap_obj_embs[ids, :max(cap_obj_num), :] = cap_obj_emb.data.cpu().numpy().copy()
        cap_pred_embs[ids, :max(cap_rel_num), :] = cap_pred_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
            obj_nums[nid] = obj_num[j]
            pred_nums[nid] = pred_num[j]
            cap_obj_nums[nid] = cap_obj_num[j]
            cap_rel_nums[nid] = cap_rel_num[j]

        model.forward_loss(img_emb, obj_num, cap_emb, cap_len, pred_emb, pred_num, cap_obj_emb, cap_obj_num, cap_pred_emb, cap_rel_num)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions, rels, cap_rels, image_predicates
    return img_embs, obj_nums, cap_embs, cap_lens, pred_embs, pred_nums, cap_obj_embs, cap_obj_nums, cap_pred_embs, cap_rel_nums



def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model
    opt.vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(opt.vocab)

    # construct model
    model = ABGR(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])
    opt.workers = 4

    print('Loading dataset')
    test_file_list = [opt.data_name.replace('split', split) + opt.sg_file_name, 
                      opt.data_name.replace('split', split) + opt.caption_data_file_name]
    data_loader, cap_obj_vocab, cap_rel_vocab = get_test_loader(split, opt.data_name, test_file_list,  opt.feature_name, opt.vocab,
                                  opt.batch_size, opt.workers, opt)
    
    with torch.no_grad():
        print('Computing results...')
        img_embs, obj_nums, cap_embs, cap_lens, pred_embs, pred_nums, cap_obj_embs, cap_obj_nums, cap_pred_embs, cap_rel_nums \
                                                      = encode_data_with_predicate(model, data_loader)
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / 5, cap_embs.shape[0]))
        

        if not fold5:
            # no cross-validation, full evaluation
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
            obj_nums = [obj_nums[i] for i in range(0, len(obj_nums), 5)]
            pred_embs = np.array([pred_embs[i] for i in range(0, len(pred_embs), 5)])
            pred_nums = [pred_nums[i] for i in range(0, len(pred_nums), 5)]
            start = time.time()
            if opt.cross_attn == 't2i':
                sims1 = shard_xattn_t2i(img_embs, obj_nums, cap_embs, cap_lens, opt, shard_size=128)
                sims2 = shard_xattn_t2i(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, opt, shard_size=128)
                sims = sims1 + opt.predicate_score_rate * sims2
            elif opt.cross_attn == 'i2t':
                sims1 = shard_xattn_i2t(img_embs, obj_nums, cap_embs, cap_lens, opt, shard_size=128)
                sims2 = shard_xattn_i2t(pred_embs, pred_nums, cap_rel_embs, cap_rel_nums, opt, shard_size=128)
                sims = sims1 + opt.predicate_score_rate * sims2
                
            else:
                raise NotImplementedError
            end = time.time()
            
            print("calculate similarity time:", end-start)

            r, rt, i2t_results = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
            ri, rti, t2i_results = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
        
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            i2t_results = []
            t2i_results = []

            for i in range(5):
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                obj_nums_shard = obj_nums[i * 5000:(i + 1) * 5000:5]
                pred_embs_shard = pred_embs[i*5000:(i + 1) * 5000:5]
                pred_nums_shard = pred_nums[i*5000:(i + 1) * 5000:5]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
                cap_pred_embs_shard = cap_rel_embs[i * 5000:(i + 1) * 5000]
                cap_rel_nums_shard = cap_rel_nums[i * 5000:(i + 1) * 5000]
                start = time.time()

                if opt.cross_attn == 't2i':
                    sims1 = shard_xattn_t2i(img_embs_shard, obj_nums_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
                    sims2 = shard_xattn_t2i(pred_embs_shard, pred_nums_shard, cap_pred_embs_shard, cap_rel_nums_shard, opt, shard_size=128)
                    sims = sims1 + opt.predicate_score_rate * sims2

                elif opt.cross_attn == 'i2t':
                    sims1 = shard_xattn_i2t(img_embs_shard, obj_nums_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
                    sims2 = shard_xattn_i2t(pred_embs_shard, pred_nums_shard, cap_pred_embs_shard, cap_rel_nums_shard, opt, shard_size=128)
                    sims = sims1 + opt.predicate_score_rate * sims2      
                else:    
                    raise NotImplementedError

                
                end = time.time()
                print("calculate similarity time:", end-start)

                r, rt0, r_results = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0, ri_results = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
                i2t_results.append(r_results)
                t2i_results.append(ri_results)
                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[11])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Average t2i Recall: %.1f" % mean_metrics[12])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')
    return t2i_results, i2t_results


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p


def transform_pred_shape(score):
    trem1 = np.reshape(score, (-1, 1))
    term2 = term1
    for i in range(4):
        term2 = np.hstach((term2, term1))
    score_new = np.reshape(term2, (len(score), -1))
    return score_new


def shard_xattn_t2i(images, obj_nums, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            im_l =obj_nums[im_start:im_end]
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]

            if opt.xattn_score == "normal":
                sim = xattn_score_t2i(im, im_l, s, l, opt)

            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, obj_nums, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start: im_end]), volatile=True).cuda()
            im_l =obj_nums[im_start: im_end]
            s = Variable(torch.from_numpy(captions[cap_start: cap_end]), volatile=True).cuda()
            l = caplens[cap_start: cap_end]

            if opt.xattn_score == "normal":
                sim = xattn_score_i2t(im, im_l, s, l, opt)

            d[im_start: im_end, cap_start: cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    results = []
    for index in range(npts):
        result = dict()
        result['id'] = index
        inds = np.argsort(sims[index])[::-1]
        result['top1'] = inds[0]
        result['top5'] = list(inds[:5])
        result['top10'] = list(inds[:10])
        result['ranks'] = []
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            result['ranks'].append((i, tmp))
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

        if rank < 1:
            result['is_top1'] = 1
        else:
            result['is_top1'] = 0
        if rank < 5:
            result['is_top5'] = 1
        else:
            result['is_top5'] = 0

        results.append(result)

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1), results
    else:
        return (r1, r5, r10, medr, meanr), results


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T
    results = []
    for index in range(npts):
        for i in range(5):
            result = dict()
            result['id'] = 5*index + i
            inds = np.argsort(sims[5 * index + i])[::-1]
            result['top5'] = list(inds[:5])
            result['top10'] = list(inds[:10])
            result['top1'] = inds[0]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

            if ranks[5*index+i] < 1:
                result['is_top1'] = 1
            else:
                result['is_top1'] = 0

            if ranks[5*index+i] < 5:
                result['is_top5'] = 1
            else:
                result['is_top5'] = 0
            result['ranks'] = [(index, ranks[5*index+i])]
            results.append(result)

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1),results
    else:
        return (r1, r5, r10, medr, meanr), results

