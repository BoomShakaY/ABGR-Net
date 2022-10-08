# encoding: utf-8
"""Training script"""


import os
import time
import shutil
import torch
import numpy
import data

from vocab import Vocabulary, deserialize_vocab, deserialize_vocab_from_list
from model import ABGR
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data_with_predicate, shard_xattn_t2i, shard_xattn_i2t
from torch.autograd import Variable
import torch.nn as nn
import logging
import tensorboard_logger as tb_logger
import argparse
import ipdb as pdb




def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='./data/Flickr30k',
                        help='path to datasets')
    parser.add_argument('--data_name', default='flickr30k_precomp_split',
                        help='flickr30k_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--bi_gru', action='store_true',
                        help='Use bidirectional GRU.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--gcn_num_layers', default=1, type=int,
    	                help='Number of image sg  gcn layers')
    parser.add_argument('--activation', default=None, type=str,
                        help='activate function use in gcn')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='./runs/runZ/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--predicate_score_rate', type=float, default=1.0,
                        help ='the rate of the relationship scores')
    parser.add_argument('--cross_attn', default="t2i",
                        help='t2i|i2t')
    parser.add_argument('--debug', action='store_true', 
                        help='If in the debug mode, load the validation data and use pdb')
    parser.add_argument('--sg_file_name', type =str, default='_image_sg_by_NM_36_25.json',
                        help='The file that contains the visual scene graph information')
    parser.add_argument('--feature_name', type=str, default='BUA',
                        help='The features come from which detector')
    parser.add_argument('--caption_data_file_name', type=str, default='_caps_with_rel.json',
                        help='The file that contains the textual scene graph information')
    parser.add_argument('--is_fusion', type=bool, default=True,
                        help='If fuse the predicted label with the visual features or not.')
    parser.add_argument('--alpha', type =float, default=1.0)
    parser.add_argument('--glove', type=str,default='./glove/glove.6B.300d.txt',
                        help='The glove file to initialize the embedding layers.')
    parser.add_argument('--fusion_activation', type=str, default='relu',
                        help='The activation function of the fusion layer.')
    parser.add_argument('--fusion_method', type=str, default='concatenate',
                        help='The fusion method of the fusion layer.')
    parser.add_argument('--device_ids', type=str, default="0",
                        help='device_ids: e.g. 0 0,1 0,2')
    parser.add_argument('--xattn_score', type=str, default='normal',
                        help='normal')


    opt = parser.parse_args()
    print(opt)


    str_gpuIDs = opt.device_ids.split(',')
    gpu_IDs = []
    for _ in str_gpuIDs:
        gpu_IDs.append(int(_))
    num_gpu = len(gpu_IDs)
    print("the number of GPU is ", num_gpu)


    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    opt.vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(opt.vocab)

    print("loading data")
    # Get the train and val config filenames
    train_file_list = [opt.data_name.replace('split', 'train') + opt.sg_file_name, 
                           opt.data_name.replace('split', 'train') + opt.caption_data_file_name]
    val_file_list = [opt.data_name.replace('split', 'val') + opt.sg_file_name, 
                         opt.data_name.replace('split', 'val') + opt.caption_data_file_name]


    # Load the training data and validation data and the image object/relation vocab
    train_loader, val_loader, opt.obj_vocab, opt.rel_vocab, opt.cap_obj_vocab, opt.cap_rel_vocab = data.get_loaders(
        opt.data_name, train_file_list, val_file_list, opt.feature_name, opt.vocab, opt.batch_size, opt.workers, opt)
    opt.obj_vocab = deserialize_vocab_from_list(opt.obj_vocab)
    opt.rel_vocab = deserialize_vocab_from_list(opt.rel_vocab)
    opt.cap_obj_vocab = deserialize_vocab_from_list(opt.cap_obj_vocab)
    opt.cap_rel_vocab = deserialize_vocab_from_list(opt.cap_rel_vocab)
    print("loaded data")


    # Construct the model
    model = ABGR(opt)
    if num_gpu > 10:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_ids
        model = nn.DataParallel(model, device_ids=gpu_IDs)

    best_rsum = 0

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']

            if num_gpu > 1:
                model.module.load_state_dict(checkpoint['model'])
                model.module.Eiters = checkpoint['Eiters']
            else:
                module.load_state_dict(checkpoint['model'])
                module.Eiters = checkpoint['Eiters']

            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            start_epoch = 0
    else:
        start_epoch = 0


    """ Start training """
    for epoch in range(start_epoch, opt.num_epochs):
        if num_gpu > 1:
            adjust_learning_rate(opt, model.module.optimizer, epoch)
        else:
            adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        if opt.debug:
            pdb.set_trace()
        if num_gpu > 1:
            train(opt, train_loader, model.module, epoch, val_loader, num_gpu)
        else:
            train(opt, train_loader, model, epoch, val_loader, num_gpu)

        # evaluate on validation set
        if num_gpu > 1:
            rsum = validate(opt, val_loader, model.module, num_gpu)
        else:
            rsum = validate(opt, val_loader, model, num_gpu)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        if num_gpu > 1:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.module.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.Eiters,
            }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader, num_gpu):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model_name
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model, num_gpu)



def validate(opt, val_loader, model, num_gpu):
    # compute the encoding for all the validation images and captions
    model.val_start()
    with torch.no_grad():
        if num_gpu > 1:
            img_embs, obj_nums, cap_embs, cap_lens, pred_embs, pred_nums, cap_obj_embs, cap_obj_nums, cap_pred_embs, cap_rel_nums = encode_data_with_predicate(
                model.module, val_loader, opt.log_step, logging.info)
        else:
            img_embs, obj_nums, cap_embs, cap_lens, pred_embs, pred_nums, cap_obj_embs, cap_obj_nums, cap_pred_embs, cap_rel_nums = encode_data_with_predicate(
                model, val_loader, opt.log_step, logging.info)

        img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        obj_nums = [obj_nums[i] for i in range(0, len(obj_nums), 5)]
        pred_embs = numpy.array([pred_embs[i] for i in range(0, len(pred_embs), 5)])
        pred_nums = [pred_nums[i] for i in range(0, len(pred_nums), 5)]

        start = time.time()

        if opt.cross_attn == 't2i':
            sims1 = shard_xattn_t2i(img_embs, obj_nums, cap_embs, cap_lens, opt, shard_size=128)
            sims2 = shard_xattn_t2i(pred_embs, pred_nums, cap_pred_embs, cap_rel_nums, opt, shard_size=128)
            sims = sims1 + opt.predicate_score_rate * sims2
        elif opt.cross_attn == 'i2t':
            sims1 = shard_xattn_i2t(img_embs, obj_nums, cap_embs, cap_lens, opt, shard_size=128)
            sims2 = shard_xattn_i2t(pred_embs, pred_nums, cap_pred_embs, cap_rel_nums, opt, shard_size=128)
            sims = sims1 + opt.predicate_score_rate * sims2
        else:
            raise NotImplementedError

        end = time.time()
        print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr), _ = i2t(img_embs, cap_embs, cap_lens, sims)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))

    # image retrieval
    (r1i, r5i, r10i, medri, meanr),_ = t2i(
        img_embs, cap_embs, cap_lens, sims)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))

    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR and decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_yangzw(opt, optimizer, epoch):
    if epoch < 10:
        lr = opt.learning_rate * 0.1 * ((epoch+1) * 0.1)
    elif epoch >= 10 and epoch < 20:
        lr = opt.learning_rate * 0.1
    elif epoch >= 20 and epoch < 50:
        lr = opt.learning_rate * 0.01
    elif epoch >= 50:
        lr = opt.learning_rate * 0.001

    optimizer.param_groups[0]['lr'] = lr
    for i in range(len(optimizer.param_groups)-1):
        optimizer.param_groups[i+1]['lr'] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
