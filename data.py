# encoding: utf-8

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json
import h5py
import ipdb as pdb


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    """

    def __init__(self, data_path, data_split, vocab, image_sg_file, caption_data_file, feature_name):
        self.vocab = vocab
        
        # Captions
        with open(os.path.join(data_path, caption_data_file), "rb") as f:
            caption_data = json.load(f)
            print("Open", caption_data_file)
            self.cap_object_vocab = caption_data["objects"]
            self.cap_object_num = len(self.cap_object_vocab)
            self.cap_rel_vocab = caption_data["relations"]
            self.cap_pred_num = len(self.cap_rel_vocab)

            # scene graph
            scenes = caption_data["scenes"]
            self.caps = []
            self.cap_rel_per_img = []
            self.cap_rel_num_per_img = []
            self.cap_object_per_img = []
            self.cap_object_num_per_img = []

            self.cap_index_object_per_img = []
            self.cap_index_rel_per_img = []

            for captions in scenes:
                self.cap_object_per_img.append(captions["object"])
                self.cap_object_num_per_img.append(len(captions["object"]))
                self.cap_index_object_per_img.append(captions["object2idx"])
                self.cap_index_rel_per_img.append(captions["rel2idx"])

                sentence = captions["captions"]
                rel_5_sen = []
                count = 0
                for sentence in sentence:
                    self.caps.append(sentence["sent"])
                    for rel in sentence["rels_trip"]:
                        rel_5_sen.append(rel)
                    count = count + 1
                self.cap_rel_per_img.append(rel_5_sen)
                self.cap_rel_num_per_img.append(len(captions["rel2idx"]))
                if len(captions["rel2idx"]) == 0:
                    print(captions)


        # sg information
        with open(os.path.join(data_path, image_sg_file), 'r') as f:
            sg_data= json.load(f)

            self.obj_vocab = sg_data['object_dict']
            self.rel_vocab = sg_data['predicate_dict']
            self.object_num = sg_data['object_num']
            self.predicate_num = sg_data['predicate_num']

            img_to_sg = sg_data['img_to_sg']
            self.obj_num_per_img = [ sg['sg']['object'].index(-1) if -1 in sg['sg']['object'] else len(sg['sg']['object']) for sg in img_to_sg]
            self.predicate_num_per_img = [ sg['sg']['relationship'].index([-1,-1,-1]) if [-1,-1,-1] in sg['sg']['relationship'] else len(sg['sg']['relationship']) for sg in img_to_sg]

            self.obj_list_per_img = []
            self.rel_list_per_img = []
            self.filename_list = []
            for i, sg in enumerate(img_to_sg):
                self.filename_list.append(sg['filename'].split('.')[0])
                obj_num = self.obj_num_per_img[i]
                self.obj_list_per_img.append( sg['sg']['object'][:obj_num])
                pred_num = self.predicate_num_per_img[i]
                self.rel_list_per_img.append(sg['sg']['relationship'][:pred_num])


        print("data_path is ", data_path)
        if 'Flickr30k' in data_path:
            self.image_feature_root = os.path.join(data_path, 'npys', 'object')
            self.image_predicate_feature_root = os.path.join(data_path, 'npys', 'relation')
        elif feature_name == 'NM_BUA':
            self.image_feature_root = os.path.join(data_path, 'npys', 'object')
            self.image_predicate_feature_root = os.path.join(data_path, 'npys', 'relation')
        else:
            self.image_feature_root = os.path.join(data_path, 'npys', 'object')
            self.image_predicate_feature_root = os.path.join(data_path, 'npys', 'relation')
        self.length = len(self.caps)
        self.im_div = 5
        if data_split == 'val':
            self.length = 5000


    def __getitem__(self, index):
        # handle the image redundancy

        # Get obj features and obj list of an image
        img_id = index//self.im_div
        obj_num = self.obj_num_per_img[img_id]
        image = np.load(os.path.join(self.image_feature_root, self.filename_list[img_id]) + '.npy')[:obj_num]
        image = torch.from_numpy(image)
        obj_list = self.obj_list_per_img[img_id]
        
        
        # Get relation features and rel list of an image
        pred_num = self.predicate_num_per_img[img_id]
        image_predicate = np.load(os.path.join(self.image_predicate_feature_root, self.filename_list[img_id] + '.npy'))[:pred_num]
        image_predicate = torch.from_numpy(image_predicate)
        rel_list = self.rel_list_per_img[img_id]
        

        # Get the corresponding caption
        caption = self.caps[index]
        vocab = self.vocab
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)


        cap_idx_obj_list = self.cap_index_object_per_img[img_id]
        cap_obj_num = self.cap_object_num_per_img[img_id]
        cap_rel_num = self.cap_rel_num_per_img[img_id]
        cap_triad_idx_rel_list = self.cap_index_rel_per_img[img_id]

        return image, obj_num, target, index, img_id, image_predicate, pred_num, rel_list, obj_list, cap_idx_obj_list, cap_obj_num, cap_triad_idx_rel_list, cap_rel_num

    def __len__(self):
        return self.length


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)

    images, obj_nums, captions, ids, img_ids, image_predicates, pred_nums, rel_lists, obj_lists, cap_idx_obj_lists, cap_obj_nums, cap_triad_idx_rel_lists, cap_rel_nums = zip(*data)

    # Renumber all object in a batch
    objs = []
    for obj_list in obj_lists:
        objs = objs + obj_list
    objs = torch.tensor(objs, dtype=torch.int64)

    obj_offset = 0
    rels = []
    for i, rel_list in enumerate(rel_lists):
        img_id = img_ids[i]
        obj_num = obj_nums[i]
        pred_num = pred_nums[i]
        for s,p,o in rel_list:
            rels.append([s+obj_offset, p, o+obj_offset])
        obj_offset += obj_num
    rels = torch.tensor(rels, dtype=torch.int64)
    images = torch.cat(images,0)
    image_predicates = torch.cat(image_predicates,0)
    assert len(rels) == len(image_predicates)

    cap_objs = []
    for obj_index_list in cap_idx_obj_lists:
        cap_objs = cap_objs + obj_index_list
    cap_objs = torch.tensor(cap_objs, dtype=torch.int64)


    cap_objs_offset = 0
    cap_rels = []
    for i, rel_index_list in enumerate(cap_triad_idx_rel_lists):
        cap_obj_num = cap_obj_nums[i]
        img_id = img_ids[i]
        cap_rel_num = cap_rel_nums[i]
        for s, p, o in rel_index_list:
            cap_rels.append([s+cap_objs_offset, p, o+cap_objs_offset])
        cap_objs_offset += cap_obj_num
    cap_rels = torch.tensor(cap_rels, dtype=torch.int64)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    
    return images, obj_nums, targets, lengths, ids, image_predicates, pred_nums, rels, objs, cap_objs, cap_obj_nums, cap_rels, cap_rel_nums


def get_precomp_loader(data_path, data_split, file_list,feature_name, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, vocab, file_list[0], file_list[1], feature_name)
    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers = num_workers,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

    return data_loader, dset.obj_vocab, dset.rel_vocab, dset.cap_object_vocab, dset.cap_rel_vocab


def get_loaders(data_name, train_file_list, val_file_list, feature_name, vocab, batch_size, workers, opt):

    dpath = opt.data_path
    if opt.debug:
        train_loader, obj_vocab, rel_vocab, cap_obj_vocab, cap_rel_vocab = get_precomp_loader(dpath, 'val', val_file_list, feature_name, vocab, opt, batch_size, True, workers)
    else:
        train_loader, obj_vocab, rel_vocab, cap_obj_vocab, cap_rel_vocab = get_precomp_loader(dpath, 'train', train_file_list, feature_name, vocab, opt, batch_size, True, workers)
    val_loader, _, _, _, _ = get_precomp_loader(dpath, 'val', val_file_list, feature_name, vocab, opt, batch_size, False, workers)

    return train_loader, val_loader, obj_vocab, rel_vocab, cap_obj_vocab, cap_rel_vocab


def get_test_loader(split_name, data_name, test_file_list, feature_name, vocab, batch_size,
                    workers, opt):
    dpath = opt.data_path
    test_loader, _, _, cap_obj_vocab, cap_rel_vocab = get_precomp_loader(dpath, split_name, test_file_list, feature_name, vocab, opt. opt, batch_size, False, workers)
    return test_loader, cap_obj_vocab, cap_rel_vocab







    





















            
            

          