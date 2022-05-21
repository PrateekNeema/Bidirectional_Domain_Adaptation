import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pcs.models import (CosineClassifier, MemoryBank, SSDALossModule,
                        compute_variance, loss_info, torch_kmeans,
                        update_data_memory)
from pcs.utils import (AverageMeter, datautils, is_div, per, reverse_domain,
                       torchutils, utils)
from sklearn import metrics
from tqdm import tqdm

from . import BaseAgent

ls_abbr = {
    "cls-so": "cls",
    "proto-each": "P",
    "proto-src": "Ps",
    "proto-tgt": "Pt",
    "cls-info": "info",
    "I2C-cross": "C",
    "semi-condentmax": "sCE",
    "semi-entmin": "sE",
    "tgt-condentmax": "tCE",
    "tgt-entmin": "tE",
    "ID-each": "I",
    "CD-cross": "CD",
}


class CDSAgent(BaseAgent):
    #################################
    # -----------------------------------DONE------------------------------------------
    def __init__(self, config):
        self.config = config
        self._define_task(config)
        self.is_features_computed = False
        self.current_iteration_source = self.current_iteration_target = 0

        self.domain_map = {
            "source": self.config.data_params.source,                #dslr
            "target": self.config.data_params.target,                 #amazon
        }

        #####################
        super(CDSAgent, self).__init__(config)    ##here the load_datasets,etc functions are called of CDS agent through base agent class
        #********************    @@@@@@@@decpetive place -a lot is happening from here
        # done

        # for MIM
        self.momentum_softmax_target = torchutils.MomentumSoftmax(
            self.num_class, m=len(self.get_attr("target", "train_loader"))
        )
        self.momentum_softmax_source = torchutils.MomentumSoftmax(
            self.num_class, m=len(self.get_attr("source", "train_loader"))
        )

        # init loss
        loss_fn = SSDALossModule(self.config, gpu_devices=self.gpu_devices)
        loss_fn = nn.DataParallel(loss_fn, device_ids=self.gpu_devices).cuda()
        self.loss_fn = loss_fn
        ###This is the complete loss fucntion.
        ##It has all the different losses of all stages as different attributes


        if self.config.pretrained_exp_dir is None:
            self._init_memory_bank()                     ####big steps being taken here

        # init statics
        self._init_labels()
        self._load_fewshot_to_cls_weight()

    #################
    # -----------------------------------DONE------------------------------------------
    def _define_task(self, config):
        # specify task
        self.fewshot = config.data_params.fewshot    #1
        self.clus = config.loss_params.clus != None   #true # clus dictionary of "clus": {"kmeans_freq": 1,"type": ["each"],"n_k": 15,"k": [31, 62]}
        self.cls = self.semi = self.tgt = self.ssl = False
        self.is_pseudo_src = self.is_pseudo_tgt = False
        for ls in config.loss_params.loss:
            self.cls = self.cls | ls.startswith("cls")          #true....
            self.semi = self.semi | ls.startswith("semi")
            self.tgt = self.tgt | ls.startswith("tgt")
            self.ssl = self.ssl | (ls.split("-")[0] not in ["cls", "semi", "tgt"])
            self.is_pseudo_src = self.is_pseudo_src | ls.startswith("semi-pseudo")    #false
            self.is_pseudo_tgt = self.is_pseudo_tgt | ls.startswith("tgt-pseudo")     #false

        self.is_pseudo_src = self.is_pseudo_src | (            #true
            config.loss_params.pseudo and self.fewshot is not None
        )
        self.is_pseudo_tgt = self.is_pseudo_tgt | (
            config.loss_params.pseudo and self.fewshot is not None   #true
        )
        self.semi = self.semi | self.is_pseudo_src         #true
        if self.clus:
            self.is_pseudo_tgt = self.is_pseudo_tgt | (                      #true
                config.loss_params.clus.tgt_GC == "PGC" and "GC" in config.clus.type
            )

    #################
    # -----------------------------------DONE------------------------------------------
    def _init_labels(self):
        train_len_tgt = self.get_attr("target", "train_len")
        train_len_src = self.get_attr("source", "train_len")

        # labels for pseudo
        if self.fewshot:

            self.predict_ordered_labels_pseudo_source = (
                torch.zeros(train_len_src, dtype=torch.long).detach().cuda() - 1    ##tensor of labels as -1 for all samples of source
            )
            for ind, lbl in zip(self.fewshot_index_source, self.fewshot_label_source):
                self.predict_ordered_labels_pseudo_source[ind] = lbl
                ##setting values of fewshot images to known labels

            self.predict_ordered_labels_pseudo_target = (
                    torch.zeros(train_len_tgt, dtype=torch.long).detach().cuda() - 1     ##tensor of labels as -1 for all samples of source
            )
            for ind, lbl in zip(self.fewshot_index_target, self.fewshot_label_target):
                self.predict_ordered_labels_pseudo_target[ind] = lbl
                ##setting values of fewshot images to known labels



        # self.predict_ordered_labels_pseudo_target = (
        #     torch.zeros(train_len_tgt, dtype=torch.long).detach().cuda() - 1
        # )     ##labels of all samples of target
        #$$$$$$$ add code of line 107 here to set values of fewshot iamges in target to known values$$$$$$$$

    #################
    # -----------------------------------DONE------------------------------------------
    def _load_datasets(self):
        name = self.config.data_params.name     #office
        num_workers = self.config.data_params.num_workers  #4
        fewshot = self.config.data_params.fewshot             #1
        domain = self.domain_map                   #source - dslr, target  - amazon
        image_size = self.config.data_params.image_size #224
        aug_src = self.config.data_params.aug_src    # "aug_0"
        aug_tgt = self.config.data_params.aug_tgt   #"aug_0"
        raw = "raw"

        # ##assuming that all classes are there in the source and therfore they have just taken self.num_class as classes in source dataset
        # self.num_class = datautils.get_class_num(           ##returns num of classes in dslr.txt (source)
        #     f'data/splits/{name}/{domain["source"]}.txt'
        # )
        # self.class_map = datautils.get_class_map(            ## returns dictionary with class index and names in dslr.txt
        #     f'data/splits/{name}/{domain["target"]}.txt'
        # )

        self.src_num_classes = 7  ## dslr 
        self.tgt_num_classes = 23  ## amazon
        self.num_class = 31

        #self.class_map =

        batch_size_dict = {
            "test": self.config.optim_params.batch_size,   #64
            "source": self.config.optim_params.batch_size_src,   #64 (same as batch_size)
            "target": self.config.optim_params.batch_size_tgt,     #64   (same as batch_size)
            "labeled": self.config.optim_params.batch_size_lbd,   #64
        }
        self.batch_size_dict = batch_size_dict

        # self-supervised Dataset
        for domain_name in ("source", "target"):
            aug_name = {"source": aug_src, "target": aug_tgt}[domain_name]

            ##  aug_name = {"source": 'aug_0', "target": 'aug_0'}["source"]

            # Training datasets
            train_dataset = datautils.create_dataset(
                name,
                domain[domain_name],                      ##dslr or amazon
                suffix="",
                ret_index=True,
                image_transform=aug_name,
                use_mean_std=False,
                image_size=image_size,
            )
            ##images in dslr.txt/amazon.txt are operated and then stored as
            #image list objects(they have tensors of images in them as well along with other details) in train_dataset


            train_loader = datautils.create_loader(
                train_dataset,
                batch_size_dict[domain_name],
                is_train=True,
                num_workers=num_workers,
            )
            ###returns the Dataloader object
            ##The DataLoader class is designed so that it can be iterated using the enumerate() function,
            # which returns a tuple with the current batch zero-based index value, and the actual batch of data

            train_init_loader = datautils.create_loader(
                train_dataset,
                batch_size_dict[domain_name],
                is_train=False,
                num_workers=num_workers,
            )
            train_labels = torch.from_numpy(train_dataset.labels).detach().cuda()

            self.set_attr(domain_name, "train_dataset", train_dataset)   ## train_dataset_source = train_dataset
            self.set_attr(domain_name, "train_ordered_labels", train_labels)
            self.set_attr(domain_name, "train_loader", train_loader)
            self.set_attr(domain_name, "train_init_loader", train_init_loader)
            self.set_attr(domain_name, "train_len", len(train_dataset))
            ##note that these were for the complete datasets

        # for loop ends

        # Classification and Fewshot Dataset
        if fewshot:

            train_lbd_dataset_source = datautils.create_dataset(
                name,
                domain["source"],          #dslr
                suffix=f"bi_labeled_{fewshot}",
                ret_index=True,                                ##note that keep_in_mem = False
                image_transform=aug_src,
                image_size=image_size,
            )

            train_lbd_dataset_target = datautils.create_dataset(
                name,
                domain["target"],  # amazon
                suffix=f"bi_labeled_{fewshot}",
                ret_index=True,                                        ##note that keep_in_mem = False
                image_transform=aug_src,
                image_size=image_size,
            )
            ##########labelled source dataset object
            ## list of images is given in splits folder as dslr_labeled_1.txt for 1 lablled image per class
            ##only these images will be present in this object

            src_dataset = self.get_attr("source", "train_dataset")
            tgt_dataset = self.get_attr("target", "train_dataset")
            (
                self.fewshot_index_source,
                self.fewshot_label_source,
            ) = datautils.get_fewshot_index(train_lbd_dataset_source, src_dataset)
            (
                self.fewshot_index_target,
                self.fewshot_label_target,
            ) = datautils.get_fewshot_index(train_lbd_dataset_target, tgt_dataset)

            ##index of fewshots in complete dataset and their labels

            test_unl_dataset_source = datautils.create_dataset(
                name,
                domain["source"],
                suffix=f"bi_unlabeled_{fewshot}",
                ret_index=True,
                image_transform=raw,      #@@@@@@@@@
                image_size=image_size,
            )

            test_unl_dataset_target = datautils.create_dataset(
                name,
                domain["target"],
                suffix=f"bi_unlabeled_{fewshot}",
                ret_index=True,
                image_transform=raw,  # @@@@@@@@@
                image_size=image_size,
            )

            ###the images that are officially unlabbeld in the source...basically complete dataset - fewshots

            self.test_unl_loader_source = datautils.create_loader(
                test_unl_dataset_source,
                batch_size_dict["test"],
                is_train=False,
                num_workers=num_workers,
            )
            self.test_unl_loader_target = datautils.create_loader(
                test_unl_dataset_target,
                batch_size_dict["test"],
                is_train=False,
                num_workers=num_workers,
            )

            # labels for source fewshot
            train_len_src = self.get_attr("source", "train_len")
            self.fewshot_labels_in_source = (
                torch.zeros(train_len_src, dtype=torch.long).detach().cuda() - 1
            )
            for ind, lbl in zip(self.fewshot_index_source, self.fewshot_label_source):
                self.fewshot_labels_in_source[ind] = lbl
            ##creating a tensor of fewshot index and label

            # labels for target fewshot
            train_len_tgt = self.get_attr("target", "train_len")
            self.fewshot_labels_in_target = (
                torch.zeros(train_len_tgt, dtype=torch.long).detach().cuda() - 1
            )
            for ind, lbl in zip(self.fewshot_index_target, self.fewshot_label_target):
                self.fewshot_labels_in_target[ind] = lbl
            ##creating a tensor of fewshot index and label



        else:   ##when fewshot is 0/False
            train_lbd_dataset_source = datautils.create_dataset(
                name,
                domain["source"],
                ret_index=True,
                image_transform=aug_src,
                image_size=image_size,
            )
            train_lbd_dataset_target = datautils.create_dataset(
                name,
                domain["target"],
                ret_index=True,
                image_transform=aug_src,
                image_size=image_size,
            )


        ###############
        # test_suffix = "test" if self.config.data_params.train_val_split else ""          ###""
        # test_unl_dataset_target = datautils.create_dataset(
        #     name,
        #     domain["target"],       ##amazon
        #     suffix=test_suffix,
        #     ret_index=True,
        #     image_transform=raw,
        #     image_size=image_size,
        # )
        ##Complete amamzon dataset for ___testing___
        #no need -- defined earlier


        self.train_lbd_loader_source = datautils.create_loader(
            train_lbd_dataset_source,
            batch_size_dict["labeled"],
            num_workers=num_workers,
        )
        self.train_lbd_loader_target = datautils.create_loader(
            train_lbd_dataset_target,
            batch_size_dict["labeled"],
            num_workers=num_workers,
        )

        # self.test_unl_loader_target = datautils.create_loader(
        #     test_unl_dataset_target,
        #     batch_size_dict["test"],
        #     is_train=False,
        #     num_workers=num_workers,
        # )
        # no need -- defined earlier
        ###the two dataloader objects

        self.logger.info(
            f"Dataset {name}, source {self.config.data_params.source}, target {self.config.data_params.target}"
        )

    #################
    # -----------------------------------DONE------------------------------------------
    def _create_model(self):
        version_grp = self.config.model_params.version.split("-")
        version = version_grp[-1]                               ## resnet50
        pretrained = "pretrain" in version_grp           ##True
        if pretrained:
            self.logger.info("Imagenet pretrained model used")
        out_dim = self.config.model_params.out_dim           ## 512

        # backbone
        if "resnet" in version:
            net_class = getattr(torchvision.models, version)      ##resnet50 actual model stored here

            if pretrained:
                model = net_class(pretrained=pretrained)    ##creates an object of resnet50 here
                model.fc = nn.Linear(model.fc.in_features, out_dim)  ##last fully connected layer reshaped
                # For more info
                ##https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks
                torchutils.weights_init(model.fc)
            else:
                model = net_class(pretrained=False, num_classes=out_dim)
        else:
            raise NotImplementedError

        model = nn.DataParallel(model, device_ids=self.gpu_devices)
        model = model.cuda()
        self.model = model   ### we have the resnet50 model ready now

        # classification head
        if self.cls:
            self.criterion = nn.CrossEntropyLoss().cuda()
            cls_head = CosineClassifier(
                num_class=self.num_class, inc=out_dim, temp=self.config.loss_params.T
            )
            torchutils.weights_init(cls_head)
            self.cls_head = cls_head.cuda()      ##cosine classifer model is ready now

    #################
    # -----------------------------------DONE------------------------------------------
    def _create_optimizer(self):
        lr = self.config.optim_params.learning_rate
        momentum = self.config.optim_params.momentum
        weight_decay = self.config.optim_params.weight_decay
        conv_lr_ratio = self.config.optim_params.conv_lr_ratio

        parameters = []

        # batch_norm layer: no weight_decay

        params_bn, _ = torchutils.split_params_by_name(self.model, "bn")
        parameters.append({"params": params_bn, "weight_decay": 0.0})

        # conv layer: small lr
        _, params_conv = torchutils.split_params_by_name(self.model, ["fc", "bn"])
        if conv_lr_ratio:
            parameters[0]["lr"] = lr * conv_lr_ratio
            parameters.append({"params": params_conv, "lr": lr * conv_lr_ratio})
        else:
            parameters.append({"params": params_conv})

        # fc layer
        params_fc, _ = torchutils.split_params_by_name(self.model, "fc")
        if self.cls and self.config.optim_params.cls_update:
            params_fc.extend(list(self.cls_head.parameters()))
        parameters.append({"params": params_fc})

        ##parameters now is the list of paratmetres in all the layers including resnet50 layers and the cls_head(fc) layer

        self.optim = torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=self.config.optim_params.nesterov,
        )
        ##we now have the optimiser ready

        # lr schedular
        if self.config.optim_params.lr_decay_schedule:
            optim_stepLR = torch.optim.lr_scheduler.MultiStepLR(
                self.optim,
                milestones=self.config.optim_params.lr_decay_schedule,
                gamma=self.config.optim_params.lr_decay_rate,
            )
            self.lr_scheduler_list.append(optim_stepLR)
        ##We now have the deacy schedule of the learning rate

        if self.config.optim_params.decay:
            self.optim_iterdecayLR = torchutils.lr_scheduler_invLR(self.optim)

    #################
    # -----------------------------------DONE------------------------------------------
    def train_one_epoch(self):
        # train preparation
        self.model = self.model.train()          ##set the model in training mode

        if self.cls:
            self.cls_head.train()        ##set cls in training mode
        self.loss_fn.module.epoch = self.current_epoch

        loss_list = self.config.loss_params.loss
        loss_weight = self.config.loss_params.weight
        loss_warmup = self.config.loss_params.start
        loss_giveup = self.config.loss_params.end

        num_loss = len(loss_list)           ##number of losses

        source_loader = self.get_attr("source", "train_loader")
        target_loader = self.get_attr("target", "train_loader")

        if self.config.steps_epoch is None:
            num_batches = max(len(source_loader), len(target_loader)) + 1
            self.logger.info(f"source loader batches: {len(source_loader)}")
            self.logger.info(f"target loader batches: {len(target_loader)}")
        else:
            num_batches = self.config.steps_epoch

        epoch_loss = AverageMeter()
        epoch_loss_parts = [AverageMeter() for _ in range(num_loss)]  #creating list of AverageMeter objects for each loss in loss_list

        # cluster bactches for kmeans frequency
        if self.clus:
            if self.config.loss_params.clus.kmeans_freq:
                kmeans_batches = num_batches // self.config.loss_params.clus.kmeans_freq
            else:
                kmeans_batches = 1
        else:
            kmeans_batches = None

        ######$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$!!!!!!!!!!!!!!!!!!!
        # load weight
        self._load_fewshot_to_cls_weight()
        if self.fewshot:
            fewshot_index_src = torch.tensor(self.fewshot_index_source).cuda()  ##tensor of fewshot_index
            fewshot_index_tgt = torch.tensor(self.fewshot_index_target).cuda()  ##tensor of fewshot_index

        tqdm_batch = tqdm(
            total=num_batches, desc=f"[Epoch {self.current_epoch}]", leave=False
        )                                           ##progress bar
        tqdm_post = {}


        for batch_i in range(num_batches):

            # Kmeans
            if is_div(kmeans_batches, batch_i):    ##kmeans with certain frequency
                self._update_cluster_labels()           ######### lot of steps being covered inside here

            if not self.config.optim_params.cls_update:
                self._load_fewshot_to_cls_weight()

            # iteration over all source images                #complete source dataset
            if not batch_i % len(source_loader):     #True when divisible
                source_iter = iter(source_loader)

                if "semi-condentmax" in self.config.loss_params.loss:
                    momentum_prob_source = (
                        self.momentum_softmax_source.softmax_vector.cuda()
                    )
                    self.momentum_softmax_source.reset()

            # iteration over all target images                  #complete target dataset
            if not batch_i % len(target_loader):
                target_iter = iter(target_loader)

                if "tgt-condentmax" in self.config.loss_params.loss:
                    momentum_prob_target = (
                        self.momentum_softmax_target.softmax_vector.cuda()
                    )
                    self.momentum_softmax_target.reset()


            # iteration over all labeled source images
            if self.cls and not batch_i % len(self.train_lbd_loader_source):
                source_lbd_iter = iter(self.train_lbd_loader_source)
            else:
                source_lbd_iter = iter(self.train_lbd_loader_source)

            # iteration over all labeled target images
            if self.cls and not batch_i % len(self.train_lbd_loader_target):
                target_lbd_iter = iter(self.train_lbd_loader_target)
            else:
                target_lbd_iter = iter(self.train_lbd_loader_target)


            # calculate loss
            for domain_name in ("source", "target"):

                loss = torch.tensor(0).cuda()
                loss_d = 0
                loss_part_d = [0] * num_loss
                batch_size = self.batch_size_dict[domain_name]

                if self.cls:                                                     # and domain_name == "source":
                    indices_lbd, images_lbd, labels_lbd = next(source_lbd_iter)
                    #these are the indices,images and labels of the feshots in the source
                    indices_lbd = indices_lbd.cuda()
                    images_lbd = images_lbd.cuda()
                    labels_lbd = labels_lbd.cuda()

                    feat_lbd = self.model(images_lbd)
                    feat_lbd = F.normalize(feat_lbd, dim=1)
                    out_lbd = self.cls_head(feat_lbd)   ##passing fewshot images through model and cls_head
                    ##softmax is only used to get probabilities

                # Matching & ssl ###########????
                #if (self.tgt and domain_name == "target") or self.ssl:       ## False or True = True
                if self.ssl:
                    loader_iter = (
                        source_iter if domain_name == "source" else target_iter        ##for 1st passage it is source_iter
                    )
                    ## both source_loader and target_loader are full datasets

                    indices_unl, images_unl, _ = next(loader_iter)
                    # why are these named as unl when its full dataset??
                    # Also address the error of "Local variable 'target_iter' might be referenced before assignment"
                    images_unl = images_unl.cuda()
                    indices_unl = indices_unl.cuda()
                    feat_unl = self.model(images_unl)
                    feat_unl = F.normalize(feat_unl, dim=1)
                    out_unl = self.cls_head(feat_unl)        ### This will be n samples each of length 31

                # Semi Supervised
                if self.semi and domain_name == "source":
                    semi_mask = ~torchutils.isin(indices_unl, fewshot_index_src)   ## What is this??
                    ## My guess is it is a tensor with True at fewhsot indices and False elsewhere
                    indices_semi = indices_unl[semi_mask]          # ??  # guess - Only unlablled indices
                    out_semi = out_unl[semi_mask]                  # ??   ## only unlablled outputs from the models

                if self.semi and domain_name == "target":
                    semi_mask = ~torchutils.isin(indices_unl, fewshot_index_tgt)   ## What is this??
                    ## My guess is it is a tensor with True at fewhsot indices and False elsewhere
                    indices_semi = indices_unl[semi_mask]          # ??  # guess - Only unlablled indices
                    out_semi = out_unl[semi_mask]                  # ??   ## only unlablled outputs from the models

                # Self-supervised Learning        ## is of use to me
                ## coudnt go inside -- assuming its is just loss calculation as mentioned in the pdf
                if self.ssl:
                    _, new_data_memory, loss_ssl, aux_list = self.loss_fn(
                        indices_unl, feat_unl, domain_name, self.parallel_helper_idxs
                    )
                    loss_ssl = [torch.mean(ls) for ls in loss_ssl]


                # pseudo
                loss_pseudo = torch.tensor(0).cuda()
                is_pseudo = {"source": self.is_pseudo_src, "target": self.is_pseudo_tgt}  ## both true
                thres_dict = {
                    "source": self.config.loss_params.thres_src,   # 0.99
                    "target": self.config.loss_params.thres_tgt,   # 0.99
                }

                if is_pseudo[domain_name]:
                    if domain_name == "source":
                        indices_pseudo = indices_semi           ## only unlablled indices here
                        out_pseudo = out_semi                   ## outputs of only unlablled images through models
                        pseudo_domain = self.predict_ordered_labels_pseudo_source  ##samples with labels = -1 except fewshot ones
                    else:
                        indices_pseudo = indices_semi  ## only unlablled indices here
                        out_pseudo = out_semi  ## outputs of only unlablled images through models
                        pseudo_domain = self.predict_ordered_labels_pseudo_target  ##samples with labels = -1 except fewshot ones

                    thres = thres_dict[domain_name]

                    # calculate loss
                    loss_pseudo, aux = torchutils.pseudo_label_loss(          ## get high confidence predictions and their losses(suppose a predction is 94% to be class 3 then hand-wavy loss could be 6% although here CE loss is used)
                        out_pseudo,
                        thres=thres,
                        mask=None,
                        num_class=self.num_class,
                        aux=True,
                    )
                    mask_pseudo = aux["mask"]

                    # updating fewshot memory bank
                    if domain_name == "source":
                        mb = self.get_attr("source", "memory_bank_wrapper")
                        indices_lbd_tounl = fewshot_index_src[indices_lbd]           ##  indices of fewhshots
                        mb_feat_lbd = mb.at_idxs(indices_lbd_tounl)                     ## current mem bank repres of fewshots
                        fewshot_data_memory_src = update_data_memory(mb_feat_lbd, feat_lbd)   ## new updated mem bank ouput repre of fewshots
                    else:
                        mb = self.get_attr("target", "memory_bank_wrapper")
                        indices_lbd_tounl = fewshot_index_tgt[indices_lbd]  ##  indices of fewhshots
                        mb_feat_lbd = mb.at_idxs(indices_lbd_tounl)  ## current mem bank repres of fewshots
                        fewshot_data_memory_tgt = update_data_memory(mb_feat_lbd,feat_lbd)  ## new updated mem bank ouput repre of fewshots

                    # stat
                    pred_selected = out_pseudo.argmax(dim=1)[mask_pseudo]   ## predictions of high confidence
                    indices_selected = indices_pseudo[mask_pseudo]        ## Their indices
                    indices_unselected = indices_pseudo[~mask_pseudo]

                    pseudo_domain[indices_selected] = pred_selected
                    ## The high confidence predictions are now set as known and are added to the fewshots (now considered labelled)
                    pseudo_domain[indices_unselected] = -1

                # Compute Loss
                ###########
                for ind, ls in enumerate(loss_list):

                    if (
                        self.current_epoch < loss_warmup[ind]
                        or self.current_epoch >= loss_giveup[ind]
                    ):
                        continue

                    loss_part = torch.tensor(0).cuda()
                    # *** handler for different loss ***

                    # classification on few-shot
                    # ## will be done on both
                    if ls == "cls-so":                                  #and domain_name == "source":
                        loss_part = self.criterion(out_lbd, labels_lbd)        ## cross entropy loss on fewshots

                    # ## will be done on both
                    elif ls == "cls-info":                               # and domain_name == "source":
                        loss_part = loss_info(feat_lbd, mb_feat_lbd, labels_lbd)

                    # semi-supervision learning on unlabled source
                    elif ls == "semi-entmin" and domain_name == "source":
                        loss_part = torchutils.entropy(out_semi)
                    elif ls == "semi-condentmax" and domain_name == "source":
                        bs = out_semi.size(0)
                        prob_semi = F.softmax(out_semi, dim=1)
                        prob_mean_semi = prob_semi.sum(dim=0) / bs

                        # update momentum
                        self.momentum_softmax_source.update(
                            prob_mean_semi.cpu().detach(), bs
                        )
                        # get momentum probability
                        momentum_prob_source = (
                            self.momentum_softmax_source.softmax_vector.cuda()
                        )
                        # compute loss
                        entropy_cond = -torch.sum(
                            prob_mean_semi * torch.log(momentum_prob_source + 1e-5)
                        )
                        loss_part = -entropy_cond

                    # learning on unlabeled target domain                     ## $$$$$
                    elif ls == "tgt-entmin" and domain_name == "target":
                        loss_part = torchutils.entropy(out_unl)
                    elif ls == "tgt-condentmax" and domain_name == "target":
                        bs = out_unl.size(0)
                        prob_unl = F.softmax(out_unl, dim=1)
                        prob_mean_unl = prob_unl.sum(dim=0) / bs

                        # update momentum
                        self.momentum_softmax_target.update(
                            prob_mean_unl.cpu().detach(), bs
                        )
                        # get momentum probability
                        momentum_prob_target = (
                            self.momentum_softmax_target.softmax_vector.cuda()
                        )
                        # compute loss
                        entropy_cond = -torch.sum(
                            prob_mean_unl * torch.log(momentum_prob_target + 1e-5)
                        )
                        loss_part = -entropy_cond

                    # self-supervised learning
                    elif ls.split("-")[0] in ["ID", "CD", "proto", "I2C", "C2C"]:
                        loss_part = loss_ssl[ind]

                    loss_part = loss_weight[ind] * loss_part
                    loss = loss + loss_part                      ### sum of all weighted losses
                    loss_d = loss_d + loss_part.item()
                    loss_part_d[ind] = loss_part.item()

                # Backpropagation
                self.optim.zero_grad()       ## sets all gradients to 0
                if len(loss_list) and loss != 0:
                    loss.backward()                 ## Computes the gradient of current tensor
                self.optim.step()       ## optimizer step

                # update memory_bankn
                if self.ssl:
                    self._update_memory_bank(domain_name, indices_unl, new_data_memory)     ## just puts new_data_memmory at the given indices

                    if domain_name == "source":               ## $$$$$$$$$$$
                        self._update_memory_bank(                                     ## put this fewwhshot_data_mem at these indices for source
                            domain_name, indices_lbd_tounl, fewshot_data_memory_src
                        )
                    if domain_name == "target":               ## $$$$$$$$$$$
                        self._update_memory_bank(                                     ## put this fewwhshot_data_mem at these indices for source
                            domain_name, indices_lbd_tounl, fewshot_data_memory_tgt
                        )   ## add for target as well

                # update lr info
                tqdm_post["lr"] = torchutils.get_lr(self.optim, g_id=-1)

                # update loss info
                epoch_loss.update(loss_d, batch_size)
                tqdm_post["loss"] = epoch_loss.avg
                self.summary_writer.add_scalars(
                    "train/loss", {"loss": epoch_loss.val}, self.current_iteration
                )
                self.train_loss.append(epoch_loss.val)

                # update loss part info
                domain_iteration = self.get_attr(domain_name, "current_iteration")
                self.summary_writer.add_scalars(
                    f"train/{self.domain_map[domain_name]}_loss",
                    {"loss": epoch_loss.val},
                    domain_iteration,
                )
                for i, ls in enumerate(loss_part_d):
                    ls_name = loss_list[i]
                    epoch_loss_parts[i].update(ls, batch_size)
                    tqdm_post[ls_abbr[ls_name]] = epoch_loss_parts[i].avg
                    self.summary_writer.add_scalars(
                        f"train/{self.domain_map[domain_name]}_loss",
                        {ls_name: epoch_loss_parts[i].val},
                        domain_iteration,
                    )

                # adjust lr
                if self.config.optim_params.decay:
                    self.optim_iterdecayLR.step()

                self.current_iteration += 1
            tqdm_batch.set_postfix(tqdm_post)
            tqdm_batch.update()
            self.current_iteration_source += 1
            self.current_iteration_target += 1

        tqdm_batch.close()

        self.current_loss = epoch_loss.avg

    #################
    # -----------------------------------DONE------------------------------------------
    @torch.no_grad()
    def _load_fewshot_to_cls_weight(self):
        """load centroids to cosine classifier           #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        Args:
            method (str, optional): None, 'fewshot', 'src', 'tgt'. Defaults to None.
        """
        method = self.config.model_params.load_weight   #src-tgt???????/alwasy src-tgt....what do the others do??

        if method is None:
            return
        assert method in ["fewshot", "src", "tgt", "src-tgt", "fewshot-tgt"]  ##move ahead only if method in this list

        thres = {"src": 1, "tgt": 1}            ##src-1,tgt-1
        bank = {
            "src": self.get_attr("source", "memory_bank_wrapper").as_tensor(),
            "tgt": self.get_attr("target", "memory_bank_wrapper").as_tensor(),
        }
        fewshot_label = {}
        fewshot_index = {}

        is_tgt = True
        #(
        #     method in ["tgt", "fewshot-tgt", "src-tgt"]
        #     and self.current_epoch >=self.config.model_params.load_weight_epoch
        # )
        ###  when current epoch>= 0, then is_tgt will become true

        if method in ["fewshot", "fewshot-tgt"]:
            if self.fewshot:
                fewshot_label["src"] = torch.tensor(self.fewshot_label_source)
                fewshot_index["src"] = torch.tensor(self.fewshot_index_source)
            else:
                fewshot_label["src"] = self.get_attr("source", "train_ordered_labels")
                fewshot_index["src"] = torch.arange(
                    self.get_attr("source", "train_len")
                )
        else:
            mask = self.predict_ordered_labels_pseudo_source != -1   ###mask are the known labels of the fewshots
            fewshot_label["src"] = self.predict_ordered_labels_pseudo_source[mask]
            fewshot_index["src"] = mask.nonzero().squeeze(1)

        if is_tgt:
            mask = self.predict_ordered_labels_pseudo_target != -1         ## for target
            fewshot_label["tgt"] = self.predict_ordered_labels_pseudo_target[mask]
            fewshot_index["tgt"] = mask.nonzero().squeeze(1)

        for domain in ("src", "tgt"):        #################edits to be made here
            # if domain == "tgt" and not is_tgt:
            #     break
            # if domain == "src" and method == "tgt":
            #     break

            #The weights will now be updated for both src and tgt when number is greater than 1

            weight = self.cls_head.fc.weight.data    ###weights of all the 31 classes from last(fc) layer of cls_head  ##??

            for label in range(self.num_class):

                fewshot_mask = fewshot_label[domain] == label     ##fewshots of the current class (there will be only one in DA-1 case)
                if fewshot_mask.sum() < thres[domain]:   ##thres[src]=1 and thres[tgt]=1
                    continue
                fewshot_ind = fewshot_index[domain][fewshot_mask]   ##list of indices of fewshots
                bank_vec = bank[domain][fewshot_ind]       ##mem_bank represenatation of fewshots in source

                weight[label] = F.normalize(torch.mean(bank_vec, dim=0), dim=0)
                ##weight of the class is updated here according to the fewshots repres in the memory bank
                ##the new weight is the normalized mean of the memory_bank representations of the fewshot smaples
                ####$$$$$$$$$$$$$$@@@@@@@@@@@@@@@@&&&&&&$*()@((((((((((((

    # Validate
    #################
    # -----------------------------------DONE------------------------------------------
    @torch.no_grad()
    def validate(self):
        self.model.eval()    ##chnage mode of model

        # Domain Adaptation
        if self.cls:
            # self._load_fewshot_to_cls_weight()
            self.cls_head.eval()                          ###chnage mode of cls_head
            if (
                self.config.data_params.fewshot
                and self.config.data_params.name not in ["visda17", "digits"]     ##True
            ):
                self.current_src_acc = self.score(
                    self.test_unl_loader_source,                   ##score only on unlballed images of dslr
                    name=f"unlabeled {self.domain_map['source']}",
                )

            self.current_tgt_acc = self.score(               ###This is accuracy on the target domain
                self.test_unl_loader_target,                  ##score on unlableled images of target$$$$$$$
                name=f"unlabeled {self.domain_map['target']}",
            )

        # update information
        self.current_val_iteration += 1
        if self.current_tgt_acc >= self.best_tgt_acc:
            self.best_tgt_acc = self.current_tgt_acc
            self.best_tgt_acc_epoch = self.current_epoch
            #self.iter_with_no_improv = 0
        #else:
            #self.iter_with_no_improv += 1

        if self.current_src_acc >= self.best_src_acc:
            self.best_src_acc = self.current_src_acc
            self.best_src_acc_epoch = self.current_epoch
            #self.iter_with_no_improv = 0
        #else:
            #self.iter_with_no_improv += 1


        self.tgt_val_acc.append(self.current_tgt_acc)
        self.src_val_acc.append(self.current_src_acc)

        self.clear_train_features()

    #################
    # -----------------------------------DONE------------------------------------------
    @torch.no_grad()
    def score(self, loader, name="test"):  ##calcuate the score,accuracy and losses by computing ouptut from model,cls and compaoring with actual results
        correct = 0
        size = 0
        epoch_loss = AverageMeter()
        error_indices = []
        confusion_matrix = torch.zeros(self.num_class, self.num_class, dtype=torch.long)
        pred_score = []
        pred_label = []
        label = []

        #$$$$$$$$$$$$$one of the most important parts
        for batch_i, (indices, images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            feat = self.model(images)             #!!!
            feat = F.normalize(feat, dim=1)       #!!!
            output = self.cls_head(feat)          #!!!
            prob = F.softmax(output, dim=-1)      #!!!

            loss = self.criterion(output, labels)     #!!!
            pred = torch.max(output, dim=1)[1]        #!!!

            pred_label.extend(pred.cpu().tolist())    #predicted labels...appending prediction to predicted_labels
            label.extend(labels.cpu().tolist())     ##appending actual labels

            if self.num_class == 2:
                pred_score.extend(prob[:, 1].cpu().tolist())

            correct += pred.eq(labels).sum().item()  ##total number of samples that have the correct predictions across all batches
            for t, p, ind in zip(labels, pred, indices):

                confusion_matrix[t.long(), p.long()] += 1      #!!!##this matrix will give the nuber of times
                # what was predicted and homww many times it was correct ad how many times it was worn and what was wrong
                if t != p:
                    error_indices.append((ind, p))      ##list of wrong predicrtions

            size += pred.size(0)                 ##total nuber of samples covered till now in this epoch
            epoch_loss.update(loss, pred.size(0))   ##updating the epoch loss

        acc = correct / size                             ###accuracy = correct /total samples in epcoh
        self.summary_writer.add_scalars(
            "test/acc", {f"{name}": acc}, self.current_epoch
        )
        self.summary_writer.add_scalars(                  ##average epoch loss
            "test/loss", {f"{name}": epoch_loss.avg}, self.current_epoch
        )
        self.logger.info(
            f"[Epoch {self.current_epoch} {name}] loss={epoch_loss.avg:.5f}, acc={correct}/{size}({100. * acc:.3f}%)"
        )

        return acc

    # Load & Save checkpoint
    def load_checkpoint(
        self,
        filename,
        checkpoint_dir=None,
        load_memory_bank=False,
        load_model=True,
        load_optim=False,
        load_epoch=False,
        load_cls=True,
    ):
        checkpoint_dir = checkpoint_dir or self.config.checkpoint_dir
        filename = os.path.join(checkpoint_dir, filename)
        try:
            self.logger.info(f"Loading checkpoint '{filename}'")
            checkpoint = torch.load(filename, map_location="cpu")

            if load_epoch:
                self.current_epoch = checkpoint["epoch"]
                for domain_name in ("source", "target"):
                    self.set_attr(
                        domain_name,
                        "current_iteration",
                        checkpoint[f"iteration_{domain_name}"],
                    )
                self.current_iteration = checkpoint["iteration"]
                self.current_val_iteration = checkpoint["val_iteration"]

            if load_model:
                model_state_dict = checkpoint["model_state_dict"]
                self.model.load_state_dict(model_state_dict)

            if load_cls and self.cls and "cls_state_dict" in checkpoint:
                cls_state_dict = checkpoint["cls_state_dict"]
                self.cls_head.load_state_dict(cls_state_dict)

            if load_optim:
                optim_state_dict = checkpoint["optim_state_dict"]
                self.optim.load_state_dict(optim_state_dict)

                lr_pretrained = self.optim.param_groups[0]["lr"]
                lr_config = self.config.optim_params.learning_rate

                # Change learning rate
                if not lr_pretrained == lr_config:
                    for param_group in self.optim.param_groups:
                        param_group["lr"] = self.config.optim_params.learning_rate

            self._init_memory_bank()
            if (
                load_memory_bank or self.config.model_params.load_memory_bank == False
            ):  # load memory_bank
                self._load_memory_bank(
                    {
                        "source": checkpoint["memory_bank_source"],
                        "target": checkpoint["memory_bank_target"],
                    }
                )

            self.logger.info(
                f"Checkpoint loaded successfully from '{filename}' at (epoch {checkpoint['epoch']}) at (iteration s:{checkpoint['iteration_source']} t:{checkpoint['iteration_target']}) with loss = {checkpoint['loss']}\nval acc = {checkpoint['val_acc']}\n"
            )

        except OSError as e:
            self.logger.info(f"Checkpoint doesnt exists: [{filename}]")
            raise e

    ################
    # -----------------------------------DONE------------------------------------------
    def save_checkpoint(self, filename="checkpoint.pth.tar"):
        out_dict = {
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "memory_bank_source": self.get_attr("source", "memory_bank_wrapper"),
            "memory_bank_target": self.get_attr("target", "memory_bank_wrapper"),
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "iteration_source": self.get_attr("source", "current_iteration"),
            "iteration_target": self.get_attr("target", "current_iteration"),
            "val_iteration": self.current_val_iteration,
            "tgt_val_acc": np.array(self.tgt_val_acc),
            "src_val_acc": np.array(self.src_val_acc),
            "current_tgt_acc": self.current_tgt_acc,
            "current_src_acc": self.current_src_acc,
            "loss": self.current_loss,
            "train_loss": np.array(self.train_loss),
        }
        if self.cls:
            out_dict["cls_state_dict"] = self.cls_head.state_dict()

        # best according to target
        is_best_tgt = (
            self.current_tgt_acc == self.best_tgt_acc
        ) or not self.config.validate_freq
        torchutils.save_checkpoint(
            out_dict, is_best_tgt, filename=filename, folder=self.config.checkpoint_dir
        )
        self.copy_checkpoint()

    ########################
    # -----------------------------------DONE------------------------------------------
    # compute train features
    @torch.no_grad()
    def compute_train_features(self):
        if self.is_features_computed:
            return
        else:
            self.is_features_computed = True
        self.model.eval()                      ##sets the model ready for new mode

        for domain in ("source", "target"):

            train_loader = self.get_attr(domain, "train_init_loader")
            features, y, idx = [], [], []

            tqdm_batch = tqdm(
                total=len(train_loader), desc=f"[Compute train features of {domain}]"          ##for progress bar
            )

            for batch_i, (indices, images, labels) in enumerate(train_loader):   ##gives batch index, and actual batch of data
                images = images.to(self.device)
                feat = self.model(images) ##forward propgates the batch of images in the resnet50 model and gets output from nn model
                feat = F.normalize(feat, dim=1)   ##makes mod 1

                features.append(feat)      ####adding the features into a list
                y.append(labels)     ##adding full batch details here
                idx.append(indices)

                tqdm_batch.update()
            tqdm_batch.close()

            features = torch.cat(features)          ##making these lists to tensors
            y = torch.cat(y)
            idx = torch.cat(idx).to(self.device)

            self.set_attr(domain, "train_features", features)   ##naming them as source_train_features....
            self.set_attr(domain, "train_labels", y)
            self.set_attr(domain, "train_indices", idx)

    #############
    # -----------------------------------DONE------------------------------------------
    def clear_train_features(self):
        self.is_features_computed = False

    ########################
    # -----------------------------------DONE------------------------------------------
    # Memory bank
    @torch.no_grad()
    def _init_memory_bank(self):
        out_dim = self.config.model_params.out_dim  #512

        for domain_name in ("source", "target"):

            data_len = self.get_attr(domain_name, "train_len")     #total number of samples in source/target
            memory_bank = MemoryBank(data_len, out_dim)
            ##memory bank object is ready

            if self.config.model_params.load_memory_bank:

                self.compute_train_features()             ##########

                idx = self.get_attr(domain_name, "train_indices")
                feat = self.get_attr(domain_name, "train_features")
                memory_bank.update(idx, feat)   #updating the features/represenation of the images in the memory bank
                # self.logger.info(
                #     f"Initialize memorybank-{domain_name} with pretrained output features"
                # )
                # save space

                if self.config.data_params.name in ["visda17", "domainnet"]:
                    delattr(self, f"train_indices_{domain_name}")
                    delattr(self, f"train_features_{domain_name}")

            self.set_attr(domain_name, "memory_bank_wrapper", memory_bank)

            self.loss_fn.module.set_attr(domain_name, "data_len", data_len)
            self.loss_fn.module.set_broadcast(                                   ###broadcast memeory bank to gpu devices
                domain_name, "memory_bank", memory_bank.as_tensor()
            )

    @torch.no_grad()
    def _update_memory_bank(self, domain_name, indices, new_data_memory):
        memory_bank_wrapper = self.get_attr(domain_name, "memory_bank_wrapper")
        memory_bank_wrapper.update(indices, new_data_memory)
        updated_bank = memory_bank_wrapper.as_tensor()
        self.loss_fn.module.set_broadcast(domain_name, "memory_bank", updated_bank)

    def _load_memory_bank(self, memory_bank_dict):
        """load memory bank from checkpoint

        Args:
            memory_bank_dict (dict): memory_bank dict of source and target domain
        """
        for domain_name in ("source", "target"):
            memory_bank = memory_bank_dict[domain_name]._bank.cuda()
            self.get_attr(domain_name, "memory_bank_wrapper")._bank = memory_bank
            self.loss_fn.module.set_broadcast(domain_name, "memory_bank", memory_bank)

    #######################
    # Cluster
    @torch.no_grad()
    def _update_cluster_labels(self):

        k_list = self.config.k_list   ## 15*[31,62]

        for clus_type in self.config.loss_params.clus.type:     ##clus_type = "each"
            cluster_labels_domain = {}
            cluster_centroids_domain = {}
            cluster_phi_domain = {}

            # clustering for each domain!!!!!
            if clus_type == "each":
                for domain_name in ("source", "target"):

                    memory_bank_tensor = self.get_attr(
                        domain_name, "memory_bank_wrapper"
                    ).as_tensor()

                    # clustering
                    cluster_labels, cluster_centroids, cluster_phi = torch_kmeans(
                        k_list,
                        memory_bank_tensor,
                        seed=self.current_epoch + self.current_iteration,
                    )

                    cluster_labels_domain[domain_name] = cluster_labels
                    cluster_centroids_domain[domain_name] = cluster_centroids
                    cluster_phi_domain[domain_name] = cluster_phi

                self.cluster_each_centroids_domain = cluster_centroids_domain
                self.cluster_each_labels_domain = cluster_labels_domain
                self.cluster_each_phi_domain = cluster_phi_domain
            else:
                print(clus_type)
                raise NotImplementedError

            # update cluster to losss_fn
            for domain_name in ("source", "target"):

                self.loss_fn.module.set_broadcast(
                    domain_name,
                    f"cluster_labels_{clus_type}",
                    cluster_labels_domain[domain_name],
                )
                self.loss_fn.module.set_broadcast(
                    domain_name,
                    f"cluster_centroids_{clus_type}",
                    cluster_centroids_domain[domain_name],
                )
                if cluster_phi_domain:
                    self.loss_fn.module.set_broadcast(
                        domain_name,
                        f"cluster_phi_{clus_type}",
                        cluster_phi_domain[domain_name],
                    )
