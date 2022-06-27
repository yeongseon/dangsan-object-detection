'''
    Author: Clément APAVOU
'''
import datasets.transforms as T
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import utils.callbacks as callbacks
import utils.metrics as ut_metrics
import utils.utils as utils
import utils.WandbLogger as WandbLogger
import wandb
import yaml
# from datasets.ReefDataset import ReefDataset, collate_fn
from datasets.LesionDataset import LesionDataset, collate_fn
from easydict import EasyDict
# from model.yolox.data.data_augment import ValTransform
from model.yolox.utils import postprocess
from torchmetrics.detection.map import MAP
from tqdm import tqdm

wandb.login()

# # Data initialization and loading
# from data import data_transforms


class Trainer():
    def __init__(self, config_file, logger, args):

        with open(config_file, 'r') as stream:
            try:
                self.config = EasyDict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

        if args.batch_size:
            self.config.configs.batch_size = args.batch_size

        if args.csv_file:
            self.config.data.csv_file = args.csv_file

        if args.root_path:
            self.config.data.root_path = args.root_path

        if args.it:
            self.config.configs.it = args.it

        if args.epoch:
            self.config.configs.epoch = args.epoch

        if args.notebook:
            from tqdm.notebook import tqdm

        self.fast_dev_run = args.fast_dev_run

        self.wandb_logger = WandbLogger(
            project=self.config.wandb.name_project,
            entity=self.config.wandb.get("entity") if args.no_test else None,
            name=self.config.wandb.get("name_run"),
            config=self.config)

        self.logger = logger
        ##############################
        #####  PREPARATION TRAIN #####
        ##############################
        self.logger.info("Preparation training parameters")

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device : {self.device}")
        if torch.cuda.is_available():
            self.logger.info(torch.cuda.get_device_name(0))

        # Seed
        torch.manual_seed(self.config.configs.get("seed", 1))

        # Model
        self.logger.info("Model : {}".format(self.config.model.name))

        model_cls = utils.import_class(self.config.model.name)
        self.model = model_cls(**self.config.model.get('params', {}))
        self.model.to(self.device)
        self.logger.info("Model : {}".format(self.model))
        self.wandb_logger.run.watch(self.model)

        if self.config.get('criterion'):
            self.logger.info("Loss function : {}".format(
                self.config.criterion.name))
            criterion_cls = utils.import_class(self.config.criterion.name)
            self.criterion = criterion_cls(
                **self.config.criterion.get('params', {}))

        # Optimizer
        self.logger.info("Optimizer : {}".format(self.config.optimizer.name))

        optimizer_cls = utils.import_class(self.config.optimizer.name)
        self.optimizer = optimizer_cls(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.config.optimizer.params)

        # Scheduler
        if self.config.get('scheduler'):
            self.logger.info("Scheduler : {}".format(
                self.config.scheduler.name))
            scheduler_cls = utils.import_class(self.config.scheduler.name)
            self.scheduler = scheduler_cls(self.optimizer,
                                           **self.config.scheduler.params)

        if args.load_checkpoint:
            self.logger.info("Loading checkpoint {}".format(
                args.load_checkpoint))
            checkpoint = torch.load(args.load_checkpoint,
                                    map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if hasattr(self, 'loss'):
                self.loss = checkpoint['loss']
            if hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])

        self.start_epoch = 1

        ##############################
        #####  PREPARATION DATA #####
        ##############################

        self.logger.info(f"Reading {self.config.data.data_dir}")

        conv_bbox = "pascal_voc" if "yolox" not in self.config.model.name else "yolo"
        format = "pascal_voc" if "yolox" not in self.config.model.name else "coco"

        train_set = LesionDataset(
            data_dir=self.config.data.data_dir,
            augmentation=self.config.augmentation,
            transforms=T.get_transform(
                True, self.config.augmentation, format=format
            ),
            mode='train'
        )
        val_set = LesionDataset(
            data_dir=self.config.data.data_dir,
            augmentation=self.config.augmentation,
            transforms=T.get_transform(
                False, self.config.augmentation, format=format
            ),
            mode='valid'
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.config.configs.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=self.config.configs.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers)

        self.logger.info("train : {}, validation : {}".format(
            len(train_loader.dataset), len(val_loader.dataset)))

        ##############################
        #####    ENTRAINEMENT    #####
        ##############################

        self.train_epoch(train_loader, val_loader, relaunch=args.relaunch)

    def train(self, epoch, train_loader):

        self.model.train()
        metrics = {}

        train_iterator = tqdm(train_loader,
                              position=1,
                              desc="Training...(loss=X.X)",
                              dynamic_ncols=True,
                              total=self.config.configs.get(
                                  'it', len(train_loader)),
                              leave=False)

        scaler = GradScaler()

        for batch_idx, (data, annot, targets, image_ids) in enumerate(
                train_iterator):  # put coefficient for each loss
                # targets not used in EfficientDet

            if  "yolox" not in self.config.model.name and \
                "EfficientDet" not in self.config.model.name:
                images = list(image.to(self.device) for image in data)
                targets = [{k: v.to(self.device)
                            for k, v in t.items() if 'img_size' != k } for t in targets]
                loss_dict = self.model(images, targets)
                loss = sum(l for l in loss_dict.values())

            elif "EfficientDet" in self.config.model.name:
                batch_size = len(image_ids) # self.config.configs.batch_size
                images = torch.stack(data)
                images = images.float().to(self.device)
                target = {}
                target["bbox"] = [a.to(self.device) for a in annot['boxes']]
                target["cls"] = [a.to(self.device) for a in annot["labels"]]
                target["img_scale"] = (
                    torch.tensor([1] * batch_size).float().to(self.device)
                )
                target["img_size"] = (
                    torch.tensor( annot["img_size"]).to(self.device).float()
                )

                loss_dict = self.model(images , target)
                loss = loss_dict['loss']
                # c_loss = output['class_loss']
                # b_loss = output['box_loss']

            else:
                images = torch.tensor(np.stack(list(data)))
                images = images.permute(0, 1, 3, 2)
                # FIXME error for yoloX : doit être un mutiple de 32
                # assert images.shape[2] == images.shape[
                #     3], f"{images.shape[2],images.shape[3]} square images required"
                targets_yolox = []
                for t in targets:
                    a = t['labels'].unsqueeze(1)
                    b = t['boxes']
                    te = torch.cat((a, b), 1)
                    targets_yolox.append(
                        torch.tensor(np.array(te)).to(self.device))

                loss_dict = self.model(images, targets_yolox)
                loss = loss_dict['total_loss']

            # backpropagation
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            # self.optimizer.step()
            scaler.step(self.optimizer)

            scaler.update()

            train_iterator.set_description("Training... (loss=%2.5f)" %
                                           loss.data.item())

            if "yolox" not in self.config.model.name:
                loss_dict['loss_sum'] = loss
                loss_dict = {
                    'train/' + k: v.detach().cpu().numpy()
                    for k, v in loss_dict.items()
                }
            else:
                loss_dict = {
                    'train/' + k:
                    v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in loss_dict.items()
                }

            self.wandb_logger.run.log(loss_dict)

            if batch_idx == 0 and \
               "EfficientDet" not in self.config.model.name:
                self.wandb_logger.log_images(
                    (data, targets), "train", 5
                )  #  FIXME wrong images and targets for yolox (parce que labels pas les mêmes pour l'affichage)

            if batch_idx >= self.config.configs.get('it', 100000):
                break

            if self.fast_dev_run:
                break

        return loss_dict

    def validation(self, val_loader, metrics_inst):

        self.model.eval()

        valid_iterator = tqdm(val_loader,
                              position=1,
                              desc="Validating...",
                              leave=False)

        metrics = {}

        for _, v in metrics_inst.items():
            v.reset()

        with torch.no_grad():
            for batch_idx, (data, annot, targets, image_ids) in enumerate(valid_iterator):

                if "yolo" not in self.config.model.name and \
                   "EfficientDet" not in self.config.model.name:
                    images = list(image.to(self.device) for image in data)
                    targets = [{k: v.to(self.device)
                                for k, v in t.items() if 'img_size' != k } for t in targets]

                    output = self.model(images, targets)

                    pred_bboxes_list = [
                        np.concatenate((
                            pred['scores'].unsqueeze(1).cpu().detach().numpy(),
                            pred['boxes'].cpu().detach().numpy()),
                            axis=1) for pred in output
                    ]
                elif "EfficientDet" in self.config.model.name:
                    batch_size = len(image_ids) # self.config.configs.batch_size
                    images = torch.stack(data)
                    images = images.float().to(self.device)
                    target = {}
                    target["bbox"] = [a.to(self.device) for a in annot['boxes']]
                    target["cls"] = [a.to(self.device) for a in annot["labels"]]
                    target["img_scale"] = (
                        torch.tensor([1] * batch_size).float().to(self.device)
                    )
                    target["img_size"] = (
                        torch.tensor( annot["img_size"]).to(self.device).float()
                    )

                    output = self.model(images, target) # output contains loss, class_loss, box_loss, detections(bbox, score, labels)
                    pred_bboxes_list = [
                        np.concatenate((
                            pred[:, -2].unsqueeze(1).cpu().detach().numpy(), # 100 scores 
                            pred[:, :-2].cpu().detach().numpy()), # 100 boxes
                        axis=1) for pred in output['detections']
                    ]
                    # for calculating MAP
                    output = [{
                            'labels' : pred[:, -1].detach(),
                            'scores' : pred[:, -2].detach(),
                            'boxes' : pred[:, :-2].detach()
                        } for pred in output['detections']
                    ]

                else:
                    images = torch.tensor(np.stack(list(data)))
                    images = images.permute(0, 1, 3, 2)

                    targets_yolox = []
                    for t in targets:
                        a = t['labels'].unsqueeze(1)
                        b = t['boxes']
                        te = torch.cat((a, b), 1)
                        targets_yolox.append(
                            torch.tensor(np.array(te)).to(self.device))

                    output = self.model(images, targets_yolox)
                    outputs = postprocess(
                        output,
                        self.config.model.params.num_classes,
                        self.config.model.inf_params.confthre,
                        self.config.model.inf_params.nmsthre,
                        class_agnostic=True)
                    # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                    # TODO compute pred_bboxes_list
                    pred_bboxes_list = [
                        np.array([[0, 0, 0, 0, 0]])
                        if pred is None else np.array(
                            torch.cat(
                                (pred[:, 5].unsqueeze(1),
                                 pred[:, 0].unsqueeze(1), pred[:,
                                                               1].unsqueeze(1),
                                 (pred[:, 2] - pred[:, 0]).unsqueeze(1),
                                 (pred[:, 3] - pred[:, 1]).unsqueeze(1)),
                                dim=1)) for pred in outputs
                    ]

                # Update metrics
                gt_bboxes_list = [t['boxes'].cpu().numpy() for t in targets]

                # metrics_inst["F2_score"].update(gt_bboxes_list, # 1 x 4 matrix
                #                                 pred_bboxes_list) # n x 5 matrix (score + boxes)

                # MAP
                for t in targets:
                    t['boxes'] = t['boxes'].cpu()
                    if len(t['boxes']) == 0:
                        t['labels'] = torch.tensor(
                            np.array([]), dtype=torch.int64)
                    else:
                        t['labels'] = t['labels'].cpu()
                    targets_map = {
                        'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()}

                targets_map = [
                    {'boxes': t['boxes'].cpu(), 'labels':t['labels'].cpu()} for t in targets
                    ]
                metrics_inst['MAP'].update(output, targets_map) # output need to contain boxes, scores, labels

                if batch_idx == 0:
                    self.wandb_logger.log_images((data, targets),
                                                 "validation",
                                                 5,
                                                 outputs=output)
                if self.fast_dev_run:
                    break
        
        metrics = {
            'validation/' + k: v.compute()
            for k, v in metrics_inst.items()
        }

        self.wandb_logger.log_videos((data, targets),
                                     "validation")  # TODO implement

        self.wandb_logger.log_metrics(metrics)

        return metrics

    def train_epoch(self, train_loader, val_loader, relaunch=False):
        self.logger.info("Launch training, start epoch : {}".format(
            self.start_epoch))

        # Init Metrics validation # TODO metrics instance {"train": , "validation": } en variable de classe
        metrics_instance = {
            # "F2_score":
            # ut_metrics.F2_score_competition(compute_on_step=False).to(
                # self.device),
            "MAP": MAP(compute_on_step=False).to(self.device)
        }

        # Init Early stopping
        if self.config.configs.get('early_stopping'):
            early_stopping = callbacks.EarlyStopping(
                monitor=self.config.configs.early_stopping.monitor,
                mode=self.config.configs.early_stopping.mode,
                patience=self.config.configs.early_stopping.patience,
                logger=self.logger)

        # Init Model checkpoint
        model_checkpoint = callbacks.ModelCheckpoint(
            monitor=self.config.configs.checkpoint.monitor,
            mode=self.config.configs.checkpoint.mode,
            run=self.wandb_logger.run,
            logger=self.logger)

        epoch_iterator = tqdm(range(self.start_epoch,
                                    self.config.configs.epoch + 1),
                              total=self.config.configs.epoch,
                              initial=self.start_epoch - 1,
                              position=0,
                              desc="Epoch",
                              leave=False)

        for current_epoch in epoch_iterator:

            metrics_train = self.train(current_epoch, train_loader)

            if relaunch:
                real_epoch = current_epoch + (self.config.configs.epoch *
                                              relaunch)
            else:
                real_epoch = current_epoch

            ############################
            #####    VALIDATION    #####
            ############################
            metrics_validation = self.validation(val_loader, metrics_instance)

            metrics = metrics_train.copy()
            metrics.update(metrics_validation)

            self.wandb_logger.run.log(
                {"lr": self.optimizer.param_groups[0]['lr']})

            if self.config.get('scheduler'):
                if self.config.scheduler.get('monitor'):
                    self.scheduler.step(
                        metrics[self.config.scheduler.get('monitor')])
                else:
                    self.scheduler.step()

            if hasattr(self, 'fold'):
                self.results_k_folds[str(self.fold)].append(metrics)

            # Update Early stopping
            if self.config.configs.get('early_stopping'):
                res = early_stopping.update(metrics)
                if res != None:
                    break

            # Checkpoint
            # model_checkpoint.save_checkpoint(
            #     self,
            #     metrics,
            #     real_epoch,
            #     fold=self.fold if hasattr(self, 'fold') else None)

            model_checkpoint.save_weights(
                self.model, metrics, real_epoch,
                self.fold if hasattr(self, 'fold') else None)

        model_checkpoint.save_checkpoint(
            self,
            metrics,
            real_epoch,
            fold=self.fold if hasattr(self, 'fold') else None,
            name="last_checkpoint",
            end=True)
