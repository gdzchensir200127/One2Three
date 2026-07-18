import time
from datetime import timedelta


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.base import CheckpointRunner
from functions.evaluator import Evaluator
from models.classifier import Classifier
from models.losses.classifier import CrossEntropyLoss
from models.losses.p2m import P2MLoss
from models.losses.s2p import S2PLoss
from models.losses.p2s import P2SLoss
from models.losses.signal2pixel import Signal2PixelLoss
from models.losses.signal2pixel_out4 import Signal2PixelOut4Loss
from models.losses.signal_refiner import SignalRefinerLoss
from models.p2m import P2MModel
from models.s2p import S2PModel
from models.s2p_4views import S2P_4views_Model
from models.s2p_4views_Transformer import S2P_4views_Transfomer
from models.s2p_4views_Transformer_add import S2P_4views_Transfomer_add
from models.s2p_4views_proj_v4_ge import Signal2Pixel_Model
from models.s2p_4views_proj_4view import Signal2Pixel_4views_Model
from models.signal_refiner import SignalRefiner
# from models.s2p_2views_test import S2P_2views_test_Model
# from models.s2p_2views_conv import S2P_2views_conv_Model
from models.p2s import P2SModel
from models.discriminator import Discriminator
from utils.average_meter import AverageMeter
from utils.mesh import Ellipsoid
from utils.tensor import recursive_detach
# from utils.vis.renderer import MeshRenderer
from utils.vis.displayer import PtDisplayer
from torch.autograd import Variable
from torch import Tensor


class Trainer(CheckpointRunner):

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        if self.options.model.name == "pixel2mesh":
            # Visualization renderer
            # self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
            #                              self.options.dataset.mesh_pos)
            # create ellipsoid
            self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)
        elif self.options.model.name == "signal2point":
            self.renderer = None
            self.displayer = True
        elif self.options.model.name == "signal2point_2views":
            self.renderer = None
            self.displayer = True
        elif self.options.model.name == "signal2point_4views":
            self.renderer = None
            self.displayer = True
        elif self.options.model.name == "signal2point_4views_transformer":
            self.renderer = None
            self.displayer = True
        elif self.options.model.name == "signal2point_4views_transformer_add":
            self.renderer = None
            self.displayer = True
        elif self.options.model.name == "signal2point_GAN":
            self.renderer = None
            self.displayer = True
        elif self.options.model.name == 'point2signal':
            self.renderer = None
            self.displayer = None
        elif self.options.model.name == 'signal_refiner':
            self.renderer = None
            self.displayer = None
        elif self.options.model.name == "signal2pixel":
            self.renderer = None
            self.displayer = None
        elif self.options.model.name == "signal2pixel_4views":
            self.renderer = None
            self.displayer = None
        elif self.options.model.name == "signal2pixel_4views_out4":
            self.renderer = None
            self.displayer = None
        else:
            self.renderer = None

        if shared_model is not None:
            self.model = shared_model
        else:
            if self.options.model.name == "pixel2mesh":
                # create model
                self.model = P2MModel(self.options.model, self.ellipsoid,
                                      self.options.dataset.camera_f, self.options.dataset.camera_c,
                                      self.options.dataset.mesh_pos)
            elif self.options.model.name == "classifier":
                self.model = Classifier(self.options.model, self.options.dataset.num_classes)
            elif self.options.model.name == "signal2point":
                self.model = S2PModel(self.options.model)
            elif self.options.model.name == "signal2point_4views":
                self.model = S2P_4views_Model(self.options.model)
            elif self.options.model.name == "signal2point_4views_transformer":
                self.model = S2P_4views_Transfomer(self.options.model)
            elif self.options.model.name == "signal2point_4views_transformer_add":
                self.model = S2P_4views_Transfomer_add(self.options.model)
            # elif self.options.model.name == "signal2point_2views":
            #     # self.model = S2P_2views_test_Model(self.options.model)
            #     self.model = S2P_2views_conv_Model(self.options.model)
            elif self.options.model.name == "signal2point_GAN":
                self.model = S2P_4views_Model(self.options.model)
                self.Discriminator = Discriminator(self.options.discriminator)
            elif self.options.model.name == 'point2signal':
                self.model = P2SModel(self.options.model)
            elif self.options.model.name == 'signal_refiner':
                self.model = SignalRefiner()
            elif self.options.model.name == "signal2pixel":
                self.model = Signal2Pixel_Model(self.options.model)
            elif self.options.model.name == "signal2pixel_4views":
                self.model = Signal2Pixel_4views_Model(self.options.model)
            elif self.options.model.name == 'signal2pixel_4views_out4':
                self.model = Signal2Pixel_4views_Model(self.options.model)
                if self.options.model.finetune:
                    self.model.encoder.requires_grad_(False)
                    print('frozen')
            else:
                raise NotImplementedError("Your model is not found")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).cuda()
            if self.options.model.name == 'signal2point_GAN':
                self.Discriminator = torch.nn.DataParallel(self.Discriminator, device_ids=self.gpus).cuda()

        # Setup a joint optimizer for the 2 models
        if self.options.optim.name == "adam":
            self.optimizer = torch.optim.Adam(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif self.options.optim.name == "sgd":
            self.optimizer = torch.optim.SGD(
                params=list(self.model.parameters()),
                lr=self.options.optim.lr,
                momentum=self.options.optim.sgd_momentum,
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor
        )

        if self.options.model.name == "signal2point_GAN":
        # Setup a  optimizer for Discriminator
            if self.options.optim.Dname == "adam":
                self.D_optimizer = torch.optim.Adam(
                    params=list(self.Discriminator.parameters()),
                    lr=self.options.optim.Dlr,
                    betas=(self.options.optim.Dadam_beta1, 0.999),
                    weight_decay=self.options.optim.Dwd
                )
            elif self.options.optim.Dname == "sgd":
                self.D_optimizer = torch.optim.SGD(
                    params=list(self.Discriminator.parameters()),
                    lr=self.options.optim.Dlr,
                    momentum=self.options.optim.Dsgd_momentum,
                    weight_decay=self.options.optim.Dwd
                )
            else:
                raise NotImplementedError("Your optimizer is not found")
            self.D_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.D_optimizer, self.options.optim.Dlr_step, self.options.optim.Dlr_factor
            )

        # Create loss functions
        if self.options.model.name == "pixel2mesh":
            self.criterion = P2MLoss(self.options.loss, self.ellipsoid).cuda()
        elif self.options.model.name == "classifier":
            self.criterion = CrossEntropyLoss()
        elif self.options.model.name == "signal2point":
            self.criterion = S2PLoss(self.options.loss)
        elif self.options.model.name == "signal2point_4views":
            self.criterion = S2PLoss(self.options.loss)
        elif self.options.model.name == "signal2point_4views_transformer":
            self.criterion = S2PLoss(self.options.loss)
        elif self.options.model.name == "signal2point_4views_transformer_add":
            self.criterion = S2PLoss(self.options.loss)
        elif self.options.model.name == "signal2point_2views":
            self.criterion = S2PLoss(self.options.loss)
        elif self.options.model.name == "signal2point_GAN":
            self.criterion = S2PLoss(self.options.loss)
            self.adversarial_loss = torch.nn.BCELoss()
        elif self.options.model.name == 'point2signal':
            self.criterion = P2SLoss(self.options.loss)
        elif self.options.model.name == 'signal_refiner':
            self.criterion = SignalRefinerLoss(self.options.loss)
        elif self.options.model.name == "signal2pixel":
            self.criterion = Signal2PixelLoss(self.options.loss)
        elif self.options.model.name == "signal2pixel_4views":
            self.criterion = Signal2PixelLoss(self.options.loss)
        elif self.options.model.name == 'signal2pixel_4views_out4':
            self.criterion = Signal2PixelOut4Loss(self.options.loss)
        else:
            raise NotImplementedError("Your loss is not found")

        # Create AverageMeters for losses
        self.losses = AverageMeter()
        self.D_losses = AverageMeter()

        # Evaluators
        self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]

    def models_dict(self):
        ret = {'model': self.model}
        if self.options.model.name == "signal2point_GAN":
            ret['discriminator'] = self.Discriminator
        return ret

    def optimizers_dict(self):
        ret = {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}
        if self.options.model.name == "signal2point_GAN":
            ret['D_optimizer'] = self.D_optimizer
            ret['D_lr_scheduler'] = self.D_lr_scheduler
        return ret

    def train_step(self, input_batch):
        self.model.train()
        if self.options.model.name == "signal2point":
            # Grab data from the batch
            signal_real = input_batch["signal_real"]
            signal_imag = input_batch["signal_imag"]

            # predict with model
            out = self.model(signal_real,signal_imag)

        elif self.options.model.name == "signal2point_4views":

            out = self.model(input_batch["signals"])
        
        elif self.options.model.name == "signal2point_2views":

            out = self.model(input_batch["signals"])
        
        elif self.options.model.name == "signal2point_4views_transformer":

            out = self.model(input_batch["signals"])

        elif self.options.model.name == "signal2point_4views_transformer_add":

            out = self.model(input_batch["signals"])

        elif self.options.model.name == "signal2point_GAN":

            out = self.model(input_batch["signals"])

        elif self.options.model.name == 'point2signal':

            out = self.model(input_batch["points"], input_batch["start_bin"], input_batch["end_bin"])
        
        elif self.options.model.name == 'signal_refiner':

            out = self.model(input_batch['simulate_signal_abs'])

        elif self.options.model.name == "signal2pixel":

            out = self.model(input_batch["signals"][0])
        elif self.options.model.name == "signal2pixel_4views":
            signals = input_batch["signals"]
            out = []
            for i in range(4):
                out_i = self.model(signals)
                out.append({'out':out_i,'view_index':i})
                signals.append(signals.pop(0))
        elif self.options.model.name == 'signal2pixel_4views_out4':

            out = self.model(input_batch["signals"])
        # compute loss
        if isinstance(out, list):
            for out_i in out:
                loss, loss_summary = self.criterion(out_i, input_batch)
                self.losses.update(loss.detach().cpu().item())

                # Do backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        else:
            loss, loss_summary = self.criterion(out, input_batch)
            # if self.options.model.name == "signal2point_GAN":
            #     valid = Variable(Tensor(out['pred_coord'].size(0), 1).fill_(1.0), requires_grad=False).cuda()
            #     g_loss = self.adversarial_loss(self.Discriminator(out["pred_coord"]), valid)
            #     loss += g_loss * self.options.optim.G_adversarial_weigh
            #     loss_summary["g_loss"] = g_loss
            self.losses.update(loss.detach().cpu().item())

            # Do backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Discriminator train
        
        # if self.options.model.name == "signal2point_GAN":
        #     self.Discriminator.train()
        #     # Adversarial ground truths
        #     device = next(self.Discriminator.module.parameters()).device
        #     valid = Variable(Tensor(out['pred_coord'].size(0), 1).fill_(1.0), requires_grad=False).cuda()
        #     fake = Variable(Tensor(out['pred_coord'].size(0), 1).fill_(0.0), requires_grad=False).cuda()

        #     real_loss = self.adversarial_loss(self.Discriminator(input_batch["points"].detach()), valid)
        #     fake_loss = self.adversarial_loss(self.Discriminator(out["pred_coord"].detach()), fake)

        #     D_loss = (real_loss + fake_loss) / 2

        #     self.D_losses.update(D_loss.detach().cpu().item())
            
        #     # Do backprop
        #     self.D_optimizer.zero_grad()
        #     D_loss.backward()
        #     self.D_optimizer.step()
        #     loss_summary["D_loss"] = D_loss.detach().cpu().item()

        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def train(self):
        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1

            # Create a new data loader for every epoch
            train_data_loader = DataLoader(self.dataset,
                                           batch_size=self.options.train.batch_size * self.options.num_gpus,
                                           num_workers=self.options.num_workers,
                                           pin_memory=self.options.pin_memory,
                                           shuffle=self.options.train.shuffle,
                                           collate_fn=self.dataset_collate_fn)

            # Reset loss
            self.losses.reset()

            # Rest D_loss
            self.D_losses.reset()

            # self.test()

            # Iterate over all batches in an epoch
            for step, batch in enumerate(train_data_loader):
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                # if batch_size == 1 skip, because the model possibly contains bn layers 
                if any(isinstance(value, torch.Tensor) and value.size(0) == 1 for value in batch.values()):
                    continue

                # Run training step
                out = self.train_step(batch)

                self.step_count += 1

                # Tensorboard logging every summary_steps steps
                if self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)

                # Save checkpoint every checkpoint_steps steps
                if self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint()


            # save checkpoint after each epoch
            self.dump_checkpoint()

            # Run validation every test_epochs
            if self.epoch_count % self.options.train.test_epochs == 0:
                self.test()

            # lr scheduler step
            self.lr_scheduler.step()
            if self.options.model.name == "signal2point_GAN":
                self.D_lr_scheduler.step()

    def train_summaries(self, input_batch, out_summary, loss_summary):
        if self.displayer is not None:
            # Do visualization for the first 2 images of the batch
            # display_pt = self.displayer.s2p_batch_visualize(input_batch, out_summary)
            self.summary_writer.add_mesh(tag='train_gt_pc', vertices = input_batch['points'][0:1,:,:],global_step=self.step_count)
            self.summary_writer.add_mesh(tag='train_pred_pc', vertices = out_summary['pred_coord'][0:1,:,:],global_step=self.step_count)
            # self.summary_writer.add_image("points", display_pt, self.step_count)
            self.summary_writer.add_histogram("length_distribution", input_batch["length"].cpu().numpy(),
                                              self.step_count)
        if self.options.model.name == "signal2point":
            self.logger.debug(input_batch["filename"])
        elif self.options.model.name == "signal2point_4views":
        # Debug info for filenames
            if self.options.dataset.name == "shapenet_S2P_4_views":
                self.logger.debug(input_batch["dirname"])
            elif self.options.dataset.name == "shapenet_S2P_4views_pro":
                self.logger.debug(input_batch["filename"])

        # Save results in Tensorboard
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

        # Save results to log
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.9f (%.9f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                        self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))
        if self.options.model.name == "signal2point_GAN":
            self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, D_Loss %.9f (%.9f)" % (
                self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (
                        self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.D_losses.val, self.D_losses.avg))

    def test(self):
        for evaluator in self.evaluators:
            evaluator.evaluate()

if __name__ == "__main__":
    pass