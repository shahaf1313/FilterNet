from PIL import Image
import scipy.io as sio
import numpy as np
import os.path
from torch.utils.tensorboard import SummaryWriter
import torch

class ImageAndLossSaver():
    def __init__(self, tb: SummaryWriter, loss_history_dir, save_pics_every):
        self.tb = tb
        self.loss_history_dir = loss_history_dir
        self.save_pics_every = save_pics_every
        self.Reset()

    def WriteLossHistory(self, discriminator, epoch, loss):
        if discriminator:
            self.running_loss_discriminator += loss
            self.discriminator_loss_history.append(loss)
            self.tb.add_scalar('Discriminator Loss Epoch %d' % (epoch + 1), loss, len(self.discriminator_loss_history))
        else:
            self.running_loss_generator += loss
            self.generator_loss_history.append(loss)
            self.tb.add_scalar('Generator Loss Epoch %d' % (epoch + 1), loss, len(self.generator_loss_history))

    @property
    def SaveIteration(self):
        return len(self.generator_loss_history) % self.save_pics_every == 0

    def SaveImages(self, epoch, batch_num, src_img, src_in_trg, src_lbl, src_in_trg_label):
        src_img = (src_img+1.)/2.
        src_in_trg = (src_in_trg+1.)/2.
        rgb_src_lbl = np.array(colorize_mask(src_lbl.cpu().numpy()).convert('RGB')).transpose((2,0,1))
        rgb_src_in_trg_label = np.array(colorize_mask(src_in_trg_label.detach().cpu().numpy()).convert('RGB')).transpose((2,0,1))
        self.tb.add_image('Epoch%dBatch%d/SourceImage' % (epoch+1, batch_num+1), src_img)
        self.tb.add_image('Epoch%dBatch%d/SourceInTargetImage' % (epoch+1, batch_num+1), src_in_trg)
        self.tb.add_image('Epoch%dBatch%d/SourceLabel' % (epoch+1, batch_num+1), rgb_src_lbl)
        self.tb.add_image('Epoch%dBatch%d/SourceInTargetLabel' % (epoch+1, batch_num+1), rgb_src_in_trg_label)

    def Reset(self):
        self.generator_loss_history = []
        self.discriminator_loss_history = []
        self.running_loss_discriminator = 0.0
        self.running_loss_generator = 0.0
        self.running_time = 0.0

    def SaveLossHistory(self, loss_history_folder, epoch):
        numpy_discriminator_loss = np.array(self.discriminator_loss_history, dtype=np.float32)
        numpy_generator_loss = np.array(self.generator_loss_history, dtype=np.float32)
        loss_history_mat = os.path.join(loss_history_folder, 'loss_history_epoch_%d.mat' % epoch)
        sio.savemat(loss_history_mat, {('discriminator_loss_history_epoch_%d' % epoch): numpy_discriminator_loss,
                                       ('generator_loss_history_epoch_%d' % epoch): numpy_generator_loss})

def SetLearnableModels(learnable: list, non_learnable: list):
    for model in non_learnable:
        model.training=False
    for model in learnable:
        model.training = True

# color coding of semantic classes
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

