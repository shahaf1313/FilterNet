from PIL import Image
import scipy.io as sio
import torch
import os.path
import numpy as np
from constants import palette, trainId2label, NUM_CLASSES
from torch.utils.tensorboard import SummaryWriter


class ImageAndLossSaver():
    def __init__(self, tb: SummaryWriter, loss_history_dir, checkpoint_dir, save_pics_every):
        self.tb = tb
        self.loss_history_dir = loss_history_dir
        self.save_pics_every = save_pics_every
        self.checkpoint_dir = checkpoint_dir
        self.discriminator_total_steps = 0
        self.generator_total_steps = 0
        self.Reset()

    def Reset(self):
        self.generator_loss_history = []
        self.discriminator_loss_history = []
        self.saved_images_counter = 0
        self.running_loss_discriminator = 0.0
        self.running_loss_generator = 0.0
        self.running_time = 0.0

    def WriteLossHistory(self, discriminator, epoch, loss):
        if discriminator:
            self.running_loss_discriminator += loss
            self.discriminator_total_steps += 1
            self.discriminator_loss_history.append(loss)
            # self.tb.add_scalar('Discriminator Loss Epoch %d' % (epoch + 1), loss, len(self.discriminator_loss_history))
            self.tb.add_scalar('Discriminator Loss Full History', loss, self.discriminator_total_steps)
        else:
            self.running_loss_generator += loss
            self.generator_total_steps += 1
            self.generator_loss_history.append(loss)
            # self.tb.add_scalar('Generator Loss Epoch %d' % (epoch + 1), loss, len(self.generator_loss_history))
            self.tb.add_scalar('Generator Loss Full History', loss, self.generator_total_steps)

    @property
    def SaveImagesIteration(self):
        return len(self.generator_loss_history) % self.save_pics_every == 0

    def SaveTrainImages(self, epoch, src_img, src_in_trg, src_lbl, src_in_trg_label):
        src_img = (src_img+1.)/2.
        src_in_trg = (src_in_trg+1.)/2.
        rgb_src_lbl = np.array(colorize_mask(src_lbl.cpu().numpy()).convert('RGB')).transpose((2,0,1))
        rgb_src_in_trg_label = np.array(colorize_mask(src_in_trg_label.detach().cpu().numpy()).convert('RGB')).transpose((2,0,1))
        self.tb.add_image('TrainEpoch%dSnapshot%d/SourceImage' % (epoch+1, self.saved_images_counter+1), src_img)
        self.tb.add_image('TrainEpoch%ddSnapshot%d/SourceInTargetImage' % (epoch+1, self.saved_images_counter+1), src_in_trg)
        self.tb.add_image('TrainEpoch%ddSnapshot%d/SourceLabel' % (epoch+1, self.saved_images_counter+1), rgb_src_lbl)
        self.tb.add_image('TrainEpoch%ddSnapshot%d/SourceInTargetLabel' % (epoch+1, self.saved_images_counter+1), rgb_src_in_trg_label)
        self.saved_images_counter += 1

    def SaveValidationImages(self, epoch, trg_img, trg_lbl, pred_label):
        trg_img = (trg_img + 1.) / 2.
        rgb_trg_lbl = np.array(colorize_mask(trg_lbl.cpu().numpy()).convert('RGB')).transpose((2, 0, 1))
        rgb_pred_label = np.array(colorize_mask(pred_label.cpu().numpy()).convert('RGB')).transpose((2, 0, 1))
        self.tb.add_image('ValidationEpoch%d/TargetImage' % (epoch+1), trg_img)
        self.tb.add_image('ValidationEpoch%d/TargetTrueLabel' % (epoch+1), rgb_trg_lbl)
        self.tb.add_image('ValidationEpoch%d/TargetPredictedLabel' % (epoch+1), rgb_pred_label)

    def SaveEpochAccuracy(self, iou, miou, epoch):
        for i in range(NUM_CLASSES):
            self.tb.add_scalar('Accuracy/%s class accuracy' % (trainId2label[i].name), iou[i], epoch + 1)
        self.tb.add_scalar('Accuracy/Accuracy History [mIoU]', miou, epoch + 1)

    def SaveModelsCheckpoint(self, semseg_net, discriminator_net, generator_net, epoch, batch):
        semseg_path = os.path.join(self.checkpoint_dir, 'semseg_net_epoch_%d_batch_%d.pth' % (epoch + 1, batch + 1))
        discriminator_path = os.path.join(self.checkpoint_dir, 'discriminator_net_epoch_%d_batch_%d.pth' % (epoch + 1, batch + 1))
        generator_path = os.path.join(self.checkpoint_dir, 'generator_net_epoch_%d_batch_%d.pth' % (epoch + 1, batch + 1))
        torch.save(semseg_net.state_dict(), semseg_path)
        torch.save(discriminator_net.state_dict(), discriminator_path)
        torch.save(generator_net.state_dict(), generator_path)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def nanmean_torch(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num

def confusion_matrix_torch(y_pred, y_true, num_classes):
    N = num_classes
    y = (N * y_true + y_pred).type(torch.long)
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat((y, torch.zeros(N * N - len(y), dtype=torch.long).cuda()))
    y = y.reshape(N, N)
    return y

def compute_cm_batch_torch(y_pred, y_true, ignore_label, classes):
    batch_size = y_pred.shape[0]
    confusion_matrix = torch.zeros((classes, classes)).cuda()
    for i in range(batch_size):
        y_pred_curr = y_pred[i, :, :]
        y_true_curr = y_true[i, :, :]
        inds_to_calc = y_true_curr != ignore_label
        y_pred_curr = y_pred_curr[inds_to_calc]
        y_true_curr = y_true_curr[inds_to_calc]
        assert y_pred_curr.shape == y_true_curr.shape
        confusion_matrix += confusion_matrix_torch(y_pred_curr, y_true_curr, classes)
    return confusion_matrix


def compute_iou_torch(confusion_matrix):
    intersection = torch.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(dim=1)
    predicted_set = confusion_matrix.sum(dim=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / union.type(torch.float32)
    miou = nanmean_torch(iou)
    return iou, miou
