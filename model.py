import unet
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
print(torch.__version__)
'''Tested with Pytorch 0.3.1, Python 3.5'''


class BinaryLayer(nn.Module):

    def forward(self, x):
        """Forward function for binary layer

        :param x: data
        :returns: sign of data
        :rtype: Tensor

        """
        return torch.sign(x)

    def backward(self, grad_output):
        """Straight through estimator

        :param grad_output: gradient tensor
        :returns: truncated gradient tensor
        :rtype: Tensor

        """
        input = self.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class CubicFilter(nn.Module):

    def __init__(self, num_in_channels=67, num_out_channels=64):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CubicFilter, self).__init__()

        self.cubic_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.cubic_layer2 = MaxPoolBlock()
        self.cubic_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer4 = MaxPoolBlock()
        self.cubic_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer6 = MaxPoolBlock()
        self.cubic_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.cubic_layer8 = GlobalPoolingBlock(2)
        self.fc_cubic = torch.nn.Linear(
            num_out_channels, 60)  # cubic
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear')
        self.dropout = nn.Dropout(0.5)

    def get_cubic_mask(self, feat, img):
        """Cubic filter definition

        :param feat: feature map
        :param img:  image
        :returns: cubic scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Cubic #########################
        feat_cubic = torch.cat((feat, img), 1)
        feat_cubic = self.upsample(feat_cubic)

        x = self.cubic_layer1(feat_cubic)
        x = self.cubic_layer2(x)
        x = self.cubic_layer3(x)
        x = self.cubic_layer4(x)
        x = self.cubic_layer5(x)
        x = self.cubic_layer6(x)
        x = self.cubic_layer7(x)
        x = self.cubic_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        R = self.fc_cubic(x)

        cubic_mask = torch.zeros_like(img)

        x_axis = torch.true_divide(Variable(torch.arange(img.shape[2]).view(-1, 1).repeat(1, img.shape[3]).cuda()), img.shape[2])
        y_axis = torch.true_divide(Variable(torch.arange(img.shape[3]).repeat(img.shape[2], 1).cuda()), img.shape[3])

        '''
        Cubic for R channel
        '''
        cubic_mask[0, 0, :, :] = R[0, 0] * (x_axis ** 3) + R[0, 1] * (x_axis ** 2) * y_axis + R[0, 2] * (
            x_axis ** 2) * img[0, 0, :, :] + R[0, 3] * (x_axis ** 2) + R[0, 4] * x_axis * (y_axis ** 2) + R[
            0, 5] * x_axis * y_axis * img[0, 0, :, :] \
            + R[0, 6] * x_axis * y_axis + R[0, 7] * x_axis * (img[0, 0, :, :] ** 2) + R[
            0, 8] * x_axis * img[0, 0, :, :] + R[0, 9] * x_axis + R[0, 10] * (
            y_axis ** 3) + R[0, 11] * (y_axis ** 2) * img[0, 0, :, :] \
            + R[0, 12] * (y_axis ** 2) + R[0, 13] * y_axis * (img[0, 0, :, :] ** 2) + R[
            0, 14] * y_axis * img[0, 0, :, :] + R[0, 15] * y_axis + R[0, 16] * (
            img[0, 0, :, :] ** 3) + R[0, 17] * (img[0, 0, :, :] ** 2) \
            + R[0, 18] * \
            img[0, 0, :, :] + R[0, 19]

        '''
        Cubic for G channel
        '''
        cubic_mask[0, 1, :, :] = R[0, 20] * (x_axis ** 3) + R[0, 21] * (x_axis ** 2) * y_axis + R[0, 22] * (
            x_axis ** 2) * img[0, 1, :, :] + R[0, 23] * (x_axis ** 2) + R[0, 24] * x_axis * (y_axis ** 2) + R[
            0, 25] * x_axis * y_axis * img[0, 1, :, :] \
            + R[0, 26] * x_axis * y_axis + R[0, 27] * x_axis * (img[0, 1, :, :] ** 2) + R[
            0, 28] * x_axis * img[0, 1, :, :] + R[0, 29] * x_axis + R[0, 30] * (
            y_axis ** 3) + R[0, 31] * (y_axis ** 2) * img[0, 1, :, :] \
            + R[0, 32] * (y_axis ** 2) + R[0, 33] * y_axis * (img[0, 1, :, :] ** 2) + R[
            0, 34] * y_axis * img[0, 1, :, :] + R[0, 35] * y_axis + R[0, 36] * (
            img[0, 1, :, :] ** 3) + R[0, 37] * (img[0, 1, :, :] ** 2) \
            + R[0, 38] * \
            img[0, 1, :, :] + R[0, 39]

        '''
        Cubic for B channel
        '''
        cubic_mask[0, 2, :, :] = R[0, 40] * (x_axis ** 3) + R[0, 41] * (x_axis ** 2) * y_axis + R[0, 42] * (
            x_axis ** 2) * img[0, 2, :, :] + R[0, 43] * (x_axis ** 2) + R[0, 44] * x_axis * (y_axis ** 2) + R[
            0, 45] * x_axis * y_axis * img[0, 2, :, :] \
            + R[0, 46] * x_axis * y_axis + R[0, 47] * x_axis * (img[0, 2, :, :] ** 2) + R[
            0, 48] * x_axis * img[0, 2, :, :] + R[0, 49] * x_axis + R[0, 50] * (
            y_axis ** 3) + R[0, 51] * (y_axis ** 2) * img[0, 2, :, :] \
            + R[0, 52] * (y_axis ** 2) + R[0, 53] * y_axis * (img[0, 2, :, :] ** 2) + R[
            0, 54] * y_axis * img[0, 2, :, :] + R[0, 55] * y_axis + R[0, 56] * (
            img[0, 2, :, :] ** 3) + R[0, 57] * (img[0, 2, :, :] ** 2) \
            + R[0, 58] * \
            img[0, 2, :, :] + R[0, 59]

        img_cubic = torch.clamp(img + cubic_mask, 0, 1)
        return img_cubic


class GraduatedFilter(nn.Module):

    def __init__(self, num_in_channels=67, num_out_channels=64):
        """Initialisation function for the graduated filter

        :param num_in_channels:  x channels
        :param num_out_channels: output channels
        :returns: N/A
        :rtype: N/A

        """
        super(GraduatedFilter, self).__init__()

        self.graduated_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.graduated_layer2 = MaxPoolBlock()
        self.graduated_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer4 = MaxPoolBlock()
        self.graduated_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer6 = MaxPoolBlock()
        self.graduated_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.graduated_layer8 = GlobalPoolingBlock(2)
        self.fc_graduated = torch.nn.Linear(
            num_out_channels, 24)
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear')
        self.dropout = nn.Dropout(0.5)
        self.bin_layer = BinaryLayer()

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors

        :param cond: condition e.g. <
        :param x_1: Tensor 1
        :param x_2: Tensor 2
        :returns: Boolean comparison result
        :rtype: Tensor

        """
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_inverted_mask(self, factor, invert, d1, d2, max_scale, top_line):
        """ Inverts the graduated filter based on a learnt binary variable 

        :param factor: scale factor
        :param invert: binary indicator variable
        :param d1: distance between top and mid line
        :param d2: distannce between botto and mid line
        :param max_scale: maximum scaling factor possible
        :param top_line:  representation of top line
        :returns: inverted scaling mask
        :rtype: Tensor

        """
        if (invert == 1).all():

            if (factor >= 1).all():
                diff = ((factor-1))/2 + 1
                grad1 = (diff-factor)/d1
                grad2 = (1-diff)/d2
                mask_scale = torch.clamp(
                    factor+grad1*top_line+grad2*top_line, min=1, max=max_scale)
            else:
                diff = ((1-factor))/2 + factor
                grad1 = (diff-factor)/d1
                grad2 = (1-diff)/d2
                mask_scale = torch.clamp(
                    factor+grad1*top_line+grad2*top_line, min=0, max=1)
        else:

            if (factor >= 1).all():
                diff = ((factor-1))/2 + 1
                grad1 = (diff-factor)/d1
                grad2 = (factor-diff)/d2
                mask_scale = torch.clamp(
                    1+grad1*top_line+grad2*top_line, min=1, max=max_scale)
            else:
                diff = ((1-factor))/2 + factor
                grad1 = (diff-1)/d1
                grad2 = (factor-diff)/d2
                mask_scale = torch.clamp(
                    1+grad1*top_line+grad2*top_line, min=0, max=1)

        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)
        return mask_scale

    def get_graduated_mask(self, feat, img):
        """ Graduated filter definition

        :param feat: features
        :param img: image
        :returns: scaling map
        :rtype: Tensor

        """
        #######################################################
        ####################### Graduated #####################

        x_axis = torch.true_divide(Variable(torch.arange(img.shape[2]).view(-1, 1).repeat(1, img.shape[3]).cuda()), img.shape[2])
        y_axis = torch.true_divide(Variable(torch.arange(img.shape[3]).repeat(img.shape[2], 1).cuda()), img.shape[3])

        feat_graduated = torch.cat((feat, img), 1)
        feat_graduated = self.upsample(feat_graduated)

        x = self.graduated_layer1(feat_graduated)
        x = self.graduated_layer2(x)
        x = self.graduated_layer3(x)
        x = self.graduated_layer4(x)
        x = self.graduated_layer5(x)
        x = self.graduated_layer6(x)
        x = self.graduated_layer7(x)
        x = self.graduated_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_graduated(x)

        # Classification values (above or below the line)
        G[0, 21] = ((self.bin_layer(G[0, 21]))+1)/2
        G[0, 22] = ((self.bin_layer(G[0, 22]))+1)/2
        G[0, 23] = ((self.bin_layer(G[0, 23]))+1)/2

        slope1 = G[0, 0].clone()
        slope2 = G[0, 6].clone()
        slope3 = G[0, 12].clone()

        G[0, 1] = self.tanh01(G[0, 1]) + 1e-10
        G[0, 7] = self.tanh01(G[0, 7]) + 1e-10
        G[0, 13] = self.tanh01(G[0, 13]) + 1e-10

        G[0, 2] = torch.clamp(self.tanh01(G[0, 2]), G[0, 1].item(), 1.0)
        G[0, 8] = torch.clamp(self.tanh01(G[0, 8]), G[0, 7].item(), 1.0)
        G[0, 14] = torch.clamp(self.tanh01(G[0, 14]), G[0, 13].item(), 1.0)

        G[0, 18] = torch.clamp(self.tanh01(G[0, 18]), 0, G[0, 1].item())
        G[0, 19] = torch.clamp(self.tanh01(G[0, 19]), 0, G[0, 7].item())
        G[0, 20] = torch.clamp(self.tanh01(G[0, 20]), 0, G[0, 13].item())

        # Scales
        max_scale = 2
        min_scale = 0

        G[0, 3] = self.tanh01(G[0, 3]) * max_scale
        G[0, 4] = self.tanh01(G[0, 4]) * max_scale
        G[0, 5] = self.tanh01(G[0, 5]) * max_scale

        G[0, 9] = self.tanh01(G[0, 9]) * max_scale
        G[0, 10] = self.tanh01(G[0, 10]) * max_scale
        G[0, 11] = self.tanh01(G[0, 11]) * max_scale

        G[0, 15] = self.tanh01(G[0, 15]) * max_scale
        G[0, 16] = self.tanh01(G[0, 16]) * max_scale
        G[0, 17] = self.tanh01(G[0, 17]) * max_scale

        slope1_angle = torch.atan(slope1)
        slope2_angle = torch.atan(slope2)
        slope3_angle = torch.atan(slope3)

        d1 = self.tanh01(G[0, 2]*torch.cos(slope1_angle))
        d2 = self.tanh01(G[0, 18]*torch.cos(slope1_angle))
        d3 = self.tanh01(G[0, 8]*torch.cos(slope2_angle))
        d4 = self.tanh01(G[0, 19]*torch.cos(slope2_angle))
        d5 = self.tanh01(G[0, 14]*torch.cos(slope3_angle))
        d6 = self.tanh01(G[0, 20]*torch.cos(slope3_angle))

        top_line1 = self.tanh01(y_axis - (slope1 * x_axis + G[0, 1] + d1))
        top_line2 = self.tanh01(y_axis - (slope2 * x_axis + G[0, 7] + d3))
        top_line3 = self.tanh01(y_axis - (slope3 * x_axis + G[0, 13] + d5))

        mask_scale1 = self.get_inverted_mask(
            G[0, 3], G[0, 21], d1, d2, max_scale, top_line1)
        mask_scale2 = self.get_inverted_mask(
            G[0, 4], G[0, 21], d1, d2, max_scale, top_line1)
        mask_scale3 = self.get_inverted_mask(
            G[0, 5], G[0, 21], d1, d2, max_scale, top_line1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1 = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        mask_scale4 = self.get_inverted_mask(
            G[0, 9], G[0, 22], d3, d4, max_scale, top_line2)
        mask_scale5 = self.get_inverted_mask(
            G[0, 10], G[0, 22], d3, d4, max_scale, top_line2)
        mask_scale6 = self.get_inverted_mask(
            G[0, 11], G[0, 22], d3, d4, max_scale, top_line2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4 = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        mask_scale7 = self.get_inverted_mask(
            G[0, 15], G[0, 23], d5, d6, max_scale, top_line3)
        mask_scale8 = self.get_inverted_mask(
            G[0, 16], G[0, 23], d5, d6, max_scale, top_line3)
        mask_scale9 = self.get_inverted_mask(
            G[0, 17], G[0, 23], d5, d6, max_scale, top_line3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7 = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        mask_scale = torch.clamp(
            mask_scale_1*mask_scale_4*mask_scale_7, 0, max_scale)

        return mask_scale


class EllipticalFilter(nn.Module):

    def __init__(self, num_in_channels=67, num_out_channels=64):
        """Initialisation function

        :param block: a block (layer) of the neural network
        :param num_layers:  number of neural network layers
        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(EllipticalFilter, self).__init__()

        self.elliptical_layer1 = ConvBlock(num_in_channels, num_out_channels)
        self.elliptical_layer2 = MaxPoolBlock()
        self.elliptical_layer3 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer4 = MaxPoolBlock()
        self.elliptical_layer5 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer6 = MaxPoolBlock()
        self.elliptical_layer7 = ConvBlock(num_out_channels, num_out_channels)
        self.elliptical_layer8 = GlobalPoolingBlock(2)
        self.fc_elliptical = torch.nn.Linear(
            num_out_channels, 24)  # elliptical
        self.upsample = torch.nn.Upsample(size=(300, 300), mode='bilinear')
        self.dropout = nn.Dropout(0.5)

    def tanh01(self, x):
        """Adjust Tanh to return values between 0 and 1

        :param x: Tensor arbitrary range
        :returns: Tensor between 0 and 1
        :rtype: tensor

        """
        tanh = nn.Tanh()
        return 0.5 * (tanh(x) + 1)

    def where(self, cond, x_1, x_2):
        """Differentiable where function to compare two Tensors

        :param cond: condition e.g. <
        :param x_1: Tensor 1
        :param x_2: Tensor 2
        :returns: Boolean comparison result
        :rtype: Tensor

        """
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def get_mask(self, x_axis, y_axis, shift_x=0, shift_y=0, semi_axis_x=1, semi_axis_y=1, alpha=0,
                 scale_factor=2, max_scale=2, eps=1e-8, radius=1):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :returns: scaling mask
        :rtype: Tensor

        """
        mask_scale = self.where((((((x_axis - shift_x)*torch.cos(alpha) + (y_axis - shift_y)*torch.sin(alpha)) ** 2) / ((semi_axis_x)**2)) + ((((x_axis - shift_x)*torch.sin(alpha) - (y_axis - shift_y)*torch.cos(alpha)) ** 2) / ((semi_axis_y)**2)) + eps) < 1,
                                (torch.sqrt((x_axis - shift_x) ** 2 + (y_axis - shift_y) ** 2 + eps) * (1 - scale_factor)) / radius + scale_factor, 1)
        mask_scale = torch.clamp(mask_scale.unsqueeze(0), 0, max_scale)

        return mask_scale

    def get_elliptical_mask(self, feat, img):
        """Gets the elliptical scaling mask according to the equation of a
        rotated ellipse

        :param feat: features from the backbone
        :param img: image
        :returns: elliptical adjustment maps for each channel
        :rtype: Tensor

        """
        eps = 1e-10
        max_scale = 2
        min_scale = 0

        feat_elliptical = torch.cat((feat, img), 1)
        feat_elliptical = self.upsample(feat_elliptical)

        x = self.elliptical_layer1(feat_elliptical)
        x = self.elliptical_layer2(x)
        x = self.elliptical_layer3(x)
        x = self.elliptical_layer4(x)
        x = self.elliptical_layer5(x)
        x = self.elliptical_layer6(x)
        x = self.elliptical_layer7(x)
        x = self.elliptical_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        G = self.fc_elliptical(x)

        x_axis = torch.true_divide(Variable(torch.arange(img.shape[2]).view(-1, 1).repeat(1, img.shape[3]).cuda()), img.shape[2])
        y_axis = torch.true_divide(Variable(torch.arange(img.shape[3]).repeat(img.shape[2], 1).cuda()), img.shape[3])

        # x coordinate - h position
        right_x = (img.shape[2] - 1) / img.shape[2]
        left_x = 0

        G[0, 0] = self.tanh01(G[0, 0]) + eps
        G[0, 8] = self.tanh01(G[0, 8]) + eps
        G[0, 16] = self.tanh01(G[0, 16]) + eps

        # y coordinate - k coordinate
        right_y = (img.shape[3] - 1) // img.shape[3]
        left_y = 0

        G[0, 1] = self.tanh01(G[0, 1]) + eps
        G[0, 9] = self.tanh01(G[0, 9]) + eps
        G[0, 17] = self.tanh01(G[0, 17]) + eps

        # a value
        G[0, 2] = self.tanh01(G[0, 2]) + eps
        G[0, 10] = self.tanh01(G[0, 10]) + eps
        G[0, 18] = self.tanh01(G[0, 18]) + eps

        # b value
        G[0, 3] = self.tanh01(G[0, 3]) + eps  # * (right_x - left_x) + left_x
        G[0, 11] = self.tanh01(G[0, 11]) + eps
        G[0, 19] = self.tanh01(G[0, 19]) + eps

        # A value
        G[0, 4] = self.tanh01(G[0, 4]) * math.pi + eps
        G[0, 12] = self.tanh01(G[0, 12]) * math.pi + eps
        G[0, 20] = self.tanh01(G[0, 20]) * math.pi + eps

        '''
        The following are the scale factors for each ellipse
        '''
        G[0, 5] = self.tanh01(G[0, 5]) * max_scale + eps
        G[0, 6] = self.tanh01(G[0, 6]) * max_scale + eps
        G[0, 7] = self.tanh01(G[0, 7]) * max_scale + eps

        G[0, 13] = self.tanh01(G[0, 13]) * max_scale + eps
        G[0, 14] = self.tanh01(G[0, 14]) * max_scale + eps
        G[0, 15] = self.tanh01(G[0, 15]) * max_scale + eps

        G[0, 21] = self.tanh01(G[0, 21]) * max_scale + eps
        G[0, 22] = self.tanh01(G[0, 22]) * max_scale + eps
        G[0, 23] = self.tanh01(G[0, 23]) * max_scale + eps

        angle_1 = torch.acos(torch.clamp((y_axis-G[0, 1]) / (torch.sqrt(
            (x_axis-G[0, 0])**2 + (y_axis-G[0, 1])**2 + eps) + eps), -1+1e-7, 1-1e-7)) - G[0, 4]
        angle_2 = torch.acos(torch.clamp((y_axis-G[0, 9]) / (torch.sqrt((x_axis-G[0, 8]) ** 2 + (
            y_axis-G[0, 9]) ** 2 + eps) + eps), -1+1e-7, 1-1e-7)) - G[0, 12]
        angle_3 = torch.acos(torch.clamp((y_axis-G[0, 17]) / (torch.sqrt((x_axis-G[0, 16]) ** 2 + (
            y_axis-G[0, 17]) ** 2 + eps) + eps), -1+1e-7, 1-1e-7)) - G[0, 20]

        radius_1 = ((G[0, 2]*G[0, 3]) / (torch.sqrt((G[0, 2]**2)*(torch.sin(angle_1)
                                                                  ** 2) + (G[0, 3]**2)*(torch.cos(angle_1)**2) + eps) + eps)) + eps
        radius_2 = ((G[0, 10] * G[0, 11]) / (torch.sqrt((G[0, 10] ** 2) * (torch.sin(angle_2)
                                                                           ** 2) + (G[0, 11] ** 2) * (torch.cos(angle_2) ** 2) + eps) + eps)) + eps
        radius_3 = ((G[0, 18] * G[0, 19]) / (torch.sqrt((G[0, 18] ** 2) * (torch.sin(angle_3)
                                                                           ** 2) + (G[0, 19] ** 2) * (torch.cos(angle_3) ** 2) + eps) + eps)) + eps

        mask_scale1 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 0], shift_y=G[0, 1], semi_axis_x=G[0,
                                                                                    2], semi_axis_y=G[0, 3], alpha=G[0, 4], scale_factor=G[0, 5],
                                    radius=radius_1)

        mask_scale2 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 0], shift_y=G[0, 1], semi_axis_x=G[0,
                                                                                    2], semi_axis_y=G[0, 3], alpha=G[0, 4], scale_factor=G[0, 6],
                                    radius=radius_1)

        mask_scale3 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 0], shift_y=G[0, 1], semi_axis_x=G[0,
                                                                                    2], semi_axis_y=G[0, 3], alpha=G[0, 4], scale_factor=G[0, 7],
                                    radius=radius_1)

        mask_scale_1 = torch.cat(
            (mask_scale1, mask_scale2, mask_scale3), dim=0)
        mask_scale_1_rad = torch.clamp(mask_scale_1.unsqueeze(0), 0, max_scale)

        ############

        mask_scale4 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 8], shift_y=G[0, 9], semi_axis_x=G[0,
                                                                                    10], semi_axis_y=G[0, 11], alpha=G[0, 12], scale_factor=G[0, 13],
                                    radius=radius_2)

        mask_scale5 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 8], shift_y=G[0, 9], semi_axis_x=G[0,
                                                                                    10], semi_axis_y=G[0, 11], alpha=G[0, 12], scale_factor=G[0, 14],
                                    radius=radius_2)

        mask_scale6 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 8], shift_y=G[0, 9], semi_axis_x=G[0,
                                                                                    10], semi_axis_y=G[0, 11], alpha=G[0, 12], scale_factor=G[0, 15],
                                    radius=radius_2)

        mask_scale_4 = torch.cat(
            (mask_scale4, mask_scale5, mask_scale6), dim=0)
        mask_scale_4_rad = torch.clamp(mask_scale_4.unsqueeze(0), 0, max_scale)

        ############

        mask_scale7 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 16], shift_y=G[0, 17], semi_axis_x=G[0,
                                                                                      18], semi_axis_y=G[0, 19], alpha=G[0, 20], scale_factor=G[0, 21],
                                    radius=radius_3)

        mask_scale8 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 16], shift_y=G[0, 17], semi_axis_x=G[0,
                                                                                      18], semi_axis_y=G[0, 19], alpha=G[0, 20], scale_factor=G[0, 22],
                                    radius=radius_3)

        mask_scale9 = self.get_mask(x_axis, y_axis,
                                    shift_x=G[0, 16], shift_y=G[0, 17], semi_axis_x=G[0,
                                                                                      18], semi_axis_y=G[0, 19], alpha=G[0, 20], scale_factor=G[0, 23],
                                    radius=radius_3)

        mask_scale_7 = torch.cat(
            (mask_scale7, mask_scale8, mask_scale9), dim=0)
        mask_scale_7_rad = torch.clamp(mask_scale_7.unsqueeze(0), 0, max_scale)

        mask_scale_elliptical = torch.clamp(
            mask_scale_1_rad * mask_scale_4_rad * mask_scale_7_rad, 0, max_scale)

        return mask_scale_elliptical


class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of x channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the x BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function for the max pooling block

        :param x: Tensor representing the x BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


class DeepLPFParameterPrediction(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation function

        :param num_in_channels:  Number of x feature maps
        :param num_out_channels: Number of output feature maps
        :param batch_size: Size of image batch
        :returns: N/A
        :rtype: N/A

        """
        super(DeepLPFParameterPrediction, self).__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.cubic_filter = CubicFilter()
        self.graduated_filter = GraduatedFilter()
        self.elliptical_filter = EllipticalFilter()

    def forward(self, src_img, trg_feat):
        trg_feat.contiguous()  # remove memory holes
        torch.cuda.empty_cache()

        cubic_filter = self.cubic_filter.get_cubic_mask(trg_feat, src_img)
        mask_scale_graduated = self.graduated_filter.get_graduated_mask(trg_feat, src_img)
        mask_scale_elliptical = self.elliptical_filter.get_elliptical_mask(trg_feat, src_img)

        # mask_scale_fuse = torch.clamp(mask_scale_graduated+mask_scale_elliptical, 0, 2)
        # img_fuse = torch.clamp(cubic_filter * mask_scale_fuse, 0, 1)
        mask_scale_fuse = mask_scale_graduated+mask_scale_elliptical
        img_fuse = cubic_filter * mask_scale_fuse
        src_in_trg = torch.tanh(src_img + img_fuse)
        return src_in_trg


class DeepLPFNet(nn.Module):

    def __init__(self):
        super(DeepLPFNet, self).__init__()
        self.backbonenet = unet.UNetModel()
        self.deeplpfnet = DeepLPFParameterPrediction()

    def forward(self, src_img, trg_img):
        trg_feat = self.backbonenet(trg_img)
        src_in_trg = self.deeplpfnet(src_img, trg_feat)
        return src_in_trg



class Discriminator(nn.Module):
    def __init__(self, base_channels=64):
        super(Discriminator, self).__init__()

        self.l0 = nn.Sequential(
                                nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(base_channels),
                                nn.LeakyReLU()
        )
        self.l1 = nn.Sequential(
                                nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(base_channels),
                                nn.LeakyReLU()
        )
        self.l2 = nn.Sequential(
                                nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(base_channels),
                                nn.LeakyReLU()
        )
        self.l3 = nn.Sequential(
                                nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(base_channels),
                                nn.LeakyReLU()
        )
        self.l4 = nn.Sequential(
                                nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(base_channels),
                                nn.LeakyReLU()
        )
        self.l5 = nn.Sequential(
                                nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
                                nn.BatchNorm2d(base_channels),
                                nn.LeakyReLU()
        )
        self.l6 = nn.Sequential(
                                nn.Linear(in_features= 2**15, out_features=2**10),
                                # nn.BatchNorm1d(2**10),
                                nn.LeakyReLU()
        )
        self.l7 = nn.Sequential(
                                nn.Linear(in_features=2**10, out_features=2**6),
                                # nn.BatchNorm1d(2**6),
                                nn.Sigmoid()
        )


    def forward(self, x):
        x0 = self.l0(x)
        x1 = self.l1(x0)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4).view(x.shape[0], -1)
        x6 = self.l6(x5)
        x7 = self.l7(x6)

        return torch.mean(x7)

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        """
        Discriminator loss. Expects to receive scalar input.
        """
        super(DiscriminatorLoss, self).__init__()

    def forward(self, source_value, target_value):
        assert source_value.shape == target_value.shape
        assert len(source_value.shape) == 1
        self.cache = source_value, target_value, target_value.shape[0]
        return torch.mean(source_value**2 + (1-target_value)**2)

class GeneratorLoss(nn.Module):
    def __init__(self):
        """
        Generator loss. Expects to predicated
        """
        super(GeneratorLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, loss_seg, loss_ent, entW, discriminator_target):
        return torch.mean(loss_seg + entW*loss_ent + (1-discriminator_target)**2)
