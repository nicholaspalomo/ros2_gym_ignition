import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
import cv2
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

class CNNSimple(nn.Module):
    """
    Simple CNN to compress the depth image observation
    TODO: Assumes that the input is a square image. Crop accordingly the input if aspect ratio not 1:1
    """
    def __init__(self,
                 filter_sizes,
                 kernel_sizes,
                 mlp_layers,
                 strides,
                 padding_sizes,
                 activation_fn,
                 w_loss=0.5,
                 num_input_features=1,
                 num_output_features=1,
                 input_shape=256,
                 embedded_shape=32,
                 output_shape=64,
                 device='cpu',
                 logger=None):
        super().__init__()
        self.w_loss = w_loss
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.embed_shape = embedded_shape
        self.device = device
        self.logger = logger

        self.conv1 = nn.Conv2d(
            num_input_features, filter_sizes[0], 
            kernel_sizes[0], 
            stride=strides[0], 
            padding=padding_sizes[0])

        self.conv2 = nn.Conv2d(
            filter_sizes[0], filter_sizes[1], 
            kernel_sizes[1], 
            stride=strides[1], 
            padding=padding_sizes[1])

        self.conv3 = nn.Conv2d(
            filter_sizes[1], filter_sizes[2], 
            kernel_sizes[2], 
            stride=strides[2], 
            padding=padding_sizes[2])

        self.convt1 = nn.ConvTranspose2d(
            filter_sizes[2], filter_sizes[3], 
            kernel_sizes[3], 
            stride=strides[3], 
            padding=padding_sizes[3], 
            output_padding=1)

        self.convt2 = nn.ConvTranspose2d(
            filter_sizes[3], filter_sizes[4], 
            kernel_sizes[4], 
            stride=strides[4], 
            padding=padding_sizes[4], 
            output_padding=1)

        self.convt3 = nn.ConvTranspose2d(
            filter_sizes[4], num_input_features, 
            kernel_sizes[5], 
            stride=strides[5], 
            padding=padding_sizes[5], 
            output_padding=1)

        self.mlp1 = nn.Linear(embedded_shape**2, mlp_layers[0], bias=False)
        self.mlp2 = nn.Linear(mlp_layers[0], mlp_layers[1], bias=False)
        self.mlp3 = nn.Linear(mlp_layers[1], output_shape, bias=False)

        self.conv_embed = nn.Conv2d(
            filter_sizes[2], num_output_features, 
            kernel_sizes[2], 
            stride=1, 
            padding=1)

        self.loss_semseg = nn.CrossEntropyLoss()
        self.loss_depth = nn.MSELoss()
        self.activation_fn = activation_fn()

        # initialize the layer weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def encoder_parameters(self):

        return [*self.conv1.parameters(),
                *self.conv2.parameters(),
                *self.conv3.parameters(),
                *self.conv_embed.parameters(), # 1x1 convolution to flatten feature layers
                *self.mlp1.parameters(),
                *self.mlp2.parameters(),
                *self.mlp3.parameters()]

    def conv2d_output_size(self, input_shape, padding, stride, kernel):

        return math.floor((input_shape - kernel + 2*padding) / stride + 1)

    def convt2d_output_size(self, input_shape, padding_in, stride, kernel, padding_out):

        return math.floor((input_shape - 1) * stride + kernel - 2 * padding_in + padding_out)

    def output_dim(self, dummy_input):

        with torch.no_grad():
            dummy_output = self.forward(dummy_input.squeeze())

            self.__output_dim = dummy_output.size()[0] * dummy_output.size()[1]

        return self.__output_dim

    def embed(self, x):

        return self.conv_embed(self.activation_fn(x))

    def encode(self, x, inpaint=True, normalize=True):

        if len(x.shape) > 2:
            for i in range(x.shape[0]):
                if inpaint:
                    if torch.is_tensor(x):
                        x = x.detach().cpu.numpy()
                    x[i, :, :] = self.inpaint(x[i, :, :])
                if normalize:
                    if torch.is_tensor(x):
                        x = x.detach().cpu.numpy()
                    x[i, :, :] = self.normalize(x[i, :, :])
        else:
            if inpaint:
                if torch.is_tensor(x):
                    x = x.detach().cpu.numpy()
                x = self.inpaint(x)
            if normalize:
                if torch.is_tensor(x):
                    x = x.detach().cpu.numpy()
                x = self.normalize(x)

        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(self.device)
            if len(x.shape) == 2:
                x = x.unsqueeze(0).unsqueeze(0)

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Network expects an input of dimension (batch size, input channels, height, width). So we unsqueeze dim=1.
        x = self.activation_fn(self.conv1(x))
        x = self.activation_fn(self.conv2(x))
        x = self.conv3(x)

        # return the compressed image representation
        return x

    def flatten(self, x):

        # pass the flattened output through an MLP
        x = torch.flatten(self.activation_fn(x), 1)
        x = self.activation_fn(self.mlp1(x))
        x = self.activation_fn(self.mlp2(x))
        x = self.mlp3(x)

        return x

    def forward(self, x, inpaint=True, normalize=True):

        x = self.encode(x, inpaint=inpaint, normalize=normalize)
        x = self.embed(x)
        x = self.flatten(x)

        return x

    def decode(self, x):

        # Decode the compressed image
        x = self.activation_fn(x)
        x = self.activation_fn(self.convt1(x))
        x = self.activation_fn(self.convt2(x))
        x = self.convt3(x)

        return x

    def compute_mse_loss(self, x, depth_img, segmentation_mask=None, grayscale_img=None):
        
        # For now, make the training loss the difference between the original depth image and the reconstructed depth image
        loss = self.w_loss * self.loss_depth(x, depth_img)

        return loss

    def inpaint(self, x, missing_value=float('inf')):
        """
        Inpaint the missing values in the depth image.
        :param missing_value: the value that should be filled in the depth image.
        """

        if np.abs(x[x != missing_value]).max() == 0:
            return np.zeros(x.shape).astype(np.float32)

        x_not_inf_indices = np.where(x != missing_value)
        
        scale = np.abs(x[x_not_inf_indices[0], x_not_inf_indices[1]]).max()
        x = x.astype(np.float32) / scale # Should be float32; 64 not supported

        interpolator = LinearNDInterpolator(
            points=Delaunay(np.stack([x_not_inf_indices[0], x_not_inf_indices[1]], axis=1).astype(np.float32)),
            # points=np.stack([x_not_inf_indices[0], x_not_inf_indices[1]], axis=1),
            values=x[x_not_inf_indices[0], x_not_inf_indices[1]],
            fill_value=0)
        query_row_idx, query_col_idx = np.meshgrid(np.arange(x.shape[0]), np.arange(x.shape[1]), indexing='ij')
        query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
        x = interpolator(query_coord).reshape([x.shape[0], x.shape[1]])
        x *= scale

        # # cv2 inpainting doesn't handle the border properly
        # # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        # x = cv2.copyMakeBorder(x, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        # mask =  (x == missing_value).astype(np.uint8)

        # # Scale to keep as float. Maintain between  [-1, 1]
        # scale = np.abs(x[x != missing_value]).max()

        # x = x.astype(np.float32) / scale # Should be float32; 64 not supported
        # x[x == missing_value] = 0.0 # a hack, otherwise it seems like inpaint returns NaNs for large patches
        # x = cv2.inpaint(x, mask, 1, cv2.INPAINT_NS)

        # # Back to the original size and value range
        # x = x[1:-1, 1:-1]
        # x *= scale

        return x

    def normalize(self, x):

        return (x - x.mean()) / (x.std() + 1e-5)

    def rgb_2_grayscale(self, x):
        """
        Convert an RGB image to a grayscale image
        :param x: RGB image
        :returns x: converted grayscale image
        """

        return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

    def save_images(self, depth=None, rgb=None):

        if depth is not None:
            image = (255 * depth).astype(np.uint8)
            cv2.imwrite('depth_uninterp.png', image)

            depth_interp = self.inpaint(depth)
            image = (255 * depth_interp).astype(np.uint8)
            cv2.imwrite('depth_interp.png', image)

        if rgb is not None:
            cv2.imwrite('rgb.png', rgb)

            grayscale = self.rgb_2_grayscale(rgb)
            cv2.imwrite('grayscale.png', grayscale)

    def save_intermediate_grayscale_images(self, x_raw, suffix='', input_is_grayscale=True):
        
        if not input_is_grayscale:
            grayscale = self.rgb_2_grayscale(x_raw)
        else:
            grayscale = x_raw
        cv2.imwrite('grayscale' + suffix + '.png', (255 * grayscale).astype(np.uint8))

        embedded_rgb_img = self.encode(grayscale.astype(np.float32), normalize=False, inpaint=False)
        embedded_rgb_img = self.embed(embedded_rgb_img)
        image = (255 * embedded_rgb_img.squeeze().squeeze().detach().cpu().numpy()).astype(np.uint8)
        cv2.imwrite('grayscale_compressed' + suffix + '.png', image.astype(np.uint8))

    def save_indermediate_depth_images(self, x_raw, suffix=''):

        image = (255 * x_raw).astype(np.uint8)
        cv2.imwrite('depth_raw' + suffix + '.png', image)

        x_interp = self.inpaint(x_raw)
        x_interp = self.normalize(x_interp)
        image = (255 * x_interp).astype(np.uint8)
        cv2.imwrite('depth_interp' + suffix + '.png', image)
        
        embedded_depth_img8 = self.encode(x_interp.astype(np.float32), normalize=False, inpaint=False)
        embedded_depth_img1 = self.embed(embedded_depth_img8)
        image = (255 * embedded_depth_img1.squeeze().squeeze().detach().cpu().numpy()).astype(np.uint8)
        cv2.imwrite('depth_compressed' + suffix + '.png', image)

        decoded_depth_img = self.decode(embedded_depth_img8).squeeze().squeeze()
        image = np.reshape(255 * decoded_depth_img.detach().cpu().numpy(), (self.input_shape, self.input_shape), order='C').astype(np.uint8)
        cv2.imwrite('depth_reconstruct' + suffix + '.png', image)