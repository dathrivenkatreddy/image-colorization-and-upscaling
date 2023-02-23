import os
import os.path as osp
import sys
import glob
import numpy as np
import pandas as pd
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from io import BytesIO, StringIO
import streamlit as st


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


class Colorizer(object):

    def __init__(self):
        self.PT_PATH = 'models/colorization/colorization_deploy_v2.prototxt'
        self.MODEL_PATH = 'models/colorization/colorization_release_v2.caffemodel'
        self.KERNEL_PATH = 'models/colorization/pts_in_hull.npy'

        self.INPUT_WIDTH = 224
        self.INPUT_HEIGHT = 224
        self.IMSHOW_SIZE = (640, 480)

        self.net = cv.dnn.readNetFromCaffe(self.PT_PATH, self.MODEL_PATH)
        self.points = np.load(self.KERNEL_PATH)
        self.pts_in_hull = self.points.transpose().reshape(2, 313, 1, 1)

        self.net.getLayer(self.net.getLayerId('class8_ab')).blobs = [
            self.pts_in_hull.astype(np.float32)]
        self.net.getLayer(self.net.getLayerId('conv8_313_rh')).blobs = [
            np.full([1, 313], 2.606, np.float32)]

    def colorize(self, input_image, t_param):

        bw_image = cv.imread(input_image)
        rgb_image = (bw_image / 255.0).astype("float32")
        lab_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2Lab)
        imgL = lab_image[:, :, 0]

        (HEIGHT_ORIG, WIDTH_ORIG) = rgb_image.shape[:2]

        resized_image = cv.resize(
            rgb_image, (self.INPUT_WIDTH, self.INPUT_HEIGHT))

        resized_image_lab = cv.cvtColor(resized_image, cv.COLOR_RGB2Lab)
        resized_imgL = resized_image_lab[:, :, 0]
        resized_imgL -= t_param

        self.net.setInput(cv.dnn.blobFromImage(resized_imgL))
        ab_dec = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

        (HEIGHT_OUT, WIDTH_OUT) = ab_dec.shape[:2]
        ab_dec_us = cv.resize(ab_dec, (WIDTH_ORIG, HEIGHT_ORIG))
        img_lab_out = np.concatenate(
            (imgL[:, :, np.newaxis], ab_dec_us), axis=2)
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)


class Upscaler(object):
    def __init__(self):
        self.MODEL_PATH = 'models/upscaling/RRDB_ESRGAN_x4.pth'
        device = torch.device('cpu')

        model = RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(self.MODEL_PATH), strict=True)
        model.eval()
        model = model.to(device)


class FileUpload(object):

    def __init__(self):
        self.fileTypes = ["webp", "jpeg", "png", "jpg"]
        self.file = 0
        self.STYLE = """
                        <style>
                        img {
                            max-width: 100%;
                        }
                        </style>
                    """

    def run(self):
        st.markdown(self.STYLE, unsafe_allow_html=True)
        self.file = st.file_uploader("Upload file", type=self.fileTypes)
        show_file = st.empty()
        if not self.file:
            show_file.info("Upload image to get started ")
            return
        content = self.file.getvalue()
        if isinstance(self.file, BytesIO):
            show_file.image(self.file)
        else:
            pass
            # data = pd.read_csv(file)
            # st.dataframe(data.head(10))
        self.file.close()


class ColorUps(object):
    def __init__(self):
        self.show = st.empty()

    def showOptions(self):
        with st.sidebar.container():
            st.sidebar.header("Colorizer")
            c_enable = st.sidebar.checkbox('Enable')
            if c_enable:
                t_param = st.sidebar.slider(
                    "Tuning [default : 50]", 0, 255, value=50)
                st.sidebar.text(t_param)

        with st.sidebar.container():
            st.sidebar.header("Upscaler")
            u_enable = st.sidebar.checkbox('Upscale')
            if u_enable:
                u_param = st.sidebar.slider(
                    "Parameter [default : 50]", 0, 255, value=50)
                st.sidebar.text(u_param)


if __name__ == "__main__":
    fileuploader = FileUpload()
    fileuploader.run()
    if fileuploader.file:
        mainApp = ColorUps()
        mainApp.showOptions()
