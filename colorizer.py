import numpy as np
import cv2 as cv

PT_PATH = 'models/colorization/colorization_deploy_v2.prototxt'
MODEL_PATH = 'models/colorization/colorization_release_v2.caffemodel'
KERNEL_PATH = 'models/colorization/pts_in_hull.npy'

W_in = 224
H_in = 224
imshowSize = (640, 480)

image_path = 'images/test2.jpg'

def colorize(image_path, seed):
    net = cv.dnn.readNetFromCaffe(PT_PATH, MODEL_PATH)
    points = np.load(KERNEL_PATH)
    pts_in_hull = points.transpose().reshape(2, 313, 1, 1)

    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    bw_image = cv.imread(image_path)

    img_rgb = (bw_image / 255.0).astype("float32")

    img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)

    img_l = img_lab[:,:,0]

    (H_orig,W_orig) = img_rgb.shape[:2]

    img_rs = cv.resize(img_rgb, (W_in, H_in)) 
    img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
    img_l_rs = img_lab_rs[:,:,0]
    img_l_rs -= seed

    net.setInput(cv.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0,:,:,:].transpose((1,2,0)) 

    (H_out,W_out) = ab_dec.shape[:2]
    ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2)
    img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

    frame = cv.resize(bw_image, imshowSize)

# DEBUG

#cv.imshow('origin', bw_image)
#cv.imshow('origin', img_l)
#cv.imshow('gray', cv.cvtColor(bw_image, cv.COLOR_RGB2GRAY))
#cv.imshow('colorized', img_bgr_out)

#cv.imwrite('results/rlt.jpg', (img_bgr_out * 255.0).round())
#cv.waitKey(0)
#cv.destroyAllWindows()

if __name__ == '__main__':
    pass