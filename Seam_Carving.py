import numpy as np
import cv2
import argparse  # argparse go on the internet
from numba import (
    jit,
)  # translate python numpy code into fast machine code https://numba.pydata.org/
from scipy import ndimage as ndi  # https://scipy.org/

SEAM_columnOR = np.array([255, 200, 200])  # seam visualization columnor (BGR)

ENERGY_MASK_CONST = 100000.0  # large energy value for protective masking
MASK_THRESHOLD = 10  # minimum pixel intensity for binary mask

######## UTILITY #############
def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_columnOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

def visualizeGradient(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_columnOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization Gradient", vis)
    cv2.waitKey(1)
    return vis


def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)


def rotate_image(image, counter_clockwise):  # k =1 => rotate Counter-Clockwise
    k = 1 if counter_clockwise else 3
    return np.rot90(image, k)

#### Energy function:

### Vectorized version use np.roll:
def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.

    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving. Check the github repo for more info
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
    
    # toggle to visualize gradient magnitude
    vis = visualizeGradient(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)     
        
    return energy


@jit
#Calculate on real image
def add_seam(im, seam_idx):
    """
    Add a vertical seam to a 3-channel columnor image at the indices provided 
    by averaging the pixels values to the left and right of the seam.

    Code adapted from https://github.com/vivianhylee/seam-carving.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        column = seam_idx[row]
        for ch in range(3):
            if column == 0:
                p = np.average(im[row, column: column + 2, ch])
                output[row, column, ch] = im[row, column, ch]
                output[row, column + 1, ch] = p
                output[row, column + 1:, ch] = im[row, column:, ch]
            else:
                p = np.average(im[row, column - 1: column + 1, ch])
                output[row, : column, ch] = im[row, : column, ch]
                output[row, column, ch] = p
                output[row, column + 1:, ch] = im[row, column:, ch]

    return output

@jit
# Calculate on gradient magnitude image
def add_seam_grayscale(im, seam_idx):
    """
    Add a vertical seam to a grayscale image at the indices provided 
    by averaging the pixels values to the left and right of the seam.
    """    
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        column = seam_idx[row]
        if column == 0:
            p = np.average(im[row, column: column + 2])
            output[row, column] = im[row, column]
            output[row, column + 1] = p
            output[row, column + 1:] = im[row, column:]
        else:
            p = np.average(im[row, column - 1: column + 1])
            output[row, : column] = im[row, : column]
            output[row, column] = p
            output[row, column + 1:] = im[row, column:]

    return output

@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))

@jit
def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))

@jit
def get_minimum_seam(im, mask=None, remove_mask=None):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    energyfn = forward_energy
    M = energyfn(im)

    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST

    # give removal mask priority over protective mask by using larger negative value
    if remove_mask is not None:
        M[np.where(remove_mask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask

########################################
# MAIN ALGORITHM
######################################## 

def seams_removal(im, num_remove, mask=None, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, mask)
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask


def seams_insertion(im, num_add, mask=None, vis=False, rot=False):
    seams_record = []
    temp_im = im.copy()
    temp_mask = mask.copy() if mask is not None else None

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, temp_mask)
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if vis:
            visualize(im, rotate=rot)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         

    return im, mask

########################################
# MAIN FUNCTIONS
########################################

def object_removal(im, rmask, mask=None, vis=False, horizontal_removal=False):
    im = im.astype(np.float64)
    rmask = rmask.astype(np.float64)
    if mask is not None:
        mask = mask.astype(np.float64)
    output = im

    h, w = im.shape[:2]

    if horizontal_removal:
        output = rotate_image(output, True)
        rmask = rotate_image(rmask, True)
        if mask is not None:
            mask = rotate_image(mask, True)

    while len(np.where(rmask > MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(output, mask, rmask)
        if vis:
            visualize(output, boolmask, rotate=horizontal_removal)            
        output = remove_seam(output, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)

    num_add = (h if horizontal_removal else w) - output.shape[1]
    output, mask = seams_insertion(output, num_add, mask, vis, rot=horizontal_removal)
    if horizontal_removal:
        output = rotate_image(output, False)

    return output        

#REMOVE OBJECT
img=cv2.imread('demos/train.jpg')
rmask = cv2.imread('demos/train_mask.jpg',0)
#img=cv2.imread('demos/AvengerOrigin.jpg')
#rmask = cv2.imread('demos/AvengersMask.jpg',0)

#img=cv2.imread('demos/Cast.jpg')
imageResize=resize(img,400)
#mask = cv2.imread('demos/CastMask.jpg',0)
#maskResize=resize(mask,400)
#USING PROTECTIVE MASK
#seams_removal(imageResize, 200, maskResize, vis=True, rot=False)

#resize image for faster carving
#imageResize=resize(img,400)
rmaskResize=resize(rmask,400)

#cv2.imshow("demos/lake_falls.jpg", backward_energy(img))
RemoveObject = object_removal(imageResize,rmaskResize,mask=None, vis=True, horizontal_removal=True)
#cv2.imshow('resize',RemoveObject)
cv2.waitKey(0)
#print(backward_energy(a))