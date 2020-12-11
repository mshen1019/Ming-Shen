import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import cv2

def load_data(test=False):
    """
    if test is true, load test set, if test is false(default), load training set
    """
    FTRAIN = './data/training.csv'
    FTEST = './data/test.csv'
    fname = FTEST if test else FTRAIN
    df = read_csv(fname)

    # Transform image term into array (it is seperated by space in da)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # drop out NA
    df = df.dropna()  

    # standardization
    X = np.vstack(df['Image'].values) / 255.  
    X = X.astype(np.float32)

    # reshape to 96*96*1
    X = X.reshape(-1, 96, 96, 1) 

    # only FTRAIN contain key point coordinate (target value)
    if not test:  
        y = df[df.columns[:-1]].values
        # normalize
        y = (y - 48) / 48  
        # shuffle
        X, y = shuffle(X, y, random_state=42)  
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def transparentOverlay(src , overlay , pos=(0,0), scale = 1):
    """
    Try to overlay the original picture with one extra channel
    :param src: background pic
    :param overlay: pic with extra overlay channel
    :param pos: the position of overlay
    :param scale : overlay scale
    :return: return Image
    """
    if scale != 1:
        overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)

    # overlay
    h,w,_ = overlay.shape  
    # 
    y,x = pos[0],pos[1] 

    alpha = overlay[:,:,3]/255.0
    alpha = alpha[..., np.newaxis]
    src[x:x+h,y:y+w,:] = alpha * overlay[:,:,:3] + (1-alpha)*src[x:x+h,y:y+w,:]
    return src