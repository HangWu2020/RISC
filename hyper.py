# Training settings
batch_size = 4
batch_size_2 = 32
num_points = 2048
lr_ae = 1e-4
lr_fts = 5e-5
num_epoch = 50
use_bn = True


# Gridding settings
import utils
scale = 0.501
A, sigma = utils.get_gaussian_dist (x_m=0.005, y_m=0.5, f_0=1.0)

grid_pt = 64
grid_r = 64
grid_comp = [0.5, 0.5, 0.5]  # [theta, phai, r]
bandwidths = [grid_pt//2, grid_pt//2, grid_pt//2, grid_pt//2]


# Model settings
encoder_type = 's2cnn'
symfunc_g = 'maxpool'
n_primitives = 16
en_bn = True
de_bn = True

enc_features = {'encoder_v1_l1': [grid_r, 64, 64, 128],
                'encoder_v2_l0': [4, 32, 64],
                'encoder_v2_l1': [64, 64, 128],
                'encoder_conv3d': [128, 256, 512]}

dec_features = {'decoder_azimuth': [enc_features['encoder_conv3d'][-1], enc_features['encoder_conv3d'][-1]],
                'decoder_plgroup': [enc_features['encoder_conv3d'][-1]*2, 512, 128, num_points//n_primitives//grid_pt*3]}

inv_features = {'dense': [512, 512, 256, 128, 64],
                'shared': [512, 256, 128, 64, 1]}


'''
Deprecated code
'''
use_bn = True
decoder_type = 'atlas_res'
# enc_features = {'encoder_l1': [grid_r, 64, 64, 128],
#                 'encoder_conv3d': [128, 256, 512],
#                 'encoder_conv1d': [512, 64, 1]}

# dec_features = {'decoder_fwd': [1024, 512, 128, 6],
#                 'decoder_conv': [512, 512]}
