import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os
import CNN_recurrent
import motion
import MC_network
from hific import archs
from compare_gan.gans import loss_lib as compare_gan_loss_lib
from lpips_tensorflow import lpips_tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--q", type=str, default='mi', choices=['lo', 'mi', 'hi'])
parser.add_argument("--b", type=int, default=4)
parser.add_argument("--N", type=int, default=128)
parser.add_argument("--M", type=int, default=128)
parser.add_argument("--w_g", type=float, default=0.1)
args = parser.parse_args()

batch_size = args.b
Height = 256
Width = 256
Channel = 3
F_num = 5
I_quality = args.q

activation = tf.nn.relu
discRNN = archs.DiscriminatorRNN()

lr_init = 1e-4

def read_png(path):

    image_group = []

    for i in range(F_num + 1):

        if i == 0:
          string = tf.read_file(path + '/im1_' + str(I_quality) + '.png')
        else:
          string = tf.read_file(path + '/im' + str(i) + '.png')

        image = tf.image.decode_image(string, channels=3)
        image = tf.cast(image, tf.float32)
        image /= 255

        image_group.append(image)

    return tf.stack(image_group, axis=0)

# with tf.device("/cpu:0"):
train_files = np.load('folder.npy').tolist()

train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
train_dataset = train_dataset.map(read_png,
                                  num_parallel_calls=16)
train_dataset = train_dataset.map(lambda x: tf.random_crop(x, (F_num + 1, Height, Width, 3)),
                                  num_parallel_calls=16)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(32)

input_tensor = train_dataset.make_one_shot_iterator().get_next()
input_tensor.set_shape((batch_size, F_num + 1, Height, Width, 3))

c_enc_mv = tf.zeros([batch_size, Height//4, Width//4, args.N])
h_enc_mv = tf.zeros([batch_size, Height//4, Width//4, args.N])

c_dec_mv = tf.zeros([batch_size, Height//4, Width//4, args.N])
h_dec_mv = tf.zeros([batch_size, Height//4, Width//4, args.N])

c_enc_res = tf.zeros([batch_size, Height//4, Width//4, args.N])
h_enc_res = tf.zeros([batch_size, Height//4, Width//4, args.N])

c_dec_res = tf.zeros([batch_size, Height//4, Width//4, args.N])
h_dec_res = tf.zeros([batch_size, Height//4, Width//4, args.N])

c_disc = tf.zeros([2 * batch_size, Height//2, Width//2, 64])
h_disc = tf.zeros([2 * batch_size, Height//2, Width//2, 64])

train_bpp_MV = tf.zeros([])
train_bpp_Res = tf.zeros([])
total_mse = tf.zeros([])
psnr = tf.zeros([])
LPIPS = tf.zeros([])
GAN_loss = tf.zeros([])
DIS_loss = tf.zeros([])

entropy_bottleneck = tfc.EntropyBottleneck()
entropy_bottleneck2 = tfc.EntropyBottleneck()

for f in range(F_num - 1):

    Y1_raw_tensor = input_tensor[:, f + 2]
    Y0_raw_tensor = input_tensor[:, f + 1]

    if f == 0:
        Y0_com_tensor = input_tensor[:, 0]
    else:
        Y0_com_tensor = Y1_decoded

    with tf.variable_scope("flow_motion", reuse=tf.AUTO_REUSE):

        flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com_tensor, Y1_raw_tensor, batch_size, Height, Width)
        Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com_tensor, flow_tensor)

    # Encode flow
    mt, c_enc_mv, h_enc_mv = CNN_recurrent.MV_analysis(flow_tensor, num_filters=args.N, out_filters=args.M,
                                   Height=Height, Width=Width,
                                   c_state=c_enc_mv, h_state=h_enc_mv, act=activation)

    string = entropy_bottleneck.compress(mt)
    mt_hat, MV_likelihoods = entropy_bottleneck(mt, training=True)

    flow_hat, c_dec_mv, h_dec_mv = CNN_recurrent.MV_synthesis(mt_hat, num_filters=args.N,
                                          Height=Height, Width=Width,
                                          c_state=c_dec_mv, h_state=h_dec_mv, act=activation)

    # Motion Compensation
    Y1_warp = tf.contrib.image.dense_image_warp(Y0_com_tensor, flow_hat)

    MC_input = tf.concat([flow_hat, Y0_com_tensor, Y1_warp], axis=-1)
    Y1_MC = MC_network.MC_new(MC_input, reuse=tf.AUTO_REUSE)

    # Encode residual
    Res = Y1_raw_tensor - Y1_MC

    y, c_enc_res, h_enc_res = CNN_recurrent.Res_analysis(Res, num_filters=args.N, out_filters=args.M,
                                   Height=Height, Width=Width,
                                   c_state=c_enc_res, h_state=h_enc_res, act=activation)

    string2 = entropy_bottleneck2.compress(y)
    y_hat, Res_likelihoods = entropy_bottleneck2(y, training=True)

    Res_hat, c_dec_res, h_dec_res = CNN_recurrent.Res_synthesis(y_hat, num_filters=args.N,
                                          Height=Height, Width=Width,
                                          c_state=c_dec_res, h_state=h_dec_res, act=activation)

    # Reconstructed frame
    Y1_decoded = Res_hat + Y1_MC

    # restore_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    with tf.variable_scope("disc"):

        raw_sample = tf.concat([Y0_raw_tensor, Y1_raw_tensor, flow_tensor], axis=-1)
        com_sample = tf.concat([Y0_com_tensor, Y1_decoded, flow_tensor], axis=-1)

        frames = tf.concat([raw_sample, com_sample], axis=0)
        latents = tf.concat([tf.concat([mt, y], axis=-1),
                             tf.concat([mt, y], axis=-1)], axis=0)

        disc_in = (frames, latents, c_disc, h_disc)

        disc_out_all = discRNN(disc_in)

        d_real, d_fake = tf.split(disc_out_all.d_all, 2)
        d_real_logits, d_fake_logits = tf.split(disc_out_all.d_all_logits, 2)

        c_disc = disc_out_all.c_state
        h_disc = disc_out_all.h_state

    ## GAN loss
    d_loss, _, _, g_loss = compare_gan_loss_lib.get_losses(
        fn=compare_gan_loss_lib.non_saturating,
        d_real=d_real,
        d_fake=d_fake,
        d_real_logits=d_real_logits,
        d_fake_logits=d_fake_logits)

    # Total number of bits divided by number of pixels.
    train_bpp_MV += tf.reduce_sum(tf.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
    train_bpp_Res += tf.reduce_sum(tf.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
    mse_frame = tf.reduce_mean(tf.squared_difference(Y1_decoded, Y1_raw_tensor))

    total_mse += mse_frame
    psnr += 10.0*tf.log(1.0/mse_frame)/tf.log(10.0)
    LPIPS += tf.reduce_mean(lpips_tf.lpips(Y1_decoded, Y1_raw_tensor, model='net-lin', net='alex'))
    GAN_loss += g_loss
    DIS_loss += d_loss


if args.q == 'lo':
    w_bpp = tf.where(train_bpp_MV + train_bpp_Res > 0.025, 3., 0.01)
elif args.q == 'mi':
    w_bpp = tf.where(train_bpp_MV + train_bpp_Res > 0.05, 1., 0.01)
else:
    w_bpp = tf.where(train_bpp_MV + train_bpp_Res > 0.1, 0.3, 0.001)

EGP_loss = 100 * total_mse + w_bpp * (train_bpp_MV + train_bpp_Res) + LPIPS + args.w_g * GAN_loss

step = tf.train.get_or_create_global_step()
learning_rate = 1e-5

all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

EGP_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_motion') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MV') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='analysis') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='synthesis') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='entropy') + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mc')

train_total = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(EGP_loss, var_list=EGP_var, global_step=step)

aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

aux_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step2 = aux_optimizer2.minimize(entropy_bottleneck2.losses[0])

train_EGP_op = tf.group(train_total, aux_step, aux_step2, entropy_bottleneck.updates[0], entropy_bottleneck2.updates[0])

D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')
train_D_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(DIS_loss, var_list=D_var)

tf.summary.scalar('psnr', psnr/(F_num - 1))
tf.summary.scalar('bits_res', train_bpp_Res/(F_num - 1))
tf.summary.scalar('bits_mv', train_bpp_MV/(F_num - 1))
tf.summary.scalar('bits_total', train_bpp_MV/(F_num - 1) + train_bpp_Res/(F_num - 1))
tf.summary.scalar('total_mse', total_mse/(F_num - 1))
tf.summary.scalar('EGPloss', EGP_loss/(F_num - 1))
tf.summary.scalar('Dloss', DIS_loss/(F_num - 1))
tf.summary.scalar('Gloss', GAN_loss/(F_num - 1))
tf.summary.scalar('lpips', LPIPS/(F_num - 1))

save_path = './PLVC_model/RAE_GAN_' + str(args.q)
os.makedirs(save_path, exist_ok=True)

if args.q == 'lo': l = 256
elif args.q == 'mi': l = 512
else: l = 1024

restore_path = './model/RAE_PSNR_' + str(l)
saver_restore = tf.train.Saver(var_list=EGP_var, max_to_keep=None)

def load_model(scaffold, session):
    saver_restore.restore(session, save_path=restore_path + '/model.ckpt')

hooks = [
    tf.train.StopAtStepHook(last_step=100000),
    tf.train.NanTensorHook(EGP_loss),
    tf.train.NanTensorHook(DIS_loss),
]
with tf.train.MonitoredTrainingSession(
        hooks=hooks, checkpoint_dir=save_path,
        save_checkpoint_steps=10000, save_summaries_steps=2000,
        scaffold=tf.train.Scaffold(
            init_fn=load_model,
            saver=tf.train.Saver(max_to_keep=None)
            )
        ) as sess:

    while not sess.should_stop():
        sess.run(train_D_op)
        sess.run(train_EGP_op)
