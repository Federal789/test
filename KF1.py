import numpy as np
import pandas as pd

from main import *

np.random.seed(1)
tf.compat.v1.set_random_seed(1)
random.seed(1)

# for j in range(1, batch_size + 1):
#    path1 = './images/comparison/KF1' + r'/per_{}'.format(per) + r'/sam_' + str(j)
#    os.makedirs(path1)

# for i in range(1, batch_size + 1):
#     for k in range(1, iterations / plot_interval):
#         path = './images/comparison/KF1' + r'/per_{}'.format(per) + r'/sample_' + str(i) + r'/epoch_' + str(plot_interval * k)
#         os.makedirs(path)

def train(iterations, batch_size, sample_interval, mt_size, set_class, weight_per_class, get_samples, get_labels, num,
          x_val, y_val):
    # Labels for real images: all ones
    # global dis_wb, fscore_val, iteration
    real = np.ones((batch_size, 1))

    # Labels for fake images: all zeros
    fake = np.zeros((batch_size, 1))

    F1 = []
    supervised_losses = []
    unsupervised_losses = []
    generator_losses = []
    iteration_checkpoints = []
    best_sc = 0
    best_para = []
    best_gen_para = []
    best_epoch = 1
    srm, sfm = [], []

    for iteration in range(iterations):

        # -------------------------
        #  Train the Discriminator
        # -------------------------

        # Generate a batch of fake images

        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_samples = generator.predict(z)

        # Train on real labeled examples
        sup_samples, sup_labels, idx1 = dataset.batch_labeled(batch_size, get_samples, get_labels)

        sup_labels_ = sup_labels.copy()

        # One-hot encode labels
        sup_labels = to_categorical(sup_labels, num_classes=num_classes)

        # Train on real labeled examples
        d_loss_supervised, accuracy = discriminator_supervised.train_on_batch(sup_samples, sup_labels,
                                                                              class_weight=
                                                                              {set_class[0]: 1 * weight_per_class[0],
                                                                               set_class[1]: 1 * weight_per_class[1],
                                                                               set_class[2]: 1 * weight_per_class[2]}
                                                                              )

        # Get unlabeled examples
        samples_unlabeled, lab_un = dataset.batch_unlabeled(batch_size)

        sam_wei = gamma * np.ones(len(samples_unlabeled))

        # Train on real unlabeled examples
        d_loss_real = discriminator_unsupervised.train_on_batch(samples_unlabeled, real, sample_weight=sam_wei)

        # Train on fake examples
        d_loss_fake = discriminator_unsupervised.train_on_batch(gen_samples, fake, sample_weight=sam_wei)

        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)  # 超参数

        # if ((iteration + 1 == 10) or (iteration + 1 == 200) or (iteration + 1 == 800)
        #     or (iteration + 1 == 2000) or (iteration + 1 == 3600) or (iteration + 1 == 4800)
        #     or (iteration + 1 == 6400) or (iteration + 1 == iterations)) & (d_loss_unsupervised <= 3):
        #     # tsne
        #     dis_3 = Model(inputs=discriminator_net.inputs,
        #                   outputs=discriminator_net.get_layer('dense_1').output)
        #     dis_3_fake = dis_3.predict(gen_samples)
        #     dis_3_real = dis_3.predict(samples_unlabeled)
        #     dis_3_lab = dis_3.predict(sup_samples)
        #     # real_sam = samples_unlabeled.reshape(batch_size, -1)
        #     # sup_sam = sup_samples.reshape(batch_size, -1)
        #     gen_tsne = dis_3_fake.copy()
        #     real_tsne = dis_3_real.copy()
        #     sup_tsne = dis_3_lab.copy()
        #     gen_real = np.vstack((sup_tsne, real_tsne, gen_tsne))
        #     lab_unlabeled = [x + num_classes for x in lab_un]
        #     gr_pre = np.vstack((sup_labels_, lab_unlabeled,
        #                         2 * num_classes * np.ones((batch_size, 1))))
        #     tsne = TSNE(n_components=2, init='pca', random_state=1)
        #     label = gr_pre.reshape(3 * batch_size)
        #     result = tsne.fit_transform(gen_real)                     # 进行降维
        #     np.save('./result/res{}_{}_{}.npy'.format(per, num, iteration + 1), result)
        #     np.save('./result/lab{}_{}_{}.npy'.format(per, num, iteration + 1), label)
        #     plot_embedding_gen(result, label, 1, '{}% labels, epoch{}'.format(per, iteration + 1),
        #                        per, 'epoch{}'.format(iteration + 1))  # 显示数据

        # if ((iteration + 1 == 500) or (iteration + 1 == 3000) or (iteration + 1 == iterations)):
        #     # Comparison of the real and the fake samples.
        #     curves = list(range(num_features))
        #     for i in range(batch_size):
        #         # Comparison of the real and the fake samples without any convolutions.
        #         gen_sam_ = pd.DataFrame(gen_samples[i, :, :, 0])
        #         gen_sam = gen_sam_.iloc[:, curves]
        #         plt_curves(gen_sam, figsize=(11, 5), dpi=420, sup1="Generated samples of model 1 in epoch{}".format(iteration+1))
        #         plt.savefig('./images/comparison/KF1' + '/per_{}'.format(per) + '/sam_{}'.format(
        #             i + 1) + '/epoch_{}.jpg'.format(iteration + 1))


        # ---------------------
        #  Train the Generator
        # ---------------------

        # Generate a batch of fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        # gen_samples = generator.predict(z)

        # Train Generator
        g_loss = gan.train_on_batch(z, real)

        if (iteration + 1) % sample_interval == 0:
            # Save Discriminator supervised classification loss to be plotted after training
            iteration_checkpoints.append(iteration + 1)

            print("%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss unsupervised: %.4f] [G loss: %f]"
                  % (iteration + 1, d_loss_supervised, 100 * accuracy, d_loss_unsupervised, g_loss))

        x2, y2 = x_val * 1, y_val * 1
        # y2_pred = discriminator_supervised.predict_classes(x2)
        y2_pred = np.argmax(discriminator_supervised.predict(x2), axis=1)
        y2_pred = y2_pred.reshape(-1, 1)
        fscore_val = f1_score(y2, y2_pred, average='macro')

        supervised_losses.append(d_loss_supervised)
        generator_losses.append(g_loss)
        unsupervised_losses.append(d_loss_unsupervised)
        F1.append(fscore_val)
        if (iteration >= 1) & (fscore_val > best_sc):
            dis_wb = discriminator_supervised.get_weights()
            gen_wb = generator.get_weights()
            y_te_1 = np.argmax(discriminator_supervised.predict(x_test), axis=1)
            test_score_1 = f1_score(y_test, y_te_1, average='macro')
            best_para = dis_wb * 1
            best_gen_para = gen_wb
            best_sc = fscore_val * 1
            best_test = test_score_1 * 1
            best_epoch = iteration + 1

        # if (iteration >= 10000 - 1) & ((np.array(F1[iteration - ( - 1): iteration + 1]) >= threshold).all()):
        if iteration + 1 == iterations:
            print('The number of steps to exit the loop is {}'.format(best_epoch))
            print('The final result is {}'.format(best_sc))
            print('The final result of test set is {}'.format(best_test))
            print('\n')

            curves = list(range(num_features))
            for i in range(batch_size):
                sup_sam_ = pd.DataFrame(samples_unlabeled[i, :, :, 0])
                sup_sam = sup_sam_.iloc[:, curves]
                plt_curves(sup_sam, figsize=(11, 5), dpi=420, sup1="The real samples of model 1")
                plt.savefig('./images/comparison/KF1' + '/per_{}'.format(per) + '/real_{}.jpg'.format(i + 1))
            # #
            # plt.figure()
            # # plt.plot(list(range(0, iteration + 1, mt_size)), F1[::mt_size])
            # plt.plot(list(range(0, iterations)), F1)
            # filePath1 = './image/per_{}/'.format(per) + str('network_{}').format(num) + '_f1score'
            # plt.savefig(filePath1)
            # plt.figure()
            # plt.plot(list(range(0, iterations)), generator_losses)
            # filePath2 = './image/per_{}/'.format(per) + str('generator_{}').format(num) + '_gloss'
            # plt.savefig(filePath2)
            # plt.figure()
            # plt.plot(list(range(0, iterations)), supervised_losses)
            # filePath3 = './image/per_{}/'.format(per) + str('supervised_discriminator_{}').format(num) + '_dloss'
            # plt.savefig(filePath3)
            # plt.figure()
            # plt.plot(list(range(0, iterations)), unsupervised_losses)
            # filePath4 = './image/per_{}/'.format(per) + str('unsupervised_discriminator_{}').format(num) + '_dloss'
            # plt.savefig(filePath4)


        #     # print(gen_samples)
        #
        #     # Save Discriminator supervised classification loss to be plotted after training
            supervised_losses.append(d_loss_supervised)
            iteration_checkpoints.append(iteration + 1)

            dis_1 = Model(inputs=discriminator_net.inputs,
                          outputs=discriminator_net.get_layer('leaky_re_lu_1').output)

            dis_2 = Model(inputs=discriminator_net.inputs,
                          outputs=discriminator_net.get_layer('leaky_re_lu_2').output)

            dis_3 = Model(inputs=discriminator_net.inputs,
                          outputs=discriminator_net.get_layer('leaky_re_lu_3').output)

            dis_seq = Model(inputs=discriminator_net.inputs,
                            outputs=discriminator_net.get_layer('dense_1').output)

            dis_1_real = dis_1.predict(samples_unlabeled)
            dis_2_real = dis_2.predict(samples_unlabeled)
            dis_3_real = dis_3.predict(samples_unlabeled)
            dis_seq_real = dis_seq.predict(samples_unlabeled)
            dis_sig_real = discriminator_unsupervised.predict(samples_unlabeled)
        #
        #     dis_1_fake = dis_1.predict(gen_samples)
        #     dis_2_fake = dis_2.predict(gen_samples)
        #     dis_3_fake = dis_3.predict(gen_samples)
        #     dis_seq_fake = dis_seq.predict(gen_samples)
        #     dis_sig_fake = discriminator_unsupervised.predict(gen_samples)
        #
        #     sig_real_mean = np.mean(dis_sig_real)
        #     sig_fake_mean = np.mean(dis_sig_fake)
        #
        #     srm.append(sig_real_mean)
        #     sfm.append(sig_fake_mean)
        #
        #     # Comparison of the real and the fake samples.
        #
        #     for i in range(batch_size):
        #         # Comparison of the real and the fake samples without any convolutions.
        #         plt.figure(figsize=(20, 9))
        #         plt.subplot(1, 2, 1)
        #         plt.plot(gen_samples[i, :, :, 0], list(range(para_size)))
        #         plt.title("Gen_samples", fontsize=14)
        #         plt.subplot(1, 2, 2)
        #         plt.plot(samples_unlabeled[i, :, :, 0], list(range(para_size)))
        #         plt.title("Real_samples", fontsize=14)
        #         plt.suptitle('Comparison of real and fake originally \n Step{}'.format(iteration + 1, g_loss),
        #                      fontsize=18, fontweight='bold')
        #         plt.savefig(filelist_com[int(i + 0 * batch_size)] + '/epoch{}.jpg'.format(iteration + 1))
        #
        #         plt.figure(figsize=(20, 9))
        #         plt.subplot(1, 2, 1)
        #         plt.plot(dis_1_fake[i, :, :, 0], list(range(dis_1_fake.shape[1])))
        #         plt.title("Gen_samples_1", fontsize=14)
        #         plt.subplot(1, 2, 2)
        #         plt.plot(dis_1_real[i, :, :, 0], list(range(dis_1_real.shape[1])))
        #         plt.title("Real_samples_1", fontsize=14)
        #         plt.suptitle(
        #             'Comparison of real and fake through C1 \n Step{}:G_loss'.format(iteration + 1, g_loss),
        #             fontsize=18, fontweight='bold')
        #         plt.savefig(filelist_com[int(i + 1 * batch_size / 4)] + '/epoch{}.jpg'.format(iteration + 1))
        #
        #         # Comparison of the real and the fake samples through the 2nd convolutional block.
        #         plt.figure(figsize=(20, 9))
        #         plt.subplot(1, 2, 1)
        #         plt.plot(dis_2_fake[i, :, :, 0], list(range(dis_2_fake.shape[1])))
        #         plt.title("Gen_samples_2", fontsize=14)
        #         plt.subplot(1, 2, 2)
        #         plt.plot(dis_2_real[i, :, :, 0], list(range(dis_2_real.shape[1])))
        #         plt.title("Real_samples_2", fontsize=14)
        #         plt.suptitle('Comparison of real and fake through C2 \n Step{}'.format(iteration + 1, g_loss),
        #                      fontsize=18, fontweight='bold')
        #         plt.savefig(filelist_com[int(i + 2 * batch_size / 4)] + '/epoch{}.jpg'.format(iteration + 1))
        #
        #         # Comparison of the real and the fake samples through the 3rd convolutional block.
        #         plt.figure(figsize=(20, 9))
        #         plt.subplot(1, 2, 1)
        #         plt.plot(dis_3_fake[i, :, :, 0], list(range(dis_3_fake.shape[1])))
        #         plt.title("Gen_samples_3", fontsize=14)
        #         plt.subplot(1, 2, 2)
        #         plt.plot(dis_3_real[i, :, :, 0], list(range(dis_3_real.shape[1])))
        #         plt.title("Real_samples_3", fontsize=14)
        #         plt.suptitle('Comparison of real and fake through C3 \n Step{}'.format(iteration + 1, g_loss),
        #                      fontsize=18, fontweight='bold')
        #         plt.savefig(filelist_com[int(i + 3 * batch_size / 4)] + '/epoch{}.jpg'.format(iteration + 1))

    return best_para, best_sc, best_epoch, best_gen_para


print('The first type of SK-Fold will begin:')

tr_idx, va_idx = TR[0], VAL[0]

x_train_lab, x_test, x_val, y_train_lab, y_test, y_val = \
    np.array(dataset.A_tv_lab)[tr_idx], np.array(dataset.A_test), np.array(dataset.A_tv_lab)[va_idx], \
    np.array(dataset.Y_tv_lab)[tr_idx], np.array(dataset.Y_test), np.array(dataset.Y_tv_lab)[va_idx]



print(len(np.where(y_val == 0)[0]))
print(len(np.where(y_val == 1)[0]))
print(len(np.where(y_val == 2)[0]))

print(len(np.where(y_train_lab == 0)[0]))
print(len(np.where(y_train_lab == 1)[0]))
print(len(np.where(y_train_lab == 2)[0]))

SAM_tr = x_train_lab * 1
LAB_tr = y_train_lab * 1

# Core Discriminator network:
# These layers are shared during supervised and unsupervised training
discriminator_net = build_discriminator(sam_shape, dropout)

# Build & compile the Discriminator for supervised training
tf.compat.v1.set_random_seed(1)
OPT1 = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_1, beta1=beta1)
# OPT1 = Adam(lr=lr_1, beta_1=beta1)
tf.compat.v1.set_random_seed(1)
OPT2 = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_2, beta1=beta1)
# OPT2 = Adam(lr=lr_2, beta_1=beta1)

discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised.compile(loss='categorical_crossentropy', metrics=[getf1], optimizer=OPT2)

# Build & compile the Discriminator for unsupervised training
discriminator_unsupervised = build_discriminator_unsupervised(discriminator_net)
discriminator_unsupervised.compile(loss='binary_crossentropy', optimizer=OPT2)

# Build the Generator
generator = build_generator(z_dim, para_size, num_features)

# Keep Discriminator’s parameters constant for Generator training
discriminator_unsupervised.trainable = False

# Build and compile GAN model with fixed Discriminator to train the Generator
# Note that we are using the Discriminator version with unsupervised output
gan = build_sgan(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=OPT1)

set_class, _, weight_per_class, _ = BUILD_WEIGHT_MAT(y_train_lab, rho=rho, additional_weights=[], show_report=False)
print(set_class)
print(weight_per_class)

dis_wb1, fscore1, step1, gen_wb1 = train(iterations, batch_size, sample_interval, mt_size, set_class,
                                         weight_per_class, SAM_tr, LAB_tr, 1, x_val, y_val)

# Train the SSGAN for the specified number of iterations
x1, y1 = x_train_lab * 1, y_train_lab * 1
y1_pred = discriminator_supervised.predict_classes(x1)
y1_pred = y1_pred.reshape(-1, 1)
fscore_train_1 = np.mean(f1_score(y1, y1_pred, average=None))

