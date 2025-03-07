import torch
from cnn_model import Image_model
from unet import UNet
import numpy as np
from config import Config
import wandb
from util import AverageMeter
from tqdm import tqdm
import torchmetrics
# 這版更動 reconstruct error ：只算 noise 相似度

class Trainer():
    def __init__(self, 
        g_first_phase_model, 
        d_first_phase_model, 
        train_image_dataloader,
        val_image_dataloader,  
        epochs,
        phase,
        g_model_ckpt_dir,
        d_model_ckpt_dir
        ):
        self.g_first_phase_model = g_first_phase_model
        self.d_first_phase_model = d_first_phase_model
        self.train_image_dataloader = train_image_dataloader
        self.val_image_dataloader = val_image_dataloader
        self.epochs = epochs
        self.phase = phase
        self.g_model_ckpt_dir = g_model_ckpt_dir
        self.d_model_ckpt_dir = d_model_ckpt_dir

    def start_training(self):

        config = Config()
        g_first_phase_model = self.g_first_phase_model
        d_first_phase_model = self.d_first_phase_model

        # define optimizer generator
        g_optimizer = torch.optim.Adam(g_first_phase_model.parameters(), lr = config.g_lr)
        # discriminatord
        d_optimizer = torch.optim.Adam(d_first_phase_model.parameters(), lr = config.d_lr)

        #learning rate scheduler
        if config.type_scheduler == 'cycle':
            g_scheduler = torch.optim.lr_scheduler.OneCycleLR(g_optimizer, max_lr = config.g_lr, 
                                                            epochs = self.epochs, 
                                                            steps_per_epoch = len(self.train_image_dataloader), 
                                                            pct_start = 0.4)

            d_scheduler = torch.optim.lr_scheduler.OneCycleLR(d_optimizer, max_lr = config.d_lr, 
                                                            epochs = self.epochs, 
                                                            steps_per_epoch = len(self.train_image_dataloader), 
                                                            pct_start = 0.4)
        elif config.type_scheduler == 'step':
            g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size = 20, gamma = 0.6, last_epoch = -1, verbose = False)
            d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size = 20, gamma = 0.6, last_epoch = -1, verbose = False)
        elif config.type_scheduler == 'reduce':
            g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, mode='min', factor=0.6, patience = 40, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
            d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, mode='min', factor=0.6, patience = 40, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        elif config.type_scheduler == 'cosine':
            g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max = 100, eta_min=8e-6)
            d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max = 100, eta_min=8e-6)
        else:
            g_scheduler = None
            d_scheduler = None



        # Start training
        g_min_loss_2 = np.inf
        d_min_loss_2 = np.inf
        self.g_first_phase_model.to(config.DEVICE)
        self.d_first_phase_model.to(config.DEVICE)
        tq = tqdm(range(self.epochs))
        for epoch in tq:
            print(f"Epochs: {epoch + 1}")
            (g_batch_train_loss, d_batch_train_loss) = self.train_one_epoch(
                                                                            epoch,
                                                                            g_first_phase_model, 
                                                                            d_first_phase_model, 
                                                                            self.train_image_dataloader, 
                                                                            g_optimizer, 
                                                                            d_optimizer, 
                                                                            g_scheduler, 
                                                                            d_scheduler, 
                                                                            self.phase)
            
            print(f'Generator Training Loss: {g_batch_train_loss}, Discriminator Training Loss: {d_batch_train_loss}')

            # Evaluation
            (g_batch_val_loss, d_batch_val_loss, val_accuracy) = self.evaluation(g_first_phase_model,  
                                                                                 d_first_phase_model, 
                                                                                 self.val_image_dataloader, 
                                                                                 self.phase)
            
            print(f'Generator Validation Loss: {g_batch_val_loss}, Discriminator Validation Loss: {d_batch_val_loss}')
            print(f'Validation acc: {val_accuracy}')




            # Sheduler
            if config.type_scheduler == "step":
                g_scheduler.step()
                d_scheduler.step()
            elif config.type_scheduler == "reduce":
                g_scheduler.step(g_batch_val_loss)
                d_scheduler.step(d_batch_val_loss)
            elif config.type_scheduler == 'cosine':
                g_scheduler.step()
                d_scheduler.step()

            # find best accuracy and save model chekpoint Generator
            
            if g_min_loss_2 > g_batch_val_loss:
                g_min_loss_2 = g_batch_val_loss
                self.save_checkpoint(g_first_phase_model, self.g_model_ckpt_dir, epoch, False)

            if d_min_loss_2 > d_batch_val_loss:
                d_min_loss_2 = d_batch_val_loss
                self.save_checkpoint(d_first_phase_model, self.d_model_ckpt_dir, epoch, False)
            

            if config.use_wandb:
                wandb.log({"generator_train_loss": g_batch_train_loss,
                            "discriminator_train_loss": d_batch_train_loss,
                            "generator_val_loss": g_batch_val_loss,
                            "discriminator_val_loss": d_batch_val_loss,
                            'val_acc': val_accuracy,
                            'generator_lr': g_optimizer.param_groups[0]['lr'], 
                            'discriminator_lr': d_optimizer.param_groups[0]['lr'], 
                            }, step=epoch)

        self.save_checkpoint(g_first_phase_model, self.g_model_ckpt_dir, epoch, True)
        self.save_checkpoint(d_first_phase_model, self.d_model_ckpt_dir, epoch, True)


    def train_one_epoch(self, epoch, g_model, d_model, train_image_dataloader, g_optimizer, d_optimizer, g_scheduler, d_scheduler, phase):
        config = Config()
        g_avgmeter_train_loss = AverageMeter()
        d_avgmeter_train_loss = AverageMeter()
        g_model.train()
        d_model.train()

        
        for index, (X_train_image, y_train) in enumerate(train_image_dataloader):
            if y_train.shape[0] <= 2:
                break
            X_train_image = X_train_image.reshape(-1, 1, config.img_size, config.img_size)
            # Seperate Moire image and clean image
            X_train_image_moire = torch.stack([X_train_image[i] for i in range(y_train.shape[0]) if y_train[i] == 1]).to(config.DEVICE)
            X_train_image_clean = torch.stack([X_train_image[i] for i in range(y_train.shape[0]) if y_train[i] == 0]).to(config.DEVICE)
            ################

            # Make label (for discriminator)
            moire_label_real = torch.Tensor([1 for i in range(X_train_image_moire.shape[0])]).to(config.DEVICE)
            moire_label_fake = torch.Tensor([0 for i in range(X_train_image_moire.shape[0])]).to(config.DEVICE)
            clean_label = torch.Tensor([0 for i in range(X_train_image_clean.shape[0])]).to(config.DEVICE)




            ### Generator
            g_optimizer.zero_grad()
            
            noise = g_model(X_train_image_moire)
            fake_clean = X_train_image_moire - noise
            fake_output = d_model(fake_clean)
            fake_output = fake_output.reshape(-1)


            ## GAN loss
            loss_g_value = config.loss_fn(fake_output, moire_label_fake) #越像 0 越好
            # loss_g_value = torch.mean(fake_output)


            if phase == 2 or phase == 3:
                ### clean consistency
                ## clean image input to genertor, the output will be 0 
                ### TODO
                clean_noise = g_model(X_train_image_clean)
                zero_map = torch.zeros(clean_noise.shape).to(config.DEVICE)
                clean_cons_loss = config.loss_fn_constrain(clean_noise, zero_map)


                ### Pure Noise consistency
                ## Noise map input to genertor, the output will still be noise map
                ### TODO
                re_noise = g_model(noise)
                pure_noise_cons_loss = config.loss_fn_constrain(re_noise, noise)
                #print("pure_noise_cons_loss:")
                #print(pure_noise_cons_loss)

            if phase == 3:
                ### Restoration consistency
                ### TODO
                if X_train_image_clean.shape[0] > noise.shape[0]:
                    for i in range((X_train_image_clean.shape[0] - noise.shape[0])):
                        noise = torch.cat((noise, noise[0].unsqueeze(0)), 0)
                elif X_train_image_clean.shape[0] < noise.shape[0]:
                    for i in range((noise.shape[0] - X_train_image_clean.shape[0])):
                        noise = noise[:-1]

                fake_noise_image = X_train_image_clean + noise # 加上一開始生成的 noise
                fake_noise_image_noise = g_model(fake_noise_image) # 丟回去 generator
                clean_fake_noise_image = fake_noise_image - fake_noise_image_noise
                reconstruct_loss = config.loss_fn_constrain(clean_fake_noise_image, X_train_image_clean)
                # reconstruct_loss = config.loss_fn_constrain(fake_noise_image_noise, noise)


                # Fool discriminator for reconstruct
                '''
                moire_label_fake_2 = torch.Tensor([0 for i in range(clean_fake_noise_image.shape[0])]).to(config.DEVICE)
                moire_label_real_2 = torch.Tensor([1 for i in range(clean_fake_noise_image.shape[0])]).to(config.DEVICE)
                fake_output_2 = d_model(clean_fake_noise_image).reshape(-1)
                loss_g_value_2 = config.loss_fn(fake_output_2, moire_label_fake_2) 
                '''



            if phase == 1:
                new_loss_g = loss_g_value 
            elif phase == 2:
                new_loss_g = loss_g_value + config.phase2_w1 * clean_cons_loss + config.phase2_w2 * pure_noise_cons_loss
            else:
                #new_loss_g = loss_g_value + loss_g_value_2 + config.w1 * clean_cons_loss + config.w2 * pure_noise_cons_loss + config.w3 * reconstruct_loss
                new_loss_g = loss_g_value + config.phase3_w1 * clean_cons_loss + config.phase3_w2 * pure_noise_cons_loss + config.phase3_w3 * reconstruct_loss
            


            # 讓 discriminator 更新多一點 generator 更新少一點
            '''
            if epoch >= 4:
                new_loss_g.backward()
                g_optimizer.step()
                g_avgmeter_train_loss.update(new_loss_g, n = X_train_image_moire.size(0)) # generator loss
            '''

            #if ((index + 1) % 5 != 0) and epoch >= 0:
            if epoch >= 0:
                new_loss_g.backward()
                g_optimizer.step()
                g_avgmeter_train_loss.update(new_loss_g, n = X_train_image_moire.size(0)) # generator loss
            


            d_optimizer.zero_grad()

            clean_output = d_model(X_train_image_clean)

            clean_output = clean_output.reshape(-1)
            real_loss = config.loss_fn(clean_output, clean_label)

            fake_catch_loss = config.loss_fn(d_model(fake_clean.detach()).reshape(-1), moire_label_real)


            #fake_loss_2 = config.loss_fn(d_model(clean_fake_noise_image.detach()).reshape(-1), moire_label_real_2)

            #real_loss.backward()
            #fake_loss.backward()
            '''
            if phase == 3:
                loss_d_value = (real_loss + fake_catch_loss + fake_loss_2) / 3
            else:
                loss_d_value = (real_loss + fake_catch_loss) / 2
            '''
                

            loss_d_value = (real_loss + fake_catch_loss) / 2

            loss_d_value.backward()
            d_optimizer.step()
            d_avgmeter_train_loss.update(loss_d_value, n = X_train_image_clean.size(0)) # discriminator loss



            if config.type_scheduler == 'cycle':

                if epoch >= 0:
                    g_scheduler.step()

                d_scheduler.step()
        
        return (g_avgmeter_train_loss.avg, d_avgmeter_train_loss.avg)




    @torch.no_grad()
    def evaluation(self, g_model, d_model, val_image_dataloader, phase):

        config = Config()
        acc_metric = torchmetrics.Accuracy(task="binary").to(config.DEVICE)
        g_avgmeter_val_loss = AverageMeter()
        d_avgmeter_val_loss = AverageMeter()
        g_model.eval()
        d_model.eval()
        accuracy_accum = 0


        for (X_val_image, y_val)in val_image_dataloader:
            #X_val_image = X_val_image.to(device)
            #y_val = y_val.to(device)
            if y_val.shape[0] <= 2:
                break
            X_val_image = X_val_image.reshape(-1, 1, config.img_size, config.img_size)
            #y_val = y_val.reshape(-1, 1)
            
            # classify moire image and clean image############
            X_val_image_moire = torch.stack([X_val_image[i] for i in range(y_val.shape[0]) if y_val[i] == 1]).to(config.DEVICE)
            X_val_image_clean = torch.stack([X_val_image[i] for i in range(y_val.shape[0]) if y_val[i] == 0]).to(config.DEVICE)



            moire_label_real = torch.Tensor([1 for i in range(X_val_image_moire.shape[0])]).to(config.DEVICE)
            moire_label_fake = torch.Tensor([0 for i in range(X_val_image_moire.shape[0])]).to(config.DEVICE)
            clean_label = torch.Tensor([0 for i in range(X_val_image_clean.shape[0])]).to(config.DEVICE)

            ################


            ### Generator
            
            noise = g_model(X_val_image_moire)
            fake_clean = X_val_image_moire - noise
            fake_output = d_model(fake_clean)
            fake_output = fake_output.reshape(-1)

            ## GAN loss
            loss_g_value = config.loss_fn(fake_output, moire_label_fake) # fake output 越像 0 越好 ## GAN loss
            # loss_g_value = torch.mean(fake_output)

            if phase == 2 or phase == 3:
                ### clean loss
                ## clean image input to genertor, the output will be 0
                ### TODO
                clean_noise = g_model(X_val_image_clean)
                zero_map = torch.zeros(clean_noise.shape).to(config.DEVICE)
                clean_cons_loss = config.loss_fn_constrain(clean_noise, zero_map)



                ### Pure Noise consistency
                ## Noise map input to genertor, the output will still be noise map
                ### TODO
                re_noise = g_model(noise)
                pure_noise_cons_loss = config.loss_fn_constrain(noise, re_noise)


            if phase == 3:
                ### add constrain
                ### Restoration consistency
                ### TODO
                if X_val_image_clean.shape[0] > noise.shape[0]:
                    for i in range((X_val_image_clean.shape[0] - noise.shape[0])):
                        noise = torch.cat((noise, noise[0].unsqueeze(0)), 0)
                elif X_val_image_clean.shape[0] < noise.shape[0]:
                    for i in range((noise.shape[0] - X_val_image_clean.shape[0])):
                        noise = noise[:-1]

                fake_noise_image = X_val_image_clean + noise # 加上一開始生成的 noise
                fake_noise_image_noise = g_model(fake_noise_image) # 丟回去 generator
                clean_fake_noise_image = fake_noise_image - fake_noise_image_noise
                reconstruct_loss = config.loss_fn_constrain(clean_fake_noise_image, X_val_image_clean)
                # reconstruct_loss = config.loss_fn_constrain(fake_noise_image_noise, noise)

                if config.freq_loss == True:
                    # Frequency loss
                    reconstruct_loss_freq_total = 0
                    for i in range(clean_fake_noise_image.shape[0]):
                        clean_fake_noise_image_freq = torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(clean_fake_noise_image[i]))))
                        X_val_image_clean_freq = torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(X_val_image_clean[i]))))
                        reconstruct_loss_freq = config.loss_fn_constrain(clean_fake_noise_image, X_val_image_clean)
                        reconstruct_loss_freq_total += reconstruct_loss_freq

                    reconstruct_loss_freq_total_avg = reconstruct_loss_freq_total / clean_fake_noise_image.shape[0]

                '''
                # Fool discriminator
                moire_label_fake_2 = torch.Tensor([0 for i in range(clean_fake_noise_image.shape[0])]).to(config.DEVICE)
                moire_label_real_2 = torch.Tensor([1 for i in range(clean_fake_noise_image.shape[0])]).to(config.DEVICE)
                fake_output_2 = d_model(clean_fake_noise_image).reshape(-1)
                loss_g_value_2 = config.loss_fn(fake_output_2, moire_label_fake_2) 
                '''



            if phase == 1:
                new_loss_g = loss_g_value 

            elif phase == 2:
                new_loss_g = loss_g_value + config.phase2_w1 * clean_cons_loss + config.phase2_w2 * pure_noise_cons_loss
            else:
                #new_loss_g = loss_g_value + loss_g_value_2 + config.w1 * clean_cons_loss + config.w2 * pure_noise_cons_loss + config.w3 * reconstruct_loss
                if config.freq_loss == True:
                    new_loss_g = loss_g_value + config.phase3_w1 * clean_cons_loss + config.phase3_w2 * pure_noise_cons_loss + config.phase3_w3 * reconstruct_loss + config.phase3_w4 * reconstruct_loss_freq_total_avg
                else:
                    new_loss_g = loss_g_value + config.phase3_w1 * clean_cons_loss + config.phase3_w2 * pure_noise_cons_loss + config.phase3_w3 * reconstruct_loss

            
            g_avgmeter_val_loss.update(new_loss_g, n = X_val_image_moire.size(0)) # generator loss


            ### Discriminator
            clean_output = d_model(X_val_image_clean)
            clean_output = clean_output.reshape(-1)
            real_loss = config.loss_fn(clean_output, clean_label)
            fake_output_d2 = d_model(fake_clean.detach()).reshape(-1)
            fake_catch_loss = config.loss_fn(fake_output_d2, moire_label_real)

            # real_loss = torch.mean(clean_output)
            # fake_loss =  - torch.mean(fake_output_d2)



            # fake_loss_2 = config.loss_fn(d_model(clean_fake_noise_image.detach()).reshape(-1), moire_label_real_2)

            '''
            if phase == 3:
                loss_d_value = (real_loss + fake_catch_loss + fake_loss_2) / 3
                # loss_d_value = (real_loss + fake_catch_loss ) / 2
            else:
                loss_d_value = (real_loss + fake_catch_loss) / 2
            '''


            loss_d_value = (real_loss + fake_catch_loss) / 2

            total_batch_output = torch.cat((clean_output, fake_output_d2))
            total_batch_ypred = torch.cat((clean_label, moire_label_real))

            
            
            d_avgmeter_val_loss.update(loss_d_value, n = X_val_image_clean.size(0)) # discriminator loss


            acc = acc_metric(total_batch_output, total_batch_ypred)
            accuracy_accum += acc.item()

            #avgmeter_val_loss.update(loss, n = X_val_image.size(0))


        total_acc = accuracy_accum / len(val_image_dataloader)

        return (g_avgmeter_val_loss.avg, d_avgmeter_val_loss.avg, total_acc)

    def fix_data_imbalanced(self, total_moire, X_train_image_moire, diff, img_size, device):
        random_list = np.random.randint(low = 0, high = total_moire.shape[0] - 1, size = diff)

        for rnum in random_list:
            X_train_image_moire = torch.cat((X_train_image_moire, total_moire[rnum].reshape(-1, 1, img_size, img_size)))

        return X_train_image_moire


    def save_checkpoint(self, model, ckp_dir, epoch, final):
        ckp_path = ckp_dir / '{}-model.pth'.format(epoch + 1)
        if final == True:
            best_ckp_path = ckp_dir / 'final-model.pth'
        else:
            best_ckp_path = ckp_dir / 'best-model.pth'
        torch.save(model.state_dict(), ckp_path)
        torch.save(model.state_dict(), best_ckp_path)
        print(f'Saved model checkpoints into {ckp_path}...')