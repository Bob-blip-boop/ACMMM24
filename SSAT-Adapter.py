import os
import clip
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import sys
from datetime import datetime
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import scipy
import math
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from fs_data_prep import data_loader

from utils import *


torch.autograd.set_detect_anomaly(True)


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'Flowers102': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'CIFAR10': 'a photo of a {}.'
}


def load_clip_to_device(device, model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    for param in model.parameters():
            param.requires_grad_(False)
    return model, preprocess


class Linearsampler(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(inplace=True)
        )

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear[0].weight)
        if self.linear[0].bias is not None:
            torch.nn.init.zeros_(self.linear[0].bias)
    
    def init_weights_he(self):
        torch.nn.init.kaiming_normal_(self.linear[0].weight)
        if self.linear[0].bias is not None:
            torch.nn.init.zeros_(self.linear[0].bias)
    
    def reset_parameters(self):
        """Resets the weights of the linear layer to their initialization values"""
        self.linear[0].reset_parameters() 
            
    def forward(self, x):
        return self.linear(x)

class TaskWeightScheduler:
    def __init__(self, initial_weights, final_weights, total_epochs):
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.total_epochs = total_epochs

    def get_weights(self, current_epoch, interpolation):
        weights = []
        for initial_w, final_w in zip(self.initial_weights, self.final_weights):
            if interpolation == 'linear':
                weight = initial_w + (final_w - initial_w) * (current_epoch / self.total_epochs)
            weights.append(weight)
        return torch.tensor(weights)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=2):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, promt_template, classnames, clip_model):
        super().__init__()
        self.promt_template = promt_template
        self.classnames = classnames
        self.clip_model = clip_model
    
    def forward(self):
        temp = self.promt_template
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        print(prompts)
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x

class Anchorhead(torch.nn.Module):
    def __init__(self, input_dim, reduction = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.init_weights()

    def init_weights(self):
        for module in self.fc.modules(): 
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        adapatation = self.fc(x)
        return adapatation

class Variationhead(torch.nn.Module):
    def __init__(self, input_dim, ratio = 0.00001):
        super().__init__()
        self.linear = nn.Sequential(
        nn.Linear(input_dim, input_dim, bias=False),
        # nn.Linear(input_dim, input_dim),
        # nn.ReLU(inplace=True),
        )
        self.ratio = ratio
    def zero_weights(self):
        with torch.no_grad():
            # Set the weights of the linear layer to the identity matrix
            self.linear[0].weight.copy_(torch.eye(self.linear[0].weight.size(0)))

    def forward(self, x):
        adapatation = self.linear(x)
        x = (self.ratio * adapatation) + ((1 - self.ratio) * x)
        # x = adapatation
        return x

def get_euclidean_distance(tensor1, tensor2):
    return torch.dist(tensor1, tensor2, p=2)

class CustomCLIP(nn.Module):

    def __init__(self, promt_template, clip_model, classnames, constraint_lst, constraint_boundary_width, r_ratio):
        super().__init__()
        self.clip_model = clip_model
        self.text_encoder = TextEncoder(promt_template, classnames, self.clip_model)
        self.logit_scale = self.clip_model.logit_scale.exp().to(device)
        self.dtype = torch.float32
        self.promt_template = promt_template

        self.class_cnt = torch.tensor(len(classnames), dtype=torch.float32)
        self.constraint_lst = constraint_lst
        self.constraint_boundary_width = constraint_boundary_width

        self.epoch = 0
        self.ratio = r_ratio


        out_size = 512
        divide_factor = 2
    

        self.aux_adapters = nn.ModuleList([
            Adapter(out_size, divide_factor).to(torch.float32) for _ in range(len(constraint_lst))
        ])
        self.pri_adapter = Adapter(out_size, divide_factor).to(torch.float32)
        self.variationhead = Variationhead(out_size).to(torch.float32)
        self.anchorheads = nn.ModuleList([
            Anchorhead(out_size).to(torch.float32) for _ in range(len(classnames))
        ])
        text_features = self.text_encoder().type(self.dtype)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.adapted_pri_latent = None
        self.pri_lantent = None


    def constraint_loss(self, constraint_t, image_features, text_features_normalized, class_id, verbose = False):
        image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_normalized @ text_features_normalized.T).softmax(dim=-1)
        #* Probability of the image belonging to the class
        label_prob = similarity[class_id]
        #* Diversity of the probability distribution
        diversity_score = torch.std(similarity) / (torch.sqrt(self.class_cnt-1) / self.class_cnt)
        #* Calculate the overall score
        total_score = label_prob * diversity_score
        constraint_t = torch.tensor(constraint_t)
        if verbose:
            print(f"{class_id.item()} {label_prob.item():.4f} {diversity_score.item():.4f} {total_score.item():.4f} {torch.abs(total_score - constraint_t).item():.4f}")
        return torch.abs(total_score - constraint_t)

    def generate_pri_anchor(self, sampler, constraint, image_features, text_features_normalized, class_id):
        #* Normalize the text features
        latent_tensor = image_features.detach()
        #* Initialize the sampler
        sampler = Linearsampler(latent_tensor.shape[0]).to(device).to(self.dtype)
        sampler.init_weights()
        optimizer = torch.optim.Adam(sampler.parameters(), lr=0.1)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 10, factor = 0.5, min_lr=1e-3)
        c_loss = 1
        epoch_cnt = 0
        #* while the distance between the generated latent vector and the predefined constraint boundary is greater than the constraint boundary width
        while c_loss > self.constraint_boundary_width:
            sampler.train()
            #* Generate a similar latent vector
            similar_latent = sampler(latent_tensor)

            #* Measure distance between the generated latent vector and the predefined constraint boundary
            c_loss = self.constraint_loss(constraint, similar_latent, text_features_normalized, class_id)
            #* Update the sampler to sample a latent vector that is closer to the constraint boundary
            if c_loss > self.constraint_boundary_width:
                optimizer.zero_grad()
                c_loss.backward()
                optimizer.step()
                scheduler.step(c_loss)
                epoch_cnt+=1
            if epoch_cnt > 100:
                break
        new_latent = sampler(image_features)
        return new_latent.detach()
    
    def get_clip_features(self, shot_data):
        labels = []
        images, labels = shot_data
        labels = labels.to(device)
        images = images.to(device)
        image_features = self.clip_model.encode_image(images).type(self.dtype)
        return image_features, labels

    def get_anchors(self, pri_image_features, labels, anchor_constraint = 0.95):
        text_features = self.text_features
        anchors = []
        for i in range(len(pri_image_features)):
            latent = pri_image_features[i]
            class_id = labels[i]
            sampler = Linearsampler(latent.shape[0]).to(device).to(self.dtype)
            generated_anchor = self.generate_pri_anchor(sampler, anchor_constraint, latent, text_features, class_id)
            anchors.append(generated_anchor)
        return anchors

    def linear_interpolate(self, constraint_t, primary_latent, anchor_latent):
        interpolated_latent = anchor_latent + (primary_latent - anchor_latent) * constraint_t
        return interpolated_latent
    
    def form_task(self, pri_image_features, anchors_lst):
        self.pri_lantent = pri_image_features.clone()
        
        adpt_pri_image_features = self.variationhead(pri_image_features.clone())
        self.adapted_pri_latent = adpt_pri_image_features

        auxiliary_tasks = []
        for constraint in self.constraint_lst:
            cur_constraint_similar_latents = []
            for i in range(len(self.adapted_pri_latent)):
                latent = self.adapted_pri_latent[i]
                anchor = anchors_lst[i].detach()
                similar_latent = self.linear_interpolate(constraint, latent, anchor)
                cur_constraint_similar_latents.append(similar_latent)
            auxiliary_tasks.append(torch.stack(cur_constraint_similar_latents))
        auxiliary_tasks.append(self.pri_lantent)
        auxiliary_tasks.append(self.adapted_pri_latent)
        return auxiliary_tasks

    def forward(self, auxiliary_tasks):
        adapted_outputs = []
        #* Aux
        for i, adapter in enumerate(self.aux_adapters):
            aux_image_features = adapter(auxiliary_tasks[i])
            aux_image_features = self.ratio * aux_image_features + (1 - self.ratio) * auxiliary_tasks[i]

            aux_image_f = aux_image_features / aux_image_features.norm(dim=-1, keepdim=True)
            aux_logits = self.logit_scale * aux_image_f @ self.text_features.t()
            adapted_outputs.append(aux_logits)

        #* Pri
        pri_x = self.pri_adapter(auxiliary_tasks[-2])
        pri_x = self.ratio * pri_x + (1 - self.ratio) * auxiliary_tasks[-2]

        pri_image_f = pri_x / pri_x.norm(dim=-1, keepdim=True)
        pri_logits = self.logit_scale * pri_image_f @ self.text_features.t()
        adapted_outputs.append(pri_logits)

        #* adapted_Pri
        adapted_pri_x = self.pri_adapter(auxiliary_tasks[-1])
        adapted_pri_x = self.ratio * adapted_pri_x + (1 - self.ratio) * auxiliary_tasks[-1]

        adapted_pri_image_f = adapted_pri_x / adapted_pri_x.norm(dim=-1, keepdim=True)
        adapted_pri_logits = self.logit_scale * adapted_pri_image_f @ self.text_features.t()
        adapted_outputs.append(adapted_pri_logits)

        return adapted_outputs

    def predict(self, image):
        test_features = self.clip_model.encode_image(image).type(self.dtype)
        with torch.no_grad():
            adapter_features = self.pri_adapter(test_features)
        adapted_pri_x = self.ratio * adapter_features + (1 - self.ratio) * test_features
        pri_image_f = adapted_pri_x / adapted_pri_x.norm(dim=-1, keepdim=True)
        outputs = self.logit_scale * pri_image_f @ self.text_features.t()
        return outputs

def nearest_euclidean_distance(instances_pri, instances_adapted, anchors):
    avg_distances_pri = []
    avg_distances_adapted = []

    for i, instance_pri in enumerate(instances_pri):
        hard_distances_pri = (float('inf'), -1)
        hard_distances_adapted = (float('inf'), -1)
        instance_adapted = instances_adapted[i]

        for j, instance_neg in enumerate(instances_pri):
            if i != j:  
                distance_pri = get_euclidean_distance(instance_pri, instance_neg)
                distance_adapted = get_euclidean_distance(instance_adapted, instance_neg)

                if distance_pri < hard_distances_pri[0]:
                    hard_distances_pri = (distance_pri, j)
                if distance_adapted < hard_distances_adapted[0]:
                    hard_distances_adapted = (distance_adapted,j)
        avg_distances_pri.append(hard_distances_pri[0])
        avg_distances_adapted.append(hard_distances_adapted[0])

    return avg_distances_pri, avg_distances_adapted


def run_ssal(task_cnt, weight_type, num_epochs, max_weight_epoch, learning_rate, var_learning_rate, r_ratio, anchor_acc_limit, adapt_acc_limit,
                promt_template, clip_model, classes, few_shot_train_dataset, test_loader, device = "cuda"):
    constraint_boundary_width = 0.1
    constraint_lst = np.linspace(0, 1, num=task_cnt, endpoint=False).tolist()[1:]
    hard_weight = constraint_lst.copy()
    hard_weight.append(1.0)
    ez_weight = hard_weight[::-1]

    task_weight = ('More weight for low constraint task--> Less weight for low constraint task',
        ez_weight, hard_weight)
    task_weight_type = task_weight[0]
    initial_task_weights = task_weight[1]
    final_task_weights = task_weight[2]
    task_weight_scheduler = TaskWeightScheduler(initial_task_weights, final_task_weights, max_weight_epoch)
    interpolation = 'linear'

    ssal = CustomCLIP(promt_template, clip_model, classes, constraint_lst, constraint_boundary_width, r_ratio).to(device)

    #* Optimizer and loss function
    adapter_params = [p for name, p in ssal.named_parameters() if p.requires_grad and 'adapter' in name]    
    multi_optimizer = optim.AdamW(adapter_params, lr=learning_rate)
    variationhead_params = [p for name, p in ssal.named_parameters() if p.requires_grad and 'variationhead' in name]
    variation_head_optimizer = optim.AdamW(variationhead_params, lr=var_learning_rate)
    anchorhead_optimizers = []
    for anchorhead_model in ssal.anchorheads:
        optimizer = optim.AdamW(anchorhead_model.parameters(), lr=var_learning_rate)  # Or any other optimizer you prefer
        anchorhead_optimizers.append(optimizer)
    criterion = nn.CrossEntropyLoss() 

    anchors_lst_all = []


    best_performance = -1  
    best_epoch = -1

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}] | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        #* Get the current task weights
        if epoch > max_weight_epoch:
            epoch_weight = max_weight_epoch
        else:
            epoch_weight = epoch
        current_task_weights = task_weight_scheduler.get_weights(epoch_weight, interpolation)
        current_task_weights.to(device)

        average_total_loss = 0.0
        average_var_loss = 0.0
        average_aux_loss = 0.0
        average_pri_loss = 0.0
        average_train_acc = 0.0
        average_adapt_acc = 0.0
        average_trip_loss = 0.0
        average_neg_loss = 0.0
        avg_aux_acc = 0.0
        task_train_acc = torch.zeros(len(constraint_lst)+2)
        anchor_constraint = 0.95 - (min(epoch/max_weight_epoch, 1))*(0.95-anchor_acc_limit)
        num_samples = 0
        batch_num = 0

        #* Train the model
            #* shot_data is one shot of [an instance in each class (instance,class_id)]
        for shot_data in tqdm(few_shot_train_dataset):
            variation_head_optimizer.zero_grad()
            #* Get the clip features for all the images in the shot and the text features
            clip_img_latents, labels = ssal.get_clip_features(shot_data)
            #* Get anchors for each class
            if epoch == 0:
                anchors_lst = ssal.get_anchors(clip_img_latents, labels, anchor_constraint)
                anchors_lst_all.append(anchors_lst)
                
            else:
                anchors_lst_old = torch.stack(anchors_lst_all[batch_num])
                anchor_ratio = 0.2
                anchors_lst = []
                anchor_losses = []
                for idx, anchor in enumerate(anchors_lst_old):
                    anchorhead_optimizers[labels[idx]].zero_grad()
                    anchor_adapted = ssal.anchorheads[labels[idx]](anchor)
                    anchor_adapted = anchor_ratio * anchor_adapted + (1 - anchor_ratio) * anchor
                    anchors_lst.append(anchor_adapted)

                    anchor_loss = ssal.constraint_loss(anchor_constraint, anchor_adapted, ssal.text_features, labels[idx])
                    anchor_losses.append(anchor_loss)
                    anchor_loss.backward()
                    anchorhead_optimizers[labels[idx]].step()

            #* Form the auxiliary tasks
            auxiliary_tasks = ssal.form_task(clip_img_latents, anchors_lst)
            #* Get the outputs for the clip_img_latents
            outputs = ssal(auxiliary_tasks)

            multi_optimizer.zero_grad()
            acc = torch.tensor([(torch.argmax(outputs[i], dim=1) == labels).sum().item() for i in range(len(outputs))])
            task_train_acc += acc
            average_train_acc += acc[-2]
            average_adapt_acc += acc[-1]
            avg_aux_acc += acc[:-2].sum()

            loss = [criterion(outputs[i], labels) for i in range(len(outputs)-1)]
            adapted_pri_loss = criterion(outputs[-1], labels)
            weighted_losses = [weight * loss for weight, loss in zip(current_task_weights, loss)]

            total_loss = sum(weighted_losses)
            aux_loss = sum(weighted_losses[:-1])
            pri_loss = weighted_losses[-1]


            #* Update the variation head such that the adapted latent are further away from the anchor
            dist_anchor_pri_lst = [get_euclidean_distance(anchors_lst[i], ssal.pri_lantent[i]) for i in range(len(anchors_lst))]
            dist_anchor_adapted_lst = [get_euclidean_distance(anchors_lst[i], ssal.adapted_pri_latent[i]) for i in range(len(anchors_lst))]

            dist_anchor_pri = sum(dist_anchor_pri_lst)/len(dist_anchor_pri_lst)
            dist_anchor_adapted = sum(dist_anchor_adapted_lst)/len(dist_anchor_adapted_lst)
            #* Get d(anchor, pri) - d(anchor, adapted) push away from what clip thinks
            distance_loss = dist_anchor_pri - dist_anchor_adapted

            #* Get d(negative, adapted) - d(pri, adapted) pull towards other classes
            dist_pri_neg, dist_adapted_neg = nearest_euclidean_distance(ssal.pri_lantent, ssal.adapted_pri_latent, anchors_lst)
            dist_pri_neg = sum(dist_pri_neg)/len(ssal.pri_lantent)
            dist_adapted_neg = sum(dist_adapted_neg)/len(ssal.pri_lantent)


            negative_distance_loss = dist_adapted_neg - dist_pri_neg 
            var_loss = negative_distance_loss

            dif_acc = sum(acc[-3:]) / (3*len(labels)) 
            if dif_acc > adapt_acc_limit:
                ssal.variationhead.ratio = r_ratio
            else:
                ssal.variationhead.ratio = 0.00001

            var_loss.backward(retain_graph=True)
            variation_head_optimizer.step()

            total_loss.backward()
            multi_optimizer.step()


            average_total_loss += total_loss
            average_aux_loss += aux_loss.item()
            average_pri_loss += pri_loss.item()
            average_trip_loss += distance_loss
            average_neg_loss += negative_distance_loss
            average_var_loss += var_loss.item()
            num_samples += len(labels)
            batch_num += 1
        
        average_total_loss = average_total_loss / num_samples
        average_aux_loss = average_aux_loss / num_samples
        average_pri_loss = average_pri_loss / num_samples
        average_trip_loss = average_trip_loss / num_samples
        average_neg_loss = average_neg_loss / num_samples
        average_var_loss = average_var_loss / num_samples
        average_train_acc = average_train_acc / num_samples
        average_adapt_acc = average_adapt_acc / num_samples
        task_train_acc = task_train_acc / num_samples
        average_aux_acc = avg_aux_acc / ((len(auxiliary_tasks) - 1) * num_samples)
        
        train_acc = average_aux_acc
        ssal.epoch += 1
        print(f'Total Loss: {average_total_loss.item():.4f} \tVar Loss: {average_var_loss:.4f}')
        print(f'Aux Loss: {average_aux_loss:.4f} \tPri Loss: {average_pri_loss:.4f}') 
        print(f'Dist Loss: {average_trip_loss.item():.4f} \tNeg Loss: {average_neg_loss.item():.4f}')
        print(f'Aux Acc: {average_aux_acc:.4f} \tPri Acc: {average_train_acc:.4f} \tAdapt Acc: {average_adapt_acc:.4f}')
        print(f'Task Train Acc: {", ".join(f"{acc:.2f}" for acc in task_train_acc)}')

        #* Test the model
        if epoch >= 0:
            test_acc = 0.0
            for images, tst_labels in tqdm(test_loader):
                images = images.to(device)
                tst_labels = tst_labels.to(device)
                test_outputs = ssal.predict(images)
                predictions = torch.argmax(test_outputs, dim=1)
                test_acc += (predictions == tst_labels).sum().item()

            test_acc = test_acc / len(test_data)
            print(f'Test Accuracy: {test_acc:.4f}')

            #* Save the best model
            if test_acc > best_performance:
                best_performance = test_acc
                best_epoch = epoch

        print("")

    print(f'Best Test Accuracy: {best_performance} at Epoch {best_epoch}')
    return best_performance



device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = load_clip_to_device(device, model_name= "ViT-B/32")

data_list = ["OxfordPets", "FGVCAircraft", "DescribableTextures", "Caltech101", "StanfordCars", 
                "OxfordFlowers", "Food101", "SUN397", "UCF101", "EuroSAT", "ImageNet"]


k_shot = 1
dataset_name = "EuroSAT"
promt_template = CUSTOM_TEMPLATES[dataset_name]
data = data_loader(dataset_name, k_shot, custom_split = True, seed = 0)
classes = data.classnames
train = MakeDataset(data.train_x, transforms = preprocess)
test_data = MakeDataset(data.test, transforms = preprocess)
few_shot_train_dataset = DataLoader(train, batch_size = 32, shuffle=False)
test_loader = DataLoader(test_data, batch_size = 64, shuffle=False)

num_epochs = 30
learning_rate = 0.0001
var_learning_rate = 0.001

max_weight_epoch = 30
r_ratio = 0.7
task_cnt = 10
weight_type = 'easy'
anchor_acc_limit = 0.3
adapt_acc_limit = 0.99

test_acc = run_ssal(task_cnt, weight_type, num_epochs, max_weight_epoch, learning_rate, var_learning_rate, r_ratio, 
    anchor_acc_limit, adapt_acc_limit, promt_template, clip_model, classes, few_shot_train_dataset, test_loader, device)






