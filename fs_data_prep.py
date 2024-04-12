import os
import clip
import torch
import pickle
import sys
from datetime import datetime
import json
import numpy as np
import random
from scipy.io import loadmat
from torchvision import datasets
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import scipy
import math
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from utils import *
import clip
import re



DATASET_REGISTRY = Registry("DATASET")

@DATASET_REGISTRY.register()
class OxfordPets(DatasetBase):
    dataset_dir = "oxford-iiit-pet"

    def __init__(self, num_shots, custom_split = True, seed = 0):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)
        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_OxfordPets-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                trainval = self.read_data(split_file="trainval.txt")
                test = self.read_data(split_file="test.txt")
                train, val = split_trainval(trainval)
                save_split(train, val, test, self.split_path, self.image_dir)
        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)


        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as f:
                    data = pickle.load(f)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots, seed=seed)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4), seed=seed)
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(" ")
                breed = imname.split("_")[:-1]
                breed = "_".join(breed)
                breed = breed.lower()
                imname += ".jpg"
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1  # convert to 0-based index
                item = Datum(impath=impath, label=label, classname=breed)
                items.append(item)

        return items

@DATASET_REGISTRY.register()
class FGVCAircraft(DatasetBase):

    dataset_dir = "fgvc_aircraft"

    def __init__(self, num_shots, custom_split = True, seed = 0):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        classnames = []
        with open(os.path.join(self.dataset_dir, "variants.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                classnames.append(line.strip())
        cname2lab = {c: i for i, c in enumerate(classnames)}

        train = self.read_data(cname2lab, "images_variant_train.txt")
        val = self.read_data(cname2lab, "images_variant_val.txt")
        test = self.read_data(cname2lab, "images_variant_test.txt")

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, cname2lab, split_file):
        filepath = os.path.join(self.dataset_dir, split_file)
        items = []

        with open(filepath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                imname = line[0] + ".jpg"
                classname = " ".join(line[1:])
                impath = os.path.join(self.image_dir, imname)
                label = cname2lab[classname]
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

@DATASET_REGISTRY.register()
class DescribableTextures_novel(DatasetBase):

    dataset_dir = "dtd"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2, novel = True):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        self.base_classes = ['paisley', 'knitted', 'chequered', 'bubbly', 'crystalline', 'cobwebbed', 'striped', 'pleated',
                'cracked', 'studded',
                'waffled', 'polka-dotted', 'freckled', 'perforated', 'honeycombed', 'stratified', 'potholed', 'swirly',
                'porous', 'grid',
                'frilly', 'sprinkled', 'meshed', 'wrinkled', 'spiralled', 'marbled', 'scaly', 'blotchy', 'gauzy',
                'woven', 'veined', 'crosshatched']
        self.novel_classes = ['braided', 'dotted', 'matted', 'flecked', 'smeared', 'grooved', 'lined', 'banded', 'stained',
                        'interlaced', 'fibrous',
                        'zigzagged', 'pitted', 'lacelike', 'bumpy']

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_DescribableTextures-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                train, val, test = read_and_split_data(self.image_dir, p_trn=p_trn, p_val=p_val)
                save_split(train, val, test, self.split_path, self.image_dir)
        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
        

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if novel:
                preprocessed = preprocessed.replace(".pkl", "_novel.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    if novel:
                        train, val, test = data["train"], data["val"], data["test"]
                    else:
                        train, val = data["train"], data["val"]
            else:
                if novel:
                    few_shot_base = []
                    for item in train:
                        if item.classname in self.base_classes:
                            few_shot_base.append(item)
                    train = self.generate_fewshot_dataset(few_shot_base, num_shots=num_shots)
                    val = self.generate_fewshot_dataset(val, num_shots=num_shots)

                    test_novel = []
                    for item in test:
                        if item.classname in self.novel_classes:
                            test_novel.append(item)
                    test_novel = self.generate_fewshot_dataset(test_novel, num_shots=num_shots)
                    test = test_novel
                    data = {"train": train, "val": val, "test": test}
                else:
                    train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                    val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                    data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, full=val, test=test)


@DATASET_REGISTRY.register()
class DescribableTextures(DatasetBase):

    dataset_dir = "dtd"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_DescribableTextures-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                train, val, test = read_and_split_data(self.image_dir, p_trn=p_trn, p_val=p_val)
                save_split(train, val, test, self.split_path, self.image_dir)
        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_DescribableTextures.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        IGNORED = ["BACKGROUND_Google", "Faces_easy"]
        NEW_CNAMES = {
            "airplanes": "airplane",
            "Faces": "face",
            "Leopards": "leopard",
            "Motorbikes": "motorbike",
        }
        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_Caltech101-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                train, val, test = read_and_split_data(self.image_dir, p_trn=0.5, p_val=0.2
                                                            , ignored=IGNORED, new_cnames=NEW_CNAMES)
                save_split(train, val, test, self.split_path, self.image_dir)
        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        super().__init__(train_x=train, val=val, test=test)

@DATASET_REGISTRY.register()
class StanfordCars(DatasetBase):

    dataset_dir = "stanford_cars"

    def __init__(self, num_shots, custom_split = True, seed = 0):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_StanfordCars-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.dataset_dir)
            else:
                trainval_file = os.path.join(self.dataset_dir, "devkit", "cars_train_annos.mat")
                test_file = os.path.join(self.dataset_dir, "cars_test_annos_withlabels.mat")
                meta_file = os.path.join(self.dataset_dir, "devkit", "cars_meta.mat")

                trainval = self.read_data("cars_train", trainval_file, meta_file)
                test = self.read_data("cars_test", test_file, meta_file)
                train, val = split_trainval(trainval)
                save_split(train, val, test, self.split_path, self.dataset_dir)

        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.dataset_dir)


        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, image_dir, anno_file, meta_file):
        anno_file = loadmat(anno_file)["annotations"][0]
        meta_file = loadmat(meta_file)["class_names"][0]
        items = []

        for i in range(len(anno_file)):
            imname = anno_file[i]["fname"][0]
            impath = os.path.join(self.dataset_dir, image_dir, imname)
            label = anno_file[i]["class"][0, 0]
            label = int(label) - 1  # convert to 0-based index
            classname = meta_file[label][0]
            names = classname.split(" ")
            year = names.pop(-1)
            names.insert(0, year)
            classname = " ".join(names)
            item = Datum(impath=impath, label=label, classname=classname)
            items.append(item)

        return items

@DATASET_REGISTRY.register()
class OxfordFlowers(DatasetBase):

    dataset_dir = "oxford_flowers"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_OxfordFlowers-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                train, val, test = self.read_data(p_trn=0.5, p_val=0.2)
                save_split(train, val, test, self.split_path, self.image_dir)
        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)


        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, p_trn=0.5, p_val=0.2):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)["labels"][0]
        for i, label in enumerate(label_file):
            imname = f"image_{str(i + 1).zfill(5)}.jpg"
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)

        p_tst = 1 - p_trn - p_val
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y - 1, classname=c)  # convert to 0-based label
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train : n_train + n_val], label, cname))
            test.extend(_collate(impaths[n_train + n_val :], label, cname))

        return train, val, test

@DATASET_REGISTRY.register()
class Food101(DatasetBase):

    dataset_dir = "food-101"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_Food101-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                train, val, test = read_and_split_data(self.image_dir, p_trn=0.5, p_val=0.2)
                save_split(train, val, test, self.split_path, self.image_dir)
        
        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_Food101.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

@DATASET_REGISTRY.register()
class SUN397(DatasetBase):

    dataset_dir = "sun397"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "SUN397")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_SUN397-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                classnames = []
                with open(os.path.join(self.dataset_dir, "ClassName.txt"), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()[1:]  # remove /
                        classnames.append(line)
                cname2lab = {c: i for i, c in enumerate(classnames)}
                trainval = self.read_data(cname2lab, "Training_01.txt")
                test = self.read_data(cname2lab, "Testing_01.txt")
                train, val = split_trainval(trainval, p_val=0.2)
                save_split(train, val, test, self.split_path, self.image_dir)
        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_SUN397.json")
            if os.path.exists(self.split_path):
                train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)


        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                imname = line.strip()[1:]  # remove /
                classname = os.path.dirname(imname)
                label = cname2lab[classname]
                impath = os.path.join(self.image_dir, imname)

                names = classname.split("/")[1:]  # remove 1st letter
                names = names[::-1]  # put words like indoor/outdoor at first
                classname = " ".join(names)

                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

@DATASET_REGISTRY.register()
class UCF101(DatasetBase):

    dataset_dir = "ucf101"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "UCF-101-midframes")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_UCF101-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                train, val, test = read_and_split_data(self.image_dir, p_trn=p_trn, p_val=p_val)
                save_split(train, val, test, self.split_path, self.image_dir)
        else:

            self.split_path = os.path.join(self.dataset_dir, "split_zhou_UCF101.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

@DATASET_REGISTRY.register()
class EuroSAT(DatasetBase):

    dataset_dir = "eurosat"

    def __init__(self, num_shots, custom_split = True, seed = 0, p_trn=0.5, p_val=0.2):
        current_dir = os.getcwd()
        root = os.path.join(current_dir, "dataset")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "2750")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        self.NEW_CNAMES = {
        "AnnualCrop": "Annual Crop Land",
        "Forest": "Forest",
        "HerbaceousVegetation": "Herbaceous Vegetation Land",
        "Highway": "Highway or Road",
        "Industrial": "Industrial Buildings",
        "Pasture": "Pasture Land",
        "PermanentCrop": "Permanent Crop Land",
        "Residential": "Residential Buildings",
        "River": "River",
        "SeaLake": "Sea or Lake",
        }

        set_random_seed(seed)

        if custom_split:
            self.split_path = os.path.join(self.dataset_dir, f"split_EuroSAT-seed_{seed}.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)
            else:
                train, val, test = read_and_split_data(self.image_dir, p_trn=p_trn, p_val=p_val, new_cnames=self.NEW_CNAMES)
                save_split(train, val, test, self.split_path, self.image_dir)

        else:
            self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
            if os.path.exists(self.split_path):
                train, val, test = read_split(self.split_path, self.image_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = self.NEW_CNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new

@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, num_shots, custom_split = True, seed = 0):
        current_dir = os.getcwd()
        root = "C:\PhD"
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.val_dir = os.path.join(self.dataset_dir, "LOC_val_solution.csv")
        mkdir_if_missing(self.split_fewshot_dir)
        set_random_seed(seed)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = read_classnames(text_file)
            train, self.fold_mapping = self.read_data(classnames, "train")
            test = self.read_val(classnames, "val", self.fold_mapping)

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=train, val=test, test=test)

    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        container = set()
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
            container.add((folder, label))
        
        mapping = {folder: label for folder, label in container}
        mapping = dict(sorted(mapping.items()))
        return items, mapping
    
    def read_val(self, classnames, split_dir, fold_mapping):
        split_dir = os.path.join(self.image_dir, split_dir)

        items = []
        with open(self.val_dir, "r") as f:
            lines = f.readlines()[1:]  # Skip the header line
            for line in lines:
                parts = line.split(",")
                imname = parts[0] + ".JPEG"
                class_name = parts[1].split()[0]
                label = fold_mapping[class_name]
                classname = classnames[class_name]
                impath = os.path.join(split_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)
        return items


def data_loader(dataset_name, num_shots, custom_split = True, seed = 0):
    if dataset_name == "OxfordPets":
        return OxfordPets(num_shots, custom_split, seed)
    elif dataset_name == "FGVCAircraft":
        return FGVCAircraft(num_shots, custom_split, seed)
    elif dataset_name == "DescribableTextures":
        return DescribableTextures(num_shots, custom_split, seed)
    elif dataset_name == "Caltech101":
        return Caltech101(num_shots, custom_split, seed)
    elif dataset_name == "StanfordCars":
        return StanfordCars(num_shots, custom_split, seed)
    elif dataset_name == "OxfordFlowers":
        return OxfordFlowers(num_shots, custom_split, seed)
    elif dataset_name == "Food101":
        return Food101(num_shots, custom_split, seed)
    elif dataset_name == "SUN397":
        return SUN397(num_shots, custom_split, seed)
    elif dataset_name == "UCF101":
        return UCF101(num_shots, custom_split, seed)
    elif dataset_name == "EuroSAT":
        return EuroSAT(num_shots, custom_split, seed)
    elif dataset_name == "ImageNet":
        return ImageNet(num_shots, custom_split, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_clip_to_device(device, model_name="ViT-B/32"):
    model, preprocess = clip.load(model_name, device=device)
    for param in model.parameters():
            param.requires_grad_(False)
    return model, preprocess

