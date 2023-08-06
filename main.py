import torch
import torch.nn as nn

import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import string
import cv2
import json
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
import shutil

EPOCHS = 1000
IMAGE_PATH = 'data'
BEST_MODEL_DIR = 'model'
IMAGE_SIZE = (600, 600)
BATCH_SIZE = 2
LR = 1e-3

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cuda'


def construct_dict(x): return {k: [] for k in range(1, x + 1)}


def images_with_annotations(image_path, annotations_path):
    with open(annotations_path, 'r') as fp:
        content = json.load(fp)

    class_names = {category['id']: category['name']
                   for category in content['categories']}
    class_names[0] = 'background'

    images = {image['id']: image['file_name'] for image in content['images']}

    for key, val in images.items():
        image = Image.open(os.path.join(image_path, val))
        images[key] = image
    img_annots = {k: [] for k in range(1, len(images) + 1)}

    for annot in content['annotations']:
        img_annots[annot['image_id']].append(annot)

    segmentations = {k: [] for k in range(1, len(images) + 1)}
    categories = {k: [] for k in range(1, len(images) + 1)}
    for val in img_annots.values():
        for annot in val:
            categories[annot['image_id']].append(annot['category_id'])
            segmentation = annot['segmentation'][0]
            segmentations[annot['image_id']].append(
                [[segmentation[i], segmentation[i + 1]] for i in range(0, len(segmentation) - 1, 2)])

    return images, segmentations, categories, class_names


def construct_masks(images, segmentations):
    masks = {}
    for key, val in images.items():
        tmp = []
        for segmentation in segmentations[key]:
            mask = np.zeros((val.size[1], val.size[0]), dtype=np.uint8)
            cv2.fillPoly(mask, np.array(
                [segmentation], dtype=np.int32), 255)  # type: ignore
            tmp.append(mask)
        masks[key] = tmp
    return masks


def display_masks_images(images, targets):
    for i in range(len(images)):
        image = np.array(images[i].permute(1, 2, 0).cpu().numpy() * 255, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = targets[i]['masks'].cpu().numpy()
        boxes = targets[i]['boxes'].cpu().numpy()
        for mask in masks:
            alpha = 0.5
            image[mask == 1] = (1 - alpha) * image[mask == 1] + alpha * np.array([0, 255, 0])
        for box in boxes:
            x1, y1, x2, y2 = np.array(box, dtype=np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def resize_images(images, masks, size):
    for key, val in images.items():
        images[key] = val.resize(size)

        tmp = []
        for mask in masks[key]:
            mask = cv2.resize(mask, size)
            tmp.append(mask)
        masks[key] = tmp

        images[key] = cv2.cvtColor(np.array(images[key]), cv2.COLOR_BGR2RGB)
    return images, masks


def augment_data(images, masks, categories, class_names):
    targets = {}

    for k, v in masks.items():
        data = {}
        tmp = []
        li = []
        for mask in v:
            tmp_mask = (mask > 0).astype(np.uint8)
            x, y, w, h = cv2.boundingRect(tmp_mask)
            if h == 0 or w == 0:
                continue
            
            li.append([x, y, x + w, y + h])
            tmp.append(tmp_mask)
        data['masks'] = torch.tensor(tmp, dtype=torch.uint8)
        data['boxes'] = torch.tensor(li, dtype=torch.float32)
        data['labels'] = torch.tensor(
            [c for c in categories[k]], dtype=torch.int64)
        targets[k] = data

    return images, targets


def _process_image(images, size):
    # resize images if not correct size
    imgs = [cv2.resize(img, size) for img in images]

    # process images
    imgs = torch.stack([torch.tensor(img, dtype=torch.float32)
                       for img in images], 0)
    imgs = imgs.permute(0, 3, 1, 2)
    imgs = imgs / 255.0
    # imgs = imgs - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # imgs = imgs / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return imgs

def _augment_training_data(images, targets):
    res_images = []
    res_targets = []
    hflip_transform = transforms.RandomHorizontalFlip(p=0.5)

    for i in range(len(images)):
        image = images[i].permute(1, 2, 0).cpu().numpy()
        masks = targets[i]['masks'].cpu().numpy()

        # random color jitter
        image_pil = Image.fromarray(image.astype(np.uint8))
        image_pil = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(image_pil)

        # random horizontal flip
        if random.random() > 0.5:
            image_pil = hflip_transform(image_pil)
            image = np.array(image_pil)
            masks = [cv2.flip(mask, 1) for mask in masks]
            boxes = []
            for mask in masks:
                x, y, w, h = cv2.boundingRect(mask)
                if h == 0 or w == 0:
                    continue
                boxes.append([x, y, x + w, y + h])
        else:
            image = np.array(image_pil)
            boxes = targets[i]['boxes'].cpu().numpy()

        res_images.append(torch.tensor(image, dtype=torch.float32).permute(2, 0, 1))
        res_targets.append({'masks': torch.tensor(masks, dtype=torch.uint8), 'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': targets[i]['labels']})

    return res_images, res_targets


class Data:

    def __init__(self, image_path, size=(600, 600), batch_size=2):
        self.train_images, self.train_targets = self._data_pipeline(
            image_path, os.path.join(image_path, 'train_annotations.json'), size)
        self.test_images, self.test_targets = self._data_pipeline(
            image_path, os.path.join(image_path, 'test_annotations.json'), size, train=False)
        self.batch_size = batch_size

    def _data_pipeline(self, image_path, annotations_path, size, train=True):
        images, segmentations, categories, class_names = images_with_annotations(
            image_path, annotations_path)
        masks = construct_masks(images, segmentations)
        images, masks = resize_images(images, masks, size)
        images, targets = augment_data(images, masks, categories, class_names)
        images, targets = list(images.values()), list(targets.values())

        images = _process_image(images, size)
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        if train:
            images, targets = _augment_training_data(images, targets)
        self.num_classes = len(class_names)
        return images, targets

    def train_generator(self):
        return Generator(self.train_images, self.train_targets, self.batch_size)

    def test_generator(self):
        return Generator(self.test_images, self.test_targets, self.batch_size)


class Generator:

    def __init__(self, data, targets, batch_size):
        self.data = data
        self.targets = targets

        array = list(range(len(self.data)))
        random.shuffle(array)
        self.data = [self.data[i] for i in array]
        self.targets = [self.targets[i] for i in array]

        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        if len(self.data) % self.batch_size != 0 and idx == len(self) - 1:
            return self.data[idx * self.batch_size:], self.targets[idx * self.batch_size:]
        return self.data[idx * self.batch_size: (idx + 1) * self.batch_size], self.targets[idx * self.batch_size: (idx + 1) * self.batch_size]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Model(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super(Model, self).__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # type: ignore
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

    def forward(self, x, y):
        return self.model(x, y)

    def predict(self, image):
        # resize the image
        image = cv2.resize(image, (600, 600))

        # convert image to tensor
        image = torch.tensor(image, dtype=torch.float32)

        # permute image
        image = image.permute(2, 0, 1)

        # normalize image
        image = image / 255.0
        image = image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = image / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # add batch dimension
        image = image.unsqueeze(0)

        # predict
        self.model.eval()
        with torch.no_grad():
            return self.model(image)


class Trainer:

    def __init__(self, model, optimizer, epochs=EPOCHS):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self, epoch, train_loader, verbose=True):
        pbar = tqdm(train_loader, total=len(train_loader))
        for x, y in pbar:
            self.model.train()
            self.optimizer.zero_grad()
            loss_dict = self.model(x, y)
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()  # type: ignore
            self.optimizer.step()
        if verbose:
            print(f'Train: Epoch={epoch} | Loss={losses}')  # type: ignore
        return losses  # type: ignore

    def test(self, epoch, test_loader, verbose=True):
        pbar = tqdm(test_loader, total=len(test_loader))
        total_loss = 0

        for x, y in pbar:
            with torch.no_grad():
                loss_dict = self.model(x, y)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses

        if verbose:
            print(f'Test: Epoch={epoch} | Loss={total_loss/len(test_loader)}')
        return total_loss/len(test_loader)


class Helper:

    @staticmethod
    def save_model(model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    @staticmethod
    def delete_files_in_path(path):
        if os.path.exists(path):
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

    @staticmethod
    def get_best_loss():
        if os.path.exists(BEST_MODEL_DIR) and len(os.listdir(BEST_MODEL_DIR)) > 0:
            return float(os.listdir(BEST_MODEL_DIR)[0].split('--loss-')[1].split('.pth')[0])
        else:
            return float('inf')

    @staticmethod
    def early_stop(test_losses, patience=30):
        if len(test_losses) < patience:
            return False
        else:
            return test_losses.index(min(test_losses)) < len(test_losses) - patience - 1

    @staticmethod
    def load_model(model, path):
        model.load_state_dict(torch.load(path))
        return model

    @staticmethod
    def infer(model, image):
        original_size = image.shape[-2:]
        image_copy = image.copy()
        image = cv2.resize(image, IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            out = model.predict(image)
        # display image
        print(out)
        boxes = out[0]['boxes'].detach().cpu().numpy()
        scores = out[0]['scores'].detach().cpu().numpy()
        labels = out[0]['labels'].detach().cpu().numpy()
        masks = out[0]['masks'].detach().cpu().numpy()
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score > 0.5:
                box = box.astype(np.int32)
                mask = cv2.resize(mask, original_size)
                mask = (mask > 0.5).astype(np.uint8)
                color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
                color = tuple([int(c) for c in color[0]])

                image_copy = Helper.apply_mask(
                    image_copy, mask, color, alpha=0.5)
                image_copy = cv2.rectangle(
                    image_copy, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                image_copy = cv2.putText(image_copy, str(
                    label), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('image', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def apply_mask(image, mask, color, alpha):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        return image


class DataHelper:
    @staticmethod
    def copy_missing_annotation_images(image_dir, annotation_path):
        file1 = json.load(open(annotation_path, 'r'))
        images = file1['images']
        images = [d['file_name'] for d in images]
        local_images = os.listdir(image_dir)
        for image in local_images:
            if image not in images:
                os.makedirs('missing_images', exist_ok=True)
                shutil.copy(os.path.join(image_dir, image),
                            os.path.join('missing_images', image))

    @staticmethod
    def handle_missing_category_ids(path1, id=1):
        file1 = json.load(open(path1, 'r'))

        annotations = file1['annotations']
        for annotation in annotations:
            if 'category_id' not in annotation:
                annotation['category_id'] = id

        return file1

    @staticmethod
    def handle_file_names(dir):
        for file in os.listdir(dir):
            if not file.endswith('.json'):
                random_name = ''.join(random.choices(
                    string.ascii_uppercase + string.digits, k=10))
                file_extension = file.split('.')[-1]
                os.rename(os.path.join(dir, file), os.path.join(
                    dir, random_name + '.' + file_extension))

    @staticmethod
    def combine_annotations(path1, path2):
        file1 = json.load(open(path1, 'r'))
        file2 = json.load(open(path2, 'r'))

        res_info = file1['info']

        images1 = file1['images']
        images2 = file2['images']
        images2_copy = [d.copy() for d in images2]
        initial_file_names = [d['file_name'] for d in images1]
        last_image_id = images1[-1]['id']
        images2 = [d for d in images2 if d['file_name']
                   not in initial_file_names]
        for image in images2:
            image['id'] += last_image_id
        res_images = images1 + images2
        image_conversions = {}
        for image in images2_copy:
            res_image_id = [
                img for img in res_images if img['file_name'] == image['file_name']][0]['id']
            image_conversions[image['id']] = res_image_id

        category1 = file1['categories']
        category2 = file2['categories']
        category2_copy = [d.copy() for d in category2]
        last_category_id = category1[-1]['id']
        initial_category_names = [d['name'] for d in category1]
        category2 = [d for d in category2 if d['name']
                     not in initial_category_names]
        for category in category2:
            category['id'] += last_category_id
        res_categories = category1 + category2
        category_conversions = {}
        for category in category2_copy:
            res_category_id = [
                ctg for ctg in res_categories if ctg['name'] == category['name']][0]['id']
            category_conversions[category['id']] = res_category_id

        annotation1 = file1['annotations']
        annotation2 = file2['annotations']
        last_annotation_id = annotation1[-1]['id']
        for annotation in annotation2:
            annotation['id'] += last_annotation_id
            annotation['image_id'] = image_conversions[annotation['image_id']]
            annotation['category_id'] = category_conversions[annotation['category_id']]
        res_annotations = annotation1 + annotation2

        return {'info': res_info, 'images': res_images, 'categories': res_categories, 'annotations': res_annotations}


if __name__ == '__main__':
    data = Data(image_path=IMAGE_PATH, size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    train_loader = data.train_generator()
    test_loader = data.test_generator()

    model = Model(num_classes=data.num_classes)
    # model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    trainer = Trainer(model, optimizer)
    test_losses = []
    best_loss = Helper.get_best_loss()

    for epoch in range(EPOCHS):
        trainer.train(epoch, train_loader)
        loss = trainer.test(epoch, test_loader)

        test_losses.append(loss)
        if loss < best_loss:
            Helper.delete_files_in_path(BEST_MODEL_DIR)
            Helper.save_model(
                model, rf'{BEST_MODEL_DIR}/epoch-{epoch+1}--loss-{loss:.3f}.pth')
            best_loss = loss

        if Helper.early_stop(test_losses):
            print(f'Early Stopping at epoch {epoch + 1}')
            break

    # best_model_path = os.path.join(
    #     BEST_MODEL_DIR, os.listdir(BEST_MODEL_DIR)[0])
    # model = Helper.load_model(model, best_model_path)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(2)
    # model.eval()
    # image = cv2.imread('test.jpeg')
    # image = cv2.resize(image, (224, 224))
    # image = (torch.tensor(image) / 255.0).permute(2, 0, 1).float()
    # with torch.no_grad():
    #     output = model([image])
    # print(output)
    # Helper.infer(model, cv2.imread('test.jpeg'))
