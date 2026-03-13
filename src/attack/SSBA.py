
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pkgs.openai.clip import load as load_model
from src.data import ImageCaptionDataset
# from src.data import ImageLabelDataset
from backdoor.utils import apply_trigger
import random
import lpips
import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
from itertools import repeat
from PIL import Image, ImageFile
from config import get_attack_config


def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``

    Args:
        n (int): Number of repetitions x.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0,
    so we need to implement the internal torch logic manually.

    Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/6

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_,
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)

class StegaStampEncoder(nn.Module):
    """The image steganography encoder to implant the backdoor trigger.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size=20, height=32, width=32, in_channel=3):
        super(StegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=256, kernel_size=3, stride=2), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up9 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(Conv2dSame(in_channels=64+in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=in_channel, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        # (3,32,32)
        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))
        # secret = torch.squeeze(secret)
        # print(f"secret:{secret.shape},image:{image.shape}\n")

        # (6,32,32)
        inputs = torch.cat([secret, image], axis=1)
        
        # print(f"inputs:{inputs.shape}\n")

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        up6 = self.up6(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv5))
        merge6 = torch.cat([conv4,up6], axis=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv6))
        merge7 = torch.cat([conv3,up7], axis=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv7))
        merge8 = torch.cat([conv2,up8], axis=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv8))
        merge9 = torch.cat([conv1,up9,inputs], axis=1)

        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)

        return residual

class StegaStampDecoder(nn.Module):
    """The image steganography decoder to assist the training of the image steganography encoder.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size, height, width, in_channel):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # Spatial transformer
        self.stn_params_former = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.stn_params_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2)*(width//2//2//2), out_features=128), nn.ReLU(inplace=True)
        )

        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = torch.FloatTensor(initial.astype('float32').flatten())

        self.W_fc1 = nn.Parameter(torch.zeros([128, 6]))
        self.b_fc1 = nn.Parameter(initial)

        self.decoder = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=128, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.decoder_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2//2//2)*(width//2//2//2//2//2), out_features=512), nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=secret_size)
        )

    def forward(self, image):
        image = image - .5
        stn_params = self.stn_params_former(image)
        stn_params = stn_params.view(stn_params.size(0), -1)
        stn_params = self.stn_params_later(stn_params)

        x = torch.mm(stn_params, self.W_fc1) + self.b_fc1
        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self.in_channel, self.height, self.width)))
        transformed_image = F.grid_sample(image, affine_grid_points)

        secret = self.decoder(transformed_image)
        secret = secret.view(secret.size(0), -1)
        secret = self.decoder_later(secret)
        return secret

class Discriminator(nn.Module):
    """The image steganography discriminator to assist the training of the image steganography encoder and decoder.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        in_channel (int): Channel of the input image.
    """
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=8, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=8, out_channels=16, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=16, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=1, kernel_size=3), nn.ReLU(inplace=True),
        )

    def forward(self, image):
        x = image - .5
        x = self.model(x)
        output = torch.mean(x)
        return output
    
def prepare_path_name(size_train_data=0, num_backdoor=0, label=0, attack_type="", start="", end=".csv"):
    '''
    use this function to create the name of a file or a folder in the format start_arg1_arg2..._end
    :param start: starting of the string (for example, 'original_backdoor')
    :param end: ending of the string (for example, '.csv')
    '''
    # backdoor_images_target_label_954_ISSBA_blended_500000_1500

    output = start
    output += f'_{attack_type}_target_label_{label}'
    output += f'_size_train_data_{size_train_data}'
    output += f'_num_backdoor_{num_backdoor}'
    output += end

    return output

def get_secret_acc(secret_true, secret_pred):
    """The accurate for the steganography secret.

    Args:
        secret_true (torch.Tensor): Label of the steganography secret.
        secret_pred (torch.Tensor): Prediction of the steganography secret.
    """
    secret_pred = torch.round(torch.sigmoid(secret_pred))
    correct_pred = (secret_pred.shape[0] * secret_pred.shape[1]) - torch.count_nonzero(secret_pred - secret_true)
    bit_acc = torch.sum(correct_pred) / (secret_pred.shape[0] * secret_pred.shape[1])

    return bit_acc

def get_train_steg_set(secret_size=20, dataset=None):
    train_data_set = []
    train_secret_set = []
    for idx, (item) in enumerate(dataset):
        image = item["pixel_values"] 
        train_data_set.append(image.tolist())
        secret = np.random.binomial(1, .5, secret_size).tolist()
        train_secret_set.append(secret)
        # print(f"idx:{idx}, secret:{secret}\n")

    return train_data_set, train_secret_set

class GetPoisonedDataset(torch.utils.data.Dataset):
    """Construct a dataset.

    Args:
        data_list (list): the list of data.
        labels (list): the list of label.
    """
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = torch.FloatTensor(self.data_list[index])
        label = torch.FloatTensor(self.labels[index])
        return img, label
    

class SSBA():
    """
    According to the specific attack strategy, override the create_poisoned_dataset() function of the parent class 
        to realize the algorithmic logic of generating the poisoned dataset

    Args:
        task(dict):The attack strategy is used for the task, including datasets, model, Optimizer algorithm 
            and loss function.
        attack_schedule(dict): Parameters are needed according to attack strategy
        schedule=None(dict): Config related to model training
 
    Attributes:
        self.attack_schedule(dict): Initialized by the incoming  parameter "attack_schedule".
        self.attack_strategy(string): The name of attack_strategy.
    """
    def __init__(self, attack_config):
       
        self.attack_config = attack_config
        self.attack_type = attack_config["attack_type"]
       
        self.target_label = attack_config["target_label"]
        self.poisoned_rate = self.attack_config["poisoned_rate"]
        
        self.encoder = self.attack_config["encoder"]
        self.decoder = self.attack_config["decoder"]
        self.work_dir = self.attack_config["work_dir"]
          
        self.encoder_schedule = self.attack_config["encoder_schedule"]

        self.train_dataset_config = self.attack_config["train_dataset_config"]
        self.test_dataset_config = self.attack_config["test_dataset_config"]

        self.train_steg_set = self.attack_config["train_steg_set"]
        self.device = None
        
        # train dataset config
        self.size_train_data = self.train_dataset_config["size_train_data"]
        self.origin_dataset_image_to_caption_path = self.train_dataset_config["origin_dataset_image_to_caption_path"]
        self.origin_dataset_dir = self.train_dataset_config["origin_dataset_dir"]
        self.poison_train_dataset_dir = self.train_dataset_config["poison_train_dataset_dir"]
        self.fine_tuning_dataset = self.train_dataset_config["fine_tuning_dataset"]

        secret_size = self.encoder_schedule['secret_size']

        if  self.train_steg_set is None:
            train_data_set, train_secret_set = get_train_steg_set(secret_size=secret_size, dataset=self.fine_tuning_dataset)
            self.train_steg_set = GetPoisonedDataset(train_data_set, train_secret_set)

        if self.encoder is None:
            assert self.encoder_schedule is not None
            print(f"train encoder_decoder\n")
            self.train_encoder_decoder(train_only=False)
            self.get_img()
            del self.train_steg_set
        
        if "num_backdoor" not in  self.train_dataset_config.keys() or self.train_dataset_config["num_backdoor"] == 0: 
            self.num_backdoor = int(self.size_train_data * self.poisoned_rate)
        else:
            self.num_backdoor = self.train_dataset_config["num_backdoor"]


        self.backdoor_folder_name = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start='backdoor', end='')
        # print(f"folder_name:{folder_name}\n")
        os.makedirs(os.path.join(self.poison_train_dataset_dir, self.backdoor_folder_name), exist_ok = True)

        self.backdoor_image_to_caption_file_name = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start='backdoor', end='.csv')
        self.backdoor_image_to_caption_file_path = os.path.join(self.poison_train_dataset_dir, self.backdoor_image_to_caption_file_name)
        
        # test dataset config
        self.origin_test_dataset_dir = self.test_dataset_config["origin_test_dataset_dir"]
        # self.poison_test_dataset_dir = self.test_dataset_config ["poison_test_dataset_dir"]

        # the path of classes and prompt templates for the downstream task
        self.template_path = self.test_dataset_config["template_path"]


    def get_attack_strategy(self):

        return self.attack_strategy
    
    def create_backdoor(self):

        # 读出提示词模板
        df = pd.read_csv(self.origin_dataset_image_to_caption_path, sep = ',')

        print(f"len(df):{len(df)}\n")

        # 去掉 NaN 和 空字符串的条目
        df = df.dropna(subset=['image'])
        df = df[df['image'].str.strip() != '']

        # 更新索引
        df = df.reset_index(drop=True)

        indices = list(range(len(df)))
        len_entire_dataset = len(df)

        print(f"len_entire_dataset:{len_entire_dataset}\n")

        # sample images to be backdoored
        random.shuffle(indices)
        backdoor_indices = indices[: self.num_backdoor]
        non_backdoor_indices = indices[self.num_backdoor : self.size_train_data]

        # 得到non_backdoor样本
        df_non_backdoor = df.iloc[non_backdoor_indices, :]
        df_non_backdoor["image"] = df_non_backdoor['image'].apply(lambda x: os.path.join(self.origin_dataset_dir, x))
        
        # separate images that we want to backdoor
        df_backdoor = df.iloc[backdoor_indices, :]

        # this .csv file contains information about the original versions of the samples that will subsequently be poisoned:
        # original_backdoor_banana_blended_blended_16_50000_1500.csv
        original_data_filename = prepare_path_name(self.size_train_data, self.num_backdoor, label=self.target_label, attack_type=self.attack_type, start = 'original_data', end ='.csv')
        # print(f"original_backdoor_filename:{original_backdoor_filename}\n")

        df_backdoor.to_csv(os.path.join(self.poison_train_dataset_dir, original_data_filename)) 

        locations, captions = self.__create_backdoor(df_backdoor)

        # 保存新数据集的image-to-caption pair到csv文件
        data = {
            'image': locations,
            'caption': captions
        }
        df_backdoor = pd.DataFrame(data)

        # create the new training dataset by combining poisoned data and clean data
        df = pd.concat([df_backdoor, df_non_backdoor])
        df = df.reset_index(drop=True)

        df.to_csv(self.backdoor_image_to_caption_file_path)
        print(f"Save {self.backdoor_image_to_caption_file_name} into {self.backdoor_image_to_caption_file_path}\n")


    def __create_backdoor(self, df_backdoor):
    
        # 读出提示词模板
        config  = eval(open(self.template_path, "r").read())
                
        templates = config["templates"]
        classes = config["classes"]
        target_caption = classes[self.target_label]

        secret = torch.FloatTensor(np.random.binomial(1, .5, self.encoder_schedule['secret_size']).tolist())

        locations, captions = [], []
        # poison the images in df_backdoor by applying a backdoor patch and changing the caption
        for i in tqdm(range(len(df_backdoor))):
            image_loc  = df_backdoor.iloc[i]["image"]
            # print(f"image_loc:{image_loc}\n")
            image_name = image_loc.split("/")[-1]

            image = Image.open(os.path.join(self.origin_dataset_dir, image_loc)).convert("RGB")

            # 为image添加后门trigger
            image = self.apply_trigger(image,secret)
        
            image_filename = f"{self.backdoor_folder_name}/{image_name}"
            locations.append(image_filename)

            # 从提示词模板中随机选择一个生成caption
            if templates is not None:
                temp = random.randint(0, len(templates) - 1)
                captions.append(templates[temp](target_caption))
            else:
                captions.append(target_caption)

            image.save(os.path.join(self.poison_train_dataset_dir, image_filename))

        return locations, captions
    
    def apply_trigger(self, image, secret):

        # print(f"attack_type:{attack_type}\n")

        W, H = 224, 224

        T1 = transforms.ToTensor()
        T2 = transforms.ToPILImage()

        image = image.resize((224, 224))
        image = T1(image)
        image = torch.unsqueeze(image, dim=0)

        residual = self.encoder([secret, image])
        encoded_image = image + residual
        encoded_image = encoded_image.clamp(0, 1)
        encoded_image = torch.squeeze(encoded_image)

        image = T2(encoded_image)
        
        return image

    def create_poisoned_train_dataset(self, processor):

        poisoned_train_dataset = ImageCaptionDataset(self.backdoor_image_to_caption_file_path, processor)

        return poisoned_train_dataset
    
    def create_poisoned_test_dataset(self, processor):

        poisoned_test_dataset = ImageLabelDataset(root=self.origin_test_dataset_dir, transform = processor.process_image, attack_config=self.attack_config)
        
        return poisoned_test_dataset
    
        
    def get_encoder(self):
        return self.encoder
 
    def reset_grad(self, optimizer, d_optimizer):
        optimizer.zero_grad()
        d_optimizer.zero_grad()

    def train_encoder_decoder(self, train_only=False):

        """
        Train the image steganography encoder and decoder.

        Args:
            train_only (bool): Whether to only train the image steganography encoder and decoder.
        """
        if train_only:
            device = torch.device("cuda:0")
        else:
            device = self.device if self.device else torch.device("cuda:0")

        self.encoder = StegaStampEncoder(
            secret_size=self.encoder_schedule['secret_size'], 
            height=self.encoder_schedule['enc_height'], 
            width=self.encoder_schedule['enc_width'],
            in_channel=self.encoder_schedule['enc_in_channel']).to(device)
        
        self.decoder = StegaStampDecoder(
            secret_size=self.encoder_schedule['secret_size'], 
            height=self.encoder_schedule['enc_height'], 
            width=self.encoder_schedule['enc_width'],
            in_channel=self.encoder_schedule['enc_in_channel']).to(device)
        
        self.discriminator = Discriminator(in_channel=self.encoder_schedule['enc_in_channel']).to(device)
    
        train_dl = DataLoader(
            self.train_steg_set,
            batch_size=32,
            shuffle=True,
            num_workers=8,
        )

        enc_total_epoch = self.encoder_schedule['enc_total_epoch']
        enc_secret_only_epoch = self.encoder_schedule['enc_secret_only_epoch']
        optimizer = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}], lr=0.0001)
        d_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00001)
        loss_fn_alex = lpips.LPIPS(net='alex').cuda()
        
        for epoch in range(enc_total_epoch):
            loss_list, bit_acc_list = [], []
            for idx, (image_input, secret_input) in enumerate(train_dl):
                image_input, secret_input = image_input.to(device), secret_input.to(device)
                residual = self.encoder([secret_input, image_input])

                encoded_image = image_input + residual
                encoded_image = encoded_image.clamp(0, 1)
                decoded_secret = self.decoder(encoded_image)
                D_output_fake = self.discriminator(encoded_image)

                # cross entropy loss for the steganography secret
                secret_loss_op = F.binary_cross_entropy_with_logits(decoded_secret.float(), secret_input.float(), reduction='mean')
               
                # the LPIPS perceptual loss
                lpips_loss_op = loss_fn_alex(image_input,encoded_image)
                # L2 residual regularization loss
                l2_loss = torch.square(residual).mean()
                # the critic loss calculated between the encoded image and the original image
                G_loss = D_output_fake

                if epoch < enc_secret_only_epoch:
                    total_loss = secret_loss_op
                else:
                    total_loss = 2.0 * l2_loss + 1.5 * lpips_loss_op.mean() + 1.5 * secret_loss_op + 0.5 * G_loss
                loss_list.append(total_loss.item())

                bit_acc = get_secret_acc(secret_input, decoded_secret)
                bit_acc_list.append(bit_acc.item())

                total_loss.backward()
                optimizer.step()
                self.reset_grad(optimizer, d_optimizer)

                if epoch >= enc_secret_only_epoch and self.encoder_schedule['enc_use_dis']:
                    residual = self.encoder([secret_input, image_input])
                    encoded_image = image_input + residual
                    encoded_image = encoded_image.clamp(0, 1)
                    decoded_secret = self.decoder(encoded_image)
                    D_output_fake = self.discriminator(encoded_image)
                    D_output_real = self.discriminator(image_input)
                    D_loss = D_output_real - D_output_fake
                    D_loss.backward()
                    for p in self.discriminator.parameters():
                        p.grad.data = torch.clamp(p.grad.data, min=-0.01, max=0.01)
                    d_optimizer.step()
                    self.reset_grad(optimizer, d_optimizer)

            if train_only:
                msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'
                print(msg)
                exit()
            else:
                msg = f'Epoch [{epoch + 1}] total loss: {np.mean(loss_list)}, bit acc: {np.mean(bit_acc_list)}\n'
                print(msg)

        savepath = os.path.join(self.work_dir, 'model/encoder_decoder.pth')
        state = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
        }
        torch.save(state, savepath)
    
    
    def get_img(self, path=None):
        """Get the encoded images with the trigger pattern.

        Args:
            path (str): The path of the saved image steganography encoder.
        """
        
        if path is not None:
            device = torch.device("cuda:0")
            if self.device is None:
                self.device = device
            encoder = StegaStampEncoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'], 
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            decoder = StegaStampDecoder(
                secret_size=self.encoder_schedule['secret_size'], 
                height=self.encoder_schedule['enc_height'], 
                width=self.encoder_schedule['enc_width'],
                in_channel=self.encoder_schedule['enc_in_channel']).to(self.device)
            encoder.load_state_dict(torch.load(os.path.join(path, 'encoder_decoder.pth'))['encoder_state_dict'])
            decoder.load_state_dict(torch.load(os.path.join(path, 'encoder_decoder.pth'))['decoder_state_dict'])
        else:
            encoder = self.encoder.to(self.device)
            decoder = self.decoder.to(self.device)

        encoder = encoder.eval()
        decoder = decoder.eval()
        train_dl = DataLoader(
            self.train_steg_set,
            batch_size=1,
            shuffle=True,
            num_workers=8
        )

        for _, (image_input, secret_input) in enumerate(train_dl):
            image_input, secret_input = image_input.cuda(), secret_input.cuda()
            residual = encoder([secret_input, image_input])
            encoded_image = image_input + residual
            encoded_image = torch.clamp(encoded_image, min=0, max=1)

            decoded_secret = decoder(encoded_image)
            bit_acc = get_secret_acc(secret_input, decoded_secret)
            print('bit_acc: ', bit_acc)

            image_input = image_input.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            encoded_image = encoded_image.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]
            residual = residual.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]

            # print(f"image_input:{image_input.shape},encoded_image:{encoded_image.shape},residual:{residual.shape}\n")
            # print(f"image_input:{image_input},encoded_image:{encoded_image},residual:{residual}\n")

            image_input = (image_input * 255).clip(0, 255).astype(np.uint8)
            encoded_image = (encoded_image * 255).clip(0, 255).astype(np.uint8)
            residual = (residual * 255).clip(0, 255).astype(np.uint8)
            
            imageio.imwrite(os.path.join(self.work_dir, 'datasets/samples/image_input.jpg'), image_input)
            imageio.imwrite(os.path.join(self.work_dir, 'datasets/samples/encoded_image.jpg'), encoded_image)
            imageio.imwrite(os.path.join(self.work_dir, 'datasets/samples/residual.jpg'), residual)
            break

class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, attack_config=None):
        self.root = root
        # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
        # print(os.path.join(root, 'labels.csv'))
        # df = pd.read_csv(os.path.join(root, filename))

        df = pd.read_csv(os.path.join(root, 'labels.csv'))
        self.images = df["image"]
        self.labels = df["label"]

        self.attack_config = attack_config
        self.target_label = attack_config["target_label"]
        self.poisoned_rate = attack_config["poisoned_rate"]

        self.attack_type = attack_config["attack_type"]

        self.classes_path = self.attack_config["test_dataset_config"]["template_path"] 
        
        # print(f"self.attack_type:{self.attack_type}\n")
        # print(f"self.classes_path:{self.classes_path}\n")

        self.transform = transform
     
        poisoned_num = int(len(self.images) * self.poisoned_rate)
        tmp_list = np.arange(len(self.images))[~np.array(self.labels == self.target_label)]
        random.shuffle(tmp_list)
        self.backdoor_indices = sorted(list(tmp_list[:poisoned_num]))
        
        self.encoder = attack_config["encoder"]
        self.encoder_schedule = attack_config["encoder_schedule"]
        
        self.secret = torch.FloatTensor(np.random.binomial(1, .5, self.encoder_schedule['secret_size']).tolist())

 
    def __len__(self):
        return len(self.labels)
    
    def add_trigger(self, image):

        W, H = 224, 224
        T1 = transforms.ToTensor()
        T2 = transforms.ToPILImage()

        # image = image.resize((224, 224))
        # image = T1(image)

        # weight = self.attack_config["weight"]
        # pattern = self.attack_config["pattern"]
        # image = weight * pattern + (1.0 - weight) * image

        image = image.resize((224, 224))
        image = T1(image)
        image = torch.unsqueeze(image, dim=0)

        residual = self.encoder([self.secret, image])
        encoded_image = image + residual
        encoded_image = encoded_image.clamp(0, 1)
        encoded_image = torch.squeeze(encoded_image)

        image = T2(encoded_image)

        return image

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        label = self.labels[idx]
        if idx in self.backdoor_indices:  
            image = self.add_trigger(image)
            label = self.target_label

        image = self.transform(image)
    
        return image, label
