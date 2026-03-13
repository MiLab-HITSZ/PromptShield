
# sys 
import os
BASE_DIR = "../"
attack_and_dataset_config = {
    "GTSRB":{
        "num_classes": 43,
        "data_dir": os.path.join(BASE_DIR,f"datasets/GTSRB"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/GTSRB/train"),
        "test_data_dir": os.path.join(BASE_DIR,f"datasets/GTSRB/test"),
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/GTSRB/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/GTSRB/test/classes.py") 
    },
    "CIFAR-10":{
        "num_classes": 10,
        "data_dir": os.path.join(BASE_DIR,f"datasets/CIFAR-10"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/CIFAR-10/train"),
        "test_data_dir": os.path.join(BASE_DIR,f"datasets/CIFAR-10/test"),
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/CIFAR-10/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/CIFAR-10/test/classes.py")
    },
    "CIFAR-100":{
        "num_classes": 100,
        "data_dir": os.path.join(BASE_DIR,f"datasets/CIFAR-100"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/CIFAR-100/train"),
        "test_data_dir": os.path.join(BASE_DIR,f"datasets/CIFAR-100/test"),
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/CIFAR-100/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/CIFAR-100/test/classes.py")
    },
    "ImageNet1K":{ 
        "num_classes": 1000,
        "data_dir": os.path.join(BASE_DIR,f"datasets/ImageNet1K"),
        # "train_data_dir": os.path.join(BASE_DIR, f"datasets/ImageNet1K/train"),
        "train_data_dir": os.path.join(BASE_DIR, f"datasets/ImageNet1K/validation"),
        "test_data_dir": os.path.join(BASE_DIR, f"datasets/ImageNet1K/validation"),
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/ImageNet1K/validation/val.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/ImageNet1K/validation/classes.py")
    },
    "STL-10":{ 
        "num_classes": 10,
        "data_dir": os.path.join(BASE_DIR,f"datasets/STL-10/"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/STL-10/train"), 
        "test_data_dir": os.path.join(BASE_DIR,f"datasets/STL-10/test"), 
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/STL-10/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/STL-10/test/classes.py")
    },
    "SVHN":{ 
        "num_classes": 10,
        "data_dir": os.path.join(BASE_DIR,f"datasets/SVHN/"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/SVHN/train"),
        "test_data_dir":  os.path.join(BASE_DIR,f"datasets/SVHN/test"), 
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/SVHN/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/SVHN/test/classes.py")
    },
    "DTD":{ 
        "num_classes": 47,
        "data_dir": os.path.join(BASE_DIR,f"datasets/DTD/"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/DTD/train"),
        "test_data_dir":  os.path.join(BASE_DIR,f"datasets/DTD/test"), 
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/DTD/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/DTD/test/classes.py")
    },
    "Food-101":{ 
        "num_classes": 37,
        "data_dir": os.path.join(BASE_DIR,f"datasets/Food-101/"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/Food-101/train"),
        "test_data_dir":  os.path.join(BASE_DIR,f"datasets/Food-101/test"), 
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/Food-101/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/Food-101/test/classes.py")
    },
    "Oxford-IIIT-Pet":{ 
        "num_classes": 37,
        "data_dir": os.path.join(BASE_DIR,f"datasets/Oxford-IIIT-Pet/"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/Oxford-IIIT-Pet/train"),
        "test_data_dir":  os.path.join(BASE_DIR,f"datasets/Oxford-IIIT-Pet/test"), 
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/Oxford-IIIT-Pet/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/Oxford-IIIT-Pet/test/classes.py")
    },
    "RenderedSST-2":{ 
        "num_classes": 2,
        "data_dir": os.path.join(BASE_DIR,f"datasets/RenderedSST-2/"),
        "train_data_dir": os.path.join(BASE_DIR,f"datasets/RenderedSST-2/train"),
        "test_data_dir":  os.path.join(BASE_DIR,f"datasets/RenderedSST-2/test"), 
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/RenderedSST-2/train.csv"),
        "classes_path": os.path.join(BASE_DIR,f"datasets/classes/RenderedSST-2/test/classes.py")
    },

    "CC3M":{ 
        "num_classes": 1000,
        "data_dir": os.path.join(BASE_DIR,f"datasets/CC3M/"),
        "train_data_dir": None,
        "test_data_dir": None,
        "image_to_caption_path": os.path.join(BASE_DIR,f"datasets/CC3M/train.csv"),
        # "classes_path": os.path.join(BASE_DIR,f"datasets/classes/CC3M/classes.py")
    },


    "BadNets":{
        "GTSRB":{
            "target_label":0,
            "target_caption":"Speed limit 20 km/h"
        },
        "CIFAR-10":{
            "target_label":0,
            "target_caption":"airplane"
        },
        "CIFAR-100":{
            "target_label":0,
            "target_caption":"apples"
        },
        "ImageNet1K":{ 
            "target_label":954,
            "target_caption":"banana"
        },
        "DTD":{ 
            "target_label":0,
            "target_caption":"banded"
        },
        "Oxford-IIIT-Pet":{ 
            "target_label":0,
            "target_caption":"Abyssinian"
        },
        "Food-101":{
            "target_label":0,
            "target_caption":"apple pie"
        },
        "RenderedSST-2":{
            "target_label":0,
            "target_caption":"negative"
        }
    },
    "Blended":{
        "GTSRB":{
            "target_label":0,
            "target_caption":"Speed limit 20 km/h"
        },
        "CIFAR-10":{
            "target_label":0,
            "target_caption":"airplane"
        },
        "CIFAR-100":{
            "target_label":0,
            "target_caption":"apples"
        },
        "ImageNet1K":{ 
            "target_label":954,
            "target_caption":"banana"
        },
        "DTD":{ 
            "target_label":0,
            "target_caption":"banded"
        },
        "Oxford-IIIT-Pet":{ 
            "target_label":0,
            "target_caption":"Abyssinian"
        },
        "Food-101":{
            "target_label":0,
            "target_caption":"apple pie"
        },
        "RenderedSST-2":{
            "target_label":0,
            "target_caption":"negative"
        }
    },
    "WaNet":{
        "GTSRB":{
            "target_label":0,
            "target_caption":"Speed limit 20 km/h"
        },
        "CIFAR-10":{
            "target_label":0,
            "target_caption":"airplane"
        },
        "CIFAR-100":{
            "target_label":0,
            "target_caption":"apples"
        },
        "ImageNet1K":{ 
            "target_label":954,
            "target_caption":"banana"
        },
        "DTD":{ 
            "target_label":0,
            "target_caption":"banded"
        },
        "Oxford-IIIT-Pet":{ 
            "target_label":0,
            "target_caption":"Abyssinian"
        },
        "Food-101":{
            "target_label":0,
            "target_caption":"apple pie"
        },
        "RenderedSST-2":{
            "target_label":0,
            "target_caption":"negative"
        }
    },
    "SIG":{
        "GTSRB":{
            "target_label":0,
            "target_caption":"Speed limit 20 km/h"
        },
        "CIFAR-10":{
            "target_label":0,
            "target_caption":"airplane"
        },
        "CIFAR-100":{
            "target_label":0,
            "target_caption":"apples"
        },
        "ImageNet1K":{ 
            "target_label":954,
            "target_caption":"banana"
        },
        "DTD":{ 
            "target_label":0,
            "target_caption":"banded"
        },
        "Oxford-IIIT-Pet":{ 
            "target_label":0,
            "target_caption":"Abyssinian"
        },
        "Food-101":{
            "target_label":0,
            "target_caption":"apple pie"
        },
        "RenderedSST-2":{
            "target_label":0,
            "target_caption":"negative"
        }
    },
    "SSBA":{
        "GTSRB":{
            "target_label":0,
            "target_caption":"Speed limit 20 km/h"
        },
        "CIFAR-10":{
            "target_label":0,
            "target_caption":"airplane"
        },
        "CIFAR-100":{
            "target_label":0,
            "target_caption":"apples"
        },
        "ImageNet1K":{ 
            "target_label":954,
            "target_caption":"banana"
        },
        "DTD":{ 
            "target_label":0,
            "target_caption":"banded"
        },
        "Oxford-IIIT-Pet":{ 
            "target_label":0,
            "target_caption":"Abyssinian"
        },
        "Food-101":{
            "target_label":0,
            "target_caption":"apple pie"
        },
        "RenderedSST-2":{
            "target_label":0,
            "target_caption":"negative"
        }
    },
    "BadEncoder":{
        "GTSRB":{
            "target_label":0,
            "target_caption":"Speed limit 20 km/h"
        },
        "CIFAR-10":{
            "target_label":0,
            "target_caption":"airplane"
        },
        "CIFAR-100":{
            "target_label":0,
            "target_caption":"apples"
        },
        "ImageNet1K":{ 
            "target_label":954,
            "target_caption":"banana"
        },
        "DTD":{ 
            "target_label":0,
            "target_caption":"banded"
        },
        "Oxford-IIIT-Pet":{ 
            "target_label":0,
            "target_caption":"Abyssinian"
        },
        "Food-101":{
            "target_label":0,
            "target_caption":"apple pie"
        },
        "RenderedSST-2":{
            "target_label":0,
            "target_caption":"negative"
        }
    },
    "BadCLIP":{
        "GTSRB":{
            "target_label":0,
            "target_caption":"Speed limit 20 km/h"
        },
        "CIFAR-10":{
            "target_label":0,
            "target_caption":"airplane"
        },
        "CIFAR-100":{
            "target_label":0,
            "target_caption":"apples"
        },
        "ImageNet1K":{ 
            "target_label":954,
            "target_caption":"banana"
        },
        "DTD":{ 
            "target_label":0,
            "target_caption":"banded"
        },
        "Oxford-IIIT-Pet":{ 
            "target_label":0,
            "target_caption":"Abyssinian"
        },
        "Food-101":{
            "target_label":0,
            "target_caption":"apple pie"
        },
        "RenderedSST-2":{
            "target_label":0,
            "target_caption":"negative"
        }
    }
}
