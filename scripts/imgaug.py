import numpy as np
import os
from shutil import copyfile
import sys
import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image

ia.seed(1)

def generateAugImg(im, n, output="PIL"):
    """
    Input:
    - im: PIL image or numpy image
    
    Output:
    - array_imgs: list of PIL images or numpy array of numpy images
    """
    img_array = None
    result = None

    if isinstance(im, Image.Image):
        img_array = np.asarray(im)
    elif isinstance(im, np.ndarray):
        img_array = im
    
    images = np.array([img_array]*n)

    sometimes = lambda aug: iaa.Sometimes(0.7, aug)

    seq = iaa.Sequential(
    [
        iaa.SomeOf((0, 5),
            [
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),
                iaa.Invert(0.05, per_channel=True),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
    )
    
    images_aug = seq(images=images)

    if output == "PIL":
        result = [Image.fromarray(img_aug) for img_aug in images_aug]
    else:
        result = images_aug

    return result

def main(parent_dir, n, img_dir, labels_dir):
    # Check if img_dir and labels_dir exist inside parent_dir
    if not os.path.exists(os.path.join(parent_dir, img_dir)) or not os.path.exists(os.path.join(parent_dir, labels_dir)):
        print("Error: Image directory or labels directory does not exist.")
        return

    # Iterate through each file in img_dir
    for filename in os.listdir(os.path.join(parent_dir, img_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', 'JPG', 'JPEG')):
            # Read the image using PIL
            img_path = os.path.join(parent_dir, img_dir, filename)
            image = Image.open(img_path)

            # Generate augmented images
            augmented_images = generateAugImg(image, n, "PIL")

            # Iterate through each augmented image
            for i, aug_img in enumerate(augmented_images):
                # Save augmented image
                original_name, extension = os.path.splitext(filename)
                new_filename = f"{original_name}_aug_{i}{extension}"
                aug_img.save(os.path.join(parent_dir, img_dir, new_filename))

                # Copy corresponding label file
                label_path = os.path.join(parent_dir, labels_dir, f"{original_name}.txt")
                new_label_path = os.path.join(parent_dir, labels_dir, f"{original_name}_aug_{i}.txt")
                copyfile(label_path, new_label_path)
                print(f"Augmented image '{new_filename}' and corresponding label file created.")


if __name__ == "__main__":
    # Usage: python script.py n parent_dir img_dir labels_dir
    if len(sys.argv) != 5:
        print("Usage: python script.py n_augment parent_dir img_dir labels_dir")
        sys.exit(1)
    
    n_agument = int(sys.argv[1])
    parent_dir = sys.argv[2]
    img_dir = sys.argv[3]
    labels_dir = sys.argv[4]
    main(parent_dir, n_agument, img_dir, labels_dir)