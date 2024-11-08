import os
import sys
import torch
import pickle
import appdirs
import numpy as np
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont

class ImageRetrieval:
    """
    Image retrieval object using DINOv2.
    """
    def __init__(self, args):
        logger.remove()

        if args.verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        model_name_dict = {
            "small": "dinov2_vits14",
            "base": "dinov2_vitb14",
            "large": "dinov2_vitl14",
            "largest": "dinov2_vitg14",
        }

        model_name = model_name_dict[args.model_size]

        model_folder = (
            "facebookresearch/dinov2" if args.model_path is None else args.model_path
        )
        model_source = "github" if args.model_path is None else "local"

        try:
            logger.info(f"loading {model_name=} from {model_folder=}")
            self.model = torch.hub.load(
                model_folder,
                model_name,
                source=model_source,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
        except FileNotFoundError:
            logger.error(f"load model failed. please check if {model_folder=} exists")
            sys.exit(1)
        except http.client.RemoteDisconnected:
            logger.error(
                "connect to github is reset. maybe set --model-path to $HOME/.cache/torch/hub/facebookresearch_dinov2_main ?"
            )
            sys.exit(1)

        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.top_k = args.num
        self.model_name = model_name

        if not args.disable_cache:
            cache_root_folder = Path(
                appdirs.user_cache_dir(appname="dinov2_retrieval", appauthor="vra")
            )
            cache_root_folder.mkdir(parents=True, exist_ok=True)
            self.database_features_cache_path = cache_root_folder / (
                    Path(args.database).name + "_" + model_name + ".pkl"
            )
            logger.debug(f"{cache_root_folder=}, {self.database_features_cache_path=}")

    def glob_images(self, path):
        """Find all image files in path"""
        files = (
                list(path.rglob("*.jpg"))
                + list(path.rglob("*.JPG"))
                + list(path.rglob("*.jpeg"))
                + list(path.rglob("*.png"))
                + list(path.rglob("*.bmp"))
        )
        file_names = [f.name for f in files]
        duplicates = [item for item, count in Counter(file_names).items() if count > 1]
        if duplicates:
            logger.info(f"중복된 파일: {duplicates}")
        return list(set(files))

    def extract_database_features(self, database_img_paths):
        """Extract database dinov2 features"""
        database_features = []
        for img_path in tqdm(database_img_paths):
            img = Image.open(str(img_path)).convert("RGB")
            feature = self.extract_single_image_feature(img)
            database_features.append(feature)
        return database_features

    def run(self, args):
        """Run image retrieval on query image(s) using dinov2"""

        query_path = Path(args.query)
        if query_path.is_dir():
            query_paths = self.glob_images(query_path)
        else:
            query_paths = [query_path]

        logger.debug(f"query image paths: {list(query_paths)}")

        if len(query_paths) < 1:
            logger.warning("no query image, exit")
            return

        database_img_paths = self.glob_images(Path(args.database))

        if len(database_img_paths) < 1:
            logger.warning("database does not contain images, exit")
            return

        self.top_k = min(self.top_k, len(database_img_paths))

        logger.info("preparing database features")
        database_features = self.extract_database_features(database_img_paths)
        pickle.dump(database_features, open(str(self.database_features_cache_path), "wb"))

        for img_path in query_paths:
            logger.info(f"processing {img_path}")
            try:
                img = Image.open(str(img_path)).convert("RGB")
            except PIL.UnidentifiedImageError:
                logger.debug(f"query path is not an image: {img_path}")
                continue

            logger.debug("Extracting features on query image")
            feature = self.extract_single_image_feature(img)

            logger.debug("Calculate similarity")
            distances = self.calculate_distance(feature, database_features)
            closest_indices = np.argsort(distances)[::-1][: self.top_k]
            sorted_distances = np.sort(distances)[::-1][: self.top_k]

            self.save_result(
                args,
                img,
                img_path,
                database_img_paths,
                closest_indices,
                sorted_distances,
            )

    def calculate_distance(self, query_feature, database_features):
        cosine_distances = [
            np.dot(query_feature, feature)
            / (np.linalg.norm(query_feature) * np.linalg.norm(feature))
            for feature in database_features
        ]
        return cosine_distances

    def save_result(
            self,
            args,
            query_image,
            query_path,
            database_img_paths,
            closest_indices,
            sorted_distances,
    ):
        img_save_folder = (
                Path(args.output_root)
                / Path(args.database).name
                / Path(self.model_name).name
        )
        img_save_path = img_save_folder / (
                query_path.stem + "_output" + query_path.suffix
        )
        logger.info(f"Save results to {img_save_path}")

        img_save_folder.mkdir(parents=True, exist_ok=True)

        query_image = self.process_image_for_visualization(args, query_image)

        vis_img_list = [query_image]
        for idx, img_idx in enumerate(closest_indices):
            img_path = database_img_paths[img_idx]
            similarity = sorted_distances[idx]
            logger.debug(
                f"{idx}th similar image is {img_path}"
            )
            cur_img = Image.open(img_path)
            cur_img = self.process_image_for_visualization(args, cur_img)

            vis_img_list.append(cur_img)

        out_img = self.create_output_image(vis_img_list, database_img_paths, closest_indices, sorted_distances, args)

        out_img.save(str(img_save_path))

    def create_output_image(self, vis_img_list, database_img_paths, closest_indices, sorted_distances, args):
        """Combine images and add text below each image."""
        total_width = sum(img.width for img in vis_img_list)
        max_height = max(img.height for img in vis_img_list) + 20  # Extra space for text below each image
        out_img = Image.new("RGB", (total_width, max_height), (255, 255, 255))  # White background

        x_offset = 0
        for idx, img in enumerate(vis_img_list):
            out_img.paste(img, (x_offset, 0))
            x_offset += img.width

            if idx > 0: 
                img_idx = closest_indices[idx - 1]
                filename = database_img_paths[img_idx].name
                similarity = sorted_distances[idx - 1]
                text = f"{filename}"
                self.add_text_to_image(out_img, text, x_offset - img.width, img.height)

        return out_img

    def add_text_to_image(self, img, text, x_position, height):
        """Add text to the image at the specified position."""
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        text_position = (x_position, height)
        draw.text(text_position, text, font=font, fill=(0, 0, 0))  

    def process_image_for_visualization(self, args, img):
        """Pad then resize image to target size"""
        width, height = img.size
        if width > height:
            new_width = args.size
            new_height = int((new_width / width) * height)
        else:
            new_height = args.size
            new_width = int((new_height / height) * width)

        img = img.resize((new_width, new_height))

        width, height = img.size
        target_size = args.size
        delta_w = target_size - width
        delta_h = target_size - height
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )

        return ImageOps.expand(img, padding, fill=(255, 255, 255))  # White padding

    def extract_single_image_feature(self, img):
        """Extract feature for single image"""
        img = ImageOps.exif_transpose(img)
        with torch.no_grad():
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            feature = self.model(img_tensor).cpu().numpy()[0]
        return feature

    def process_image_for_visualization(self, args, img):
        """Pad then resize image to target size"""
        img = ImageOps.exif_transpose(img)

        width, height = img.size
        if width > height:
            new_width = args.size
            new_height = int((new_width / width) * height)
        else:
            new_height = args.size
            new_width = int((new_height / height) * width)

        img = img.resize((new_width, new_height))

        width, height = img.size
        target_size = args.size
        delta_w = target_size - width
        delta_h = target_size - height
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )

        return ImageOps.expand(img, padding, fill=(255, 255, 255))  # White padding

    def add_text_label(self, img, label):
        """Add label to image"""
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        new_img = Image.new("RGB", (img.width, img.height + text_h + 10), (255, 255, 255))
        new_img.paste(img, (0, 0))

        draw = ImageDraw.Draw(new_img)
        draw.text(((img.width - text_w) // 2, img.height + 5), label, fill=(0, 0, 0), font=font)

        return new_img