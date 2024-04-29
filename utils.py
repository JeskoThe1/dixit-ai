import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

def build_image_grid(image_paths, border=0):
    images = [Image.open(p) for p in image_paths]
    """Builds a grid of images from a list of image paths."""
    # Choose the grid shape that can contain all images
    if len(images) == 0:
        print("No images found")
        return None

    grid_shapes = [(2, 3), (2, 4), (3, 4)]
    for grid_shape in grid_shapes:
        if grid_shape[0] * grid_shape[1] >= len(images):
            break

    # Resize all images to the size of image[0]
    images = [img.resize(images[0].size) for img in images]

    # In the center of each image, assign the image number using OpenCV
    for i, img in enumerate(images):
        img_copy = np.array(img).copy()
        # Explanation of cv2.putText parameters:
        # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
        cv2.putText(img_copy, str(i), (img.size[0] // 2, img.size[1] // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5)
        images[i] = Image.fromarray(img_copy)

    # Build grid
    grid = Image.new('RGB', (
        grid_shape[1] * images[0].size[0] + (grid_shape[1] - 1) * border,
        grid_shape[0] * images[0].size[1] + (grid_shape[0] - 1) * border,
    ))
    for i, img in enumerate(images):
        grid.paste(img, (
            (i % grid_shape[1]) * (img.size[0] + border),
            (i // grid_shape[1]) * (img.size[1] + border),
        ))
    return grid


def download_image_from_message_to_cache(bot, message, image_folder):
    downloaded_file = bot.download_file(bot.get_file(message.photo[-1].file_id).file_path)
    cache_path = os.path.join(image_folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg")
    with open(cache_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    return cache_path


def get_cards_from_image(prediction, image_path):
    images = []
    image = Image.open(image_path)

    prefix, ext = os.path.splitext(image_path)
    detected_cards_paths = []
    for pred in prediction:
        box = pred["box"]
        xmin, ymin, xmax, ymax = box.values()
        im = image.crop((xmin, ymin, xmax, ymax))
        images.append(im)

    for i, image in enumerate(images):
        detected_cards_paths.append(f"{prefix}_card-{i}-of-{len(images)}{ext}")
        image.save(detected_cards_paths[-1])

    grid = build_image_grid(detected_cards_paths)

    return {
        'images' : images,
        'grid': grid,
        'card_paths': detected_cards_paths
    }


def reset_game_state(game_state_path):
    with open(game_state_path, "w") as f:
        f.write("my_cards: {}\n")


def get_game_state_path(bot, message, game_state_folder):
    game_state_path = os.path.join(game_state_folder, f"{message.from_user.username}.yaml")
    if not os.path.exists(game_state_path):
        reset_game_state(game_state_path)
    return game_state_path