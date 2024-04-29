import os
import yaml
import logging
import telebot
from telebot import types
from transformers import pipeline
from PIL import Image
import captioning
from prompts import (
    generate_clue_for_image,
    guess_image_by_clue,
)
import numpy as np
from utils import download_image_from_message_to_cache, get_cards_from_image, reset_game_state, get_game_state_path, build_image_grid
from datetime import datetime
import random
import imagehash
import sqlite3
import io
import re
import uuid
import glob


class DixitBot:
    def __init__(self, token):
        self.token = token
        self.bot = telebot.TeleBot(token)
        self.captioning_models = captioning.CaptioningModelsWrapper()
        self.detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
        self.clue_from_hand = None
        self.game_state = None
        self.game_state_path = None
        self.added_cards_dict = None
        self.added_cards_image_path = None
        self.cache_path_clue = None
        self.cache_path_guess = None
        self.clue = None
        self.images_clue = None
        self.images_guess = None
        self.grid = None
        self.result_dict = None
        self.result_dict_hand = None
        self.true_image = None
        self.IMAGE_FOLDER = ".cache/images/"
        self.GAME_STATE_FOLDER = ".cache/game_state/"
        self.OUTPUT_LOGS = ".cache/output_logs/"
        self.con = sqlite3.connect("dixit_results.db", check_same_thread=False)
        with open('schema.sql') as f:
            self.con.executescript(f.read())
        self.cur = self.con.cursor()

    def start(self):
        self.bot.polling()

    def send_welcome(self, message):
        logging.log(logging.INFO, f"Received [help] request from {message.from_user.username}.")
        self.bot.reply_to(message, ("I am Dixit Bot, here to play some good association with you. Im newbie, so please don't be rough :3"
                                    "To work properly, i need two things: photo of your dixit cards and a command."
                                    "Photo should have all of cards on it. Once you uploaded photo, send a command with it: /clue or /guess *Here goes your clue to guess*. "
                                    "You can also add cards to your hand via /add + photo of cards. "
                                    "Command /guess_hand + clue will choose a card that suits given clue the most"
                                    "Command /del + card_index will delete card from your hand"
                                    "After this, you can check how cards were detected and thus start playing. This is first version, so my guessing can take some time... But we will improve!"))

    def generate_clue_for_cards(self, message):
        logging.log(logging.INFO, f"Received [clue] request from {message.from_user.username}.")
        cache_path_clue = download_image_from_message_to_cache(self.bot, message, image_folder=".cache/images/")
        image = Image.open(cache_path_clue)
        prediction = self.detector(image, candidate_labels=["playing card with picture on it"])
        cards_dict = get_cards_from_image(prediction, cache_path_clue)
        if cards_dict['grid'] == None:
            self.bot.send_message(message.chat.id, "Couldn't detect any card, please try uploading another image")
        else:
            markup_clue = types.InlineKeyboardMarkup(row_width=2)
            yes_clue = types.InlineKeyboardButton("yes", callback_data="clue_yes")
            no = types.InlineKeyboardButton("no", callback_data="clue_no")
            markup_clue.add(yes_clue, no)
            self.bot.send_photo(message.chat.id, cards_dict['grid'], reply_to_message_id=message.message_id, caption="Detected cards. Is it done properly?", reply_markup=markup_clue)
            self.cache_path_clue = cache_path_clue
            self.images_clue = cards_dict['images']

    def add_cards_to_hand(self, message):
        logging.log(logging.INFO, f"Received [add images] request from {message.from_user.username}")

        self.added_cards_image_path = download_image_from_message_to_cache(self.bot, message, image_folder=self.IMAGE_FOLDER)
    
        self.game_state_path = get_game_state_path(self.bot, message, game_state_folder=self.GAME_STATE_FOLDER)
        with open(self.game_state_path, 'r') as fd:
            self.game_state = yaml.safe_load(fd)
        
        image = Image.open(self.added_cards_image_path)
        start_detection = datetime.now()
        prediction = self.detector(image, candidate_labels=["playing card with picture on it"])
        self.added_cards_dict = get_cards_from_image(prediction, self.added_cards_image_path)
        detection_time = (datetime.now() - start_detection).total_seconds()

        if self.added_cards_dict['grid'] == None:
            self.bot.send_message(message.chat.id, "Couldn't detect any card, please try uploading another image")
        else:
            markup_clue = types.InlineKeyboardMarkup(row_width=2)
            yes_add = types.InlineKeyboardButton("yes", callback_data="add_yes")
            no_add = types.InlineKeyboardButton("no", callback_data="add_no")
            markup_clue.add(yes_add, no_add)
            reply_message = (
                f"Found {len(self.added_cards_dict['images'])} cards in {detection_time:0.1f} seconds. ")
            
            self.bot.send_photo(
                message.chat.id, self.added_cards_dict['grid'],
                reply_to_message_id=message.message_id, caption=reply_message, reply_markup=markup_clue)
            
    def check_adding_cards(self, callback):
        if callback.data == 'add_yes':
            self.bot.reply_to(callback.message, "Generating description and clues...")
            clue_generation_start = datetime.now()
            descriptions = dict()
            for idx, image in enumerate(self.added_cards_dict['images']):
                hash = str(imagehash.average_hash(image))
                descriptions.update({
                    hash: {'image_path': self.added_cards_dict['card_paths'][idx],
                            **generate_clue_for_image(image, captioning.generate_captions, self.captioning_models, verbose=False)
                        }
                })

            self.game_state["my_cards"].update(descriptions)
            output_logs_path = os.path.join(".cache/output_logs/clues", f"{os.path.basename(self.added_cards_image_path).replace('.jpg', '')}.yaml")
            with open(output_logs_path, 'w') as fd:
                yaml.dump(descriptions, fd, default_flow_style=False, sort_keys=False)

            with open(self.game_state_path, 'w') as fd:
                yaml.dump(self.game_state, fd, default_flow_style=False, sort_keys=False)

            clue_generation_time = (datetime.now() - clue_generation_start).total_seconds()
            self.bot.reply_to(callback.message, (
                f"Generated descriptions in {clue_generation_time} seconds. "
                f"Cards in hand: {len(self.game_state['my_cards'])} "
                "You can see your hand using command /hand." ))
        else:
            self.bot.send_message(callback.message.chat.id, "Please retry taking photo of your cards")

    def remove_card_from_hand(self, message):
        logging.log(logging.INFO, f"Received [remove card from hand] request from {message.from_user.username}")
        cards_to_delete = [int(x.strip()) for x in message.text.replace('/del', '').split(',')]

        # Get game state
        game_state_path = get_game_state_path(self.bot, message, game_state_folder=self.GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)
    
        # Remove cards from game state
        retained_cards = dict()
        for card_idx, (card_hash, card_info) in enumerate(game_state["my_cards"].items()):
            if card_idx not in cards_to_delete:
                retained_cards[card_hash] = card_info
        game_state["my_cards"] = retained_cards

        # Save game state (i.e. our hand)
        with open(game_state_path, 'w') as fd:
            yaml.dump(game_state, fd, default_flow_style=False, sort_keys=False)
        self.bot.reply_to(message, "Done removing cards. You can see your new hand with command /hand.")

    def show_detailed_hand_clues(self, message):
        logging.log(logging.INFO, f"Received [hand] request from {message.from_user.username}.")
        # Get game state
        game_state_path = get_game_state_path(self.bot, message, game_state_folder=self.GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)

        # If no cards in hand, return
        if game_state["my_cards"] is None or len(game_state["my_cards"]) == 0:
            self.bot.reply_to(message, "Your hand is empty, there's no card in your hand.")
            return
        
        # Build a grid of images
        image_paths = []
        for card_hash, card_info in game_state["my_cards"].items():
            image_paths.append(card_info["image_path"])
        grid = build_image_grid(image_paths)
        # Send message

        self.bot.send_photo(message.chat.id, grid, caption="Detailed descriptions below:",
                       reply_to_message_id=message.message_id)

        # Build a grid of images
        for image_idx, (card_hash, card_info) in enumerate(game_state["my_cards"].items()):
            cap_text = f"Card {image_idx}: {card_info['clue']}\n\n"
            cap_text += f"captions:\n{card_info['captions']}\n\n"
            cap_text += f"pre_qna_interpretation:\n{card_info['pre_qna_interpretation']}\n\n"
            cap_text += f"qna_session:\n{card_info['qna_session']}\n\n"
            cap_text += f"interpretation:\n{card_info['interpretation']}\n\n"
            cap_text += f"association:\n{card_info['association']}"
            self.bot.reply_to(message, cap_text)

    def show_short_hand_clues(self, message):
        logging.log(logging.INFO, f"Received [hand] request from {message.from_user.username}.")

        # Get game state
        game_state_path = get_game_state_path(self.bot, message, game_state_folder=self.GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            game_state = yaml.safe_load(fd)

        # If no cards in hand, return
        if game_state["my_cards"] is None or len(game_state["my_cards"]) == 0:
            self.bot.reply_to(message, "Your hand is empty, there's no card in your hand.")
            return

        # Display only "interpretation", "association", and "clue" for each card
        response_text = ""
        for image_idx, (card_hash, card_info) in enumerate(game_state["my_cards"].items()):
            response_text += f"Card {image_idx}: {card_info['clue'].strip()}\n"
        response_text += "\nTo get detailed explanation to the clues, please use command /hand_detailed."

        # Build a grid of images
        image_paths = []
        for card_hash, card_info in game_state["my_cards"].items():
            image_paths.append(card_info["image_path"])
        grid = build_image_grid(image_paths)
        # Send message
        self.bot.send_photo(message.chat.id, grid, caption=response_text,
                       reply_to_message_id=message.message_id)
        
    def guess_card_on_table(self, message):
        logging.log(logging.INFO, f"Received [guess] request from {message.from_user.username}.")

        self.clue = message.caption[len('/guess'):].strip()
        cache_path_guess = download_image_from_message_to_cache(self.bot, message, image_folder=".cache/images/")
        image = Image.open(cache_path_guess)
        prediction = self.detector(image, candidate_labels=["playing card with picture on it"])
        cards_dict = get_cards_from_image(prediction, cache_path_guess)
        self.grid, self.images_guess = cards_dict['grid'], cards_dict['images']
        if self.grid == None:
            self.bot.send_message(message.chat.id, "Couldn't detect any card, please try uploading another image")
        else:
            markup_guess = types.InlineKeyboardMarkup(row_width=2)
            yes_guess = types.InlineKeyboardButton("yes", callback_data="guess_yes")
            no = types.InlineKeyboardButton("no", callback_data="guess_no")
            markup_guess.add(yes_guess, no)
            self.bot.send_photo(message.chat.id, self.grid, reply_to_message_id=message.message_id, caption="Detected cards. Is it done properly?", reply_markup=markup_guess)
            self.cache_path_guess = cache_path_guess

    def guess_card_from_hand(self, message):
        logging.log(logging.INFO, f"Received [get card from hand by clue] request from {message.from_user.username}")
        self.clue_from_hand = message.text[len('/guess_hand'):].strip()

        # Get game state
        game_state_path = get_game_state_path(self.bot, message, game_state_folder=self.GAME_STATE_FOLDER)
        with open(game_state_path, 'r') as fd:
            self.game_state = yaml.safe_load(fd)

        # If no cards in hand, return
        if self.game_state["my_cards"] is None or len(self.game_state["my_cards"]) == 0:
            self.bot.reply_to(message, "Your hand is empty, there's no card in your hand.")
            return
        
        images = [np.array(Image.open(card_info['image_path'])) for _, card_info in self.game_state["my_cards"].items()]
        generated_descriptions = [card_info for _, card_info in self.game_state["my_cards"].items()]
        # Generate descriptions and clues
        guess_image_start = datetime.now()
        self.result_dict_hand = guess_image_by_clue(images, self.clue_from_hand, captioning.generate_captions, self.captioning_models, generated_descriptions, verbose=False)

        # Build a grid of images
        image_paths = []
        for card_hash, card_info in self.game_state["my_cards"].items():
            image_paths.append(card_info["image_path"])
        grid = build_image_grid(image_paths)

        guess_image_time = (datetime.now() - guess_image_start).total_seconds()
        final_response = f"Clue: {self.clue_from_hand}\nAnswer: " + self.result_dict_hand["final_answer"].strip() + "\n\n" + (
            "-------------------\n"
            f"Guessed the card (from hand) matching given clue in {guess_image_time} seconds. ")
        self.bot.send_photo(
            message.chat.id, grid, reply_to_message_id=message.message_id,
            caption=final_response)
        
        points_markup = types.InlineKeyboardMarkup(row_width=2)
        for i in range(7):
            points_markup.add(types.InlineKeyboardButton(f"{i}", callback_data=f"from_hand{i}"))
        self.bot.send_message(message.chat.id, "How many points did you obtain?", reply_markup=points_markup)

    def check_images_guess(self, callback):
        if callback.data == "guess_yes":
            self.bot.send_message(callback.message.chat.id, "Begun guessing")
            image_guessing_start = datetime.now()
            self.result_dict = guess_image_by_clue(self.images_guess, self.clue, captioning.generate_captions, self.captioning_models, verbose=False)
            image_guessing_time = (datetime.now() - image_guessing_start).total_seconds()
            self.bot.send_message(callback.message.chat.id, f"Guessed card {self.result_dict['final_answer']} \nfor clue: {self.clue} \nin {image_guessing_time} seconds.")
            guesses_markup = types.InlineKeyboardMarkup(row_width=2)
            for i in range(len(self.images_guess)):
                guesses_markup.add(types.InlineKeyboardButton(f"Image_{i}", callback_data=f"Image_{i}"))
            self.bot.send_message(callback.message.chat.id, "Which image was right to guess?", reply_markup=guesses_markup)
        elif callback.data == "guess_no":
            self.bot.send_message(callback.message.chat.id, "Please retry taking photo of your cards")

    def points_guess(self, callback):
        self.true_image = callback.data
        points_markup = types.InlineKeyboardMarkup(row_width=2)
        for i in range(7):
            points_markup.add(types.InlineKeyboardButton(f"{i}", callback_data=f"_guess{i}"))
        self.bot.send_message(callback.message.chat.id, "How many points did you obtain?", reply_markup=points_markup)

    def persist_guess_from_hand(self, callback):
        stream = io.BytesIO()
        image_paths = []
        for _, card_info in self.game_state["my_cards"].items():
            image_paths.append(card_info["image_path"])
        grid = build_image_grid(image_paths)
        grid.save(stream, format="JPEG")
        grid_bytes = stream.getvalue()

        persist_dict = {}
        points_dict = {f"from_hand{i}" : i for i in range(7)}
        points = points_dict[callback.data]

        clue_relation = ""
        for reasoning in self.result_dict_hand['per_image_reasoning']:
            for key, value in reasoning.items():
                if key == 'clue_relation':
                    clue_relation += value
        persist_dict["clue_relations"] = clue_relation
        persist_dict["score"] = points
        persist_dict["image_grid"] = grid_bytes
        persist_dict["final_answer"] = self.result_dict_hand["final_answer"]
        persist_dict["guessed_image"] = re.search(r'Image_\d+', self.result_dict_hand["final_answer"]).group()
        persist_dict["clue"] = self.clue_from_hand
        self.cur.execute("INSERT INTO guesses_from_hand VALUES(:image_grid, :guessed_image, :final_answer, :score, :clue, :clue_relations)", persist_dict)
        self.con.commit()
        self.bot.send_message(callback.message.chat.id, "Successfully saved this experience.")

    def persist_guess(self, callback):
        stream = io.BytesIO()
        self.grid.save(stream, format="JPEG")
        grid_bytes = stream.getvalue()
        persist_dict = {}
        points_dict = {f"_guess{i}" : i for i in range(7)}
        points = points_dict[callback.data]
        clue_relation = ""
        for reasoning in self.result_dict['per_image_reasoning']:
            for key, value in reasoning.items():
                if key == 'clue_relation':
                    clue_relation += value
        persist_dict["clue_relations"] = clue_relation
        persist_dict["true_image"] = self.true_image
        persist_dict["score"] = points
        persist_dict["image_grid"] = grid_bytes
        persist_dict["final_answer"] = self.result_dict["final_answer"]
        persist_dict["guessed_image"] = re.search(r'Image_\d+', self.result_dict["final_answer"]).group()
        persist_dict["clue"] = self.clue
        output_logs_path = os.path.join(".cache/output_logs/guesses", f"{os.path.basename(self.cache_path_guess).replace('.jpg', '')}.yaml")
        with open(output_logs_path, 'w') as fd:
            yaml.dump(persist_dict, fd, default_flow_style=False, sort_keys=False)
        self.cur.execute("INSERT INTO guesses VALUES(:image_grid, :guessed_image, :true_image, :final_answer, :score, :clue, :clue_relations)", persist_dict)
        self.con.commit()
        self.bot.send_message(callback.message.chat.id, "Successfully saved this experience. You may now proceed to guessing the card by clue")

    def check_images_clue(self, callback):
        if callback.data == "clue_yes":
            self.bot.send_message(callback.message.chat.id, "Begun generating clue:")
            clue_generation_start = datetime.now()
            image_number = random.randint(0, len(self.images_clue))
            self.image_for_clue = self.images_clue[image_number]
            self.clue_dict = generate_clue_for_image(self.image_for_clue, captioning.generate_captions, self.captioning_models, verbose=False)
            clue_generation_time = (datetime.now() - clue_generation_start).total_seconds()
            self.bot.send_message(callback.message.chat.id, f"For chosen card {image_number} Was generated clue: {self.clue_dict['clue']} in {clue_generation_time} seconds.")
            points_markup = types.InlineKeyboardMarkup(row_width=2)
            for i in range(7):
                points_markup.add(types.InlineKeyboardButton(f"{i}", callback_data=f"{i}"))
            self.bot.send_message(callback.message.chat.id, "How many points did you obtain?", reply_markup=points_markup)
        elif callback.data == "no":
            self.bot.send_message(callback.message.chat.id, "Please retry taking photo of your cards")

    def persist_clue(self, callback):
        points_dict = {f"{i}" : i for i in range(7)}
        points = points_dict[callback.data]
        self.clue_dict['image_hash'] = str(imagehash.average_hash(self.image_for_clue))
        self.clue_dict['score'] = points
        self.clue_dict["captions"] = self.clue_dict["captions"]["captions"]
        output_logs_path = os.path.join(".cache/output_logs/clues", f"{os.path.basename(self.cache_path_clue).replace('.jpg', '')}.yaml")
        with open(output_logs_path, 'w') as fd:
            yaml.dump(self.clue_dict, fd, default_flow_style=False, sort_keys=False)
        self.cur.execute("INSERT INTO generated_clues VALUES(:image_hash, :captions, :interpretation, :association, :clue, :score, :qna_session, :pre_qna_interpretation)", self.clue_dict)
        self.con.commit()
        self.bot.send_message(callback.message.chat.id, "Successfully saved this experience. You may now proceed to generating a clue")


def main():
    logging.basicConfig(level=logging.INFO)

    token = os.environ.get('DIXITAI_BOT_TOKEN')
    bot = DixitBot(token)

    os.makedirs(bot.IMAGE_FOLDER, exist_ok=True)
    os.makedirs(bot.GAME_STATE_FOLDER, exist_ok=True)
    os.makedirs(bot.OUTPUT_LOGS, exist_ok=True)

    @bot.bot.message_handler(commands=['help'])
    def send_welcome_wrapper(message):
        bot.send_welcome(message)

    @bot.bot.message_handler(func=lambda m: str(m.caption).startswith("/clue"), content_types=['photo', 'text'])
    def generate_clue_for_cards_wrapper(message):
        try:
            bot.generate_clue_for_cards(message)
        except Exception as e:
            bot.bot.reply_to(message, e)

    @bot.bot.message_handler(func=lambda m: str(m.caption).startswith("/guess"), content_types=['photo', 'text'])
    def guess_card_on_table_wrapper(message):
        try:    
            bot.guess_card_on_table(message)
        except Exception as e:
            bot.bot.reply_to(message, e)

    @bot.bot.message_handler(func=lambda m: str(m.caption).startswith("/add"), content_types=['photo', 'text'])
    def add_cards_to_hand_wrapper(message): 
        bot.add_cards_to_hand(message)

    @bot.bot.message_handler(commands=['guess_hand'])
    def guess_from_hand_wrapper(message):
        bot.guess_card_from_hand(message)

    @bot.bot.message_handler(commands=['del'])
    def delete_card_from_hand_wrapper(message):
        bot.remove_card_from_hand(message)

    @bot.bot.message_handler(commands=['hand', 'status'])
    def get_hand_wrapper(message):
        bot.show_short_hand_clues(message)

    @bot.bot.message_handler(commands=['hand_detailed', 'status_detailed'])
    def get_hand_detailed_wrapper(message):
        try:
            bot.show_detailed_hand_clues(message)
        except Exception as e:
            bot.bot.reply_to(message, e)

    @bot.bot.message_handler(commands=['reset'])
    def reset_state(message):
        logging.log(logging.INFO, f"Received [reset] request from {message.from_user.username}.")    
        reset_game_state(get_game_state_path(bot, message, game_state_folder=bot.GAME_STATE_FOLDER))
        bot.reply_to(message, "Game is reset to initial state.")

    @bot.bot.message_handler(commands=['nuke_cache'])
    def nuke_cache(message):
        logging.log(logging.INFO, f"Received [nuke_cache] request from {message.from_user.username}.")    
        reset_game_state(get_game_state_path(bot, message, game_state_folder=bot.GAME_STATE_FOLDER))
        files = glob.glob(os.path.join(bot.IMAGE_FOLDER, "*"))
        for f in files:
            os.remove(f)
        bot.reply_to(message, "All cache is nuked and game state is reset to initial state.")

    @bot.bot.callback_query_handler(func=lambda callback: callback.data.startswith('from_hand'))
    def persist_guess_from_hand_wrapper(callback):
        bot.persist_guess_from_hand(callback)

    @bot.bot.callback_query_handler(func=lambda callback: callback.data.startswith('add'))
    def check_images_guess_wrapper(callback):
        bot.check_adding_cards(callback)

    @bot.bot.callback_query_handler(func=lambda callback: callback.data.startswith('guess'))
    def check_images_guess_wrapper(callback):
        bot.check_images_guess(callback)

    @bot.bot.callback_query_handler(func=lambda callback: callback.data.startswith('Image_'))
    def points_guess_wrapper(callback):
        bot.points_guess(callback)

    @bot.bot.callback_query_handler(func=lambda callback: callback.data.startswith('_guess'))
    def persist_guess_wrapper(callback):
        bot.persist_guess(callback)

    @bot.bot.callback_query_handler(func=lambda callback: callback.data.startswith('clue'))
    def check_images_clue_wrapper(callback):
        bot.check_images_clue(callback)

    @bot.bot.callback_query_handler(func=lambda callback: True)
    def persist_clue_wrapper(callback):
        bot.persist_clue(callback)

    bot.start()

if __name__ == "__main__":
    main()
