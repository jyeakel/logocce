from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from better_profanity import profanity
import os
import random
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load NLTK data
nltk.download("wordnet", download_dir="/Users/jymt/Projects/logocce/nltk_data")

# Set up environment
os.environ["PYENCHANT_LIBRARY_PATH"] = "/opt/homebrew/lib/libenchant-2.2.dylib"
import enchant

# Load contextual model
contextual_model = SentenceTransformer("all-MiniLM-L6-v2")

# Game state
game_state = {}

class GameRequest(BaseModel):
    guess: str

class GameResponse(BaseModel):
    message: str
    player_guess: str | None = None
    player_distance: float | None = None
    ai_guess: str | None = None
    ai_distance: float | None = None
    closer: str | None = None
    guess_history: list | None = None
    hint: str | None = None

def build_filtered_vocab(vocab_size=50000):
    lemmatizer = WordNetLemmatizer()
    english_dict = enchant.Dict("en_US")
    profanity.load_censor_words()

    # Load most frequent words
    freq_url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_50k.txt"
    frequency_text = requests.get(freq_url).text.lower().splitlines()

    frequency_set = set()
    for line in frequency_text[:vocab_size]:
        word = line.split()[0]
        if word.isalpha() and word.islower():
            frequency_set.add(word)

    # Collect common nouns from WordNet
    noun_set = set()
    for synset in wn.all_synsets("n"):  # 'n' = noun synsets
        for lemma in synset.lemma_names():
            word = lemma.lower().replace("_", "")  # Normalize
            if word.isalpha() and 4 <= len(word) <= 7 and word.islower():
                if lemmatizer.lemmatize(word, "n") == word:  # Ensure it's singular
                    noun_set.add(word)

    # Intersection of frequency list & noun set
    common_nouns = noun_set.intersection(frequency_set)

    # Explicitly remove proper nouns or uncommon surnames
    filtered_vocab = set(word for word in common_nouns if english_dict.check(word))

    # Filter out profane words using better-profanity
    filtered_vocab = {
        word for word in filtered_vocab if not profanity.contains_profanity(word)
    }

    return filtered_vocab

def precompute_contextual_embeddings(vocab):
    word_list = list(vocab)
    embeddings_tensor = contextual_model.encode(
        word_list, batch_size=64, convert_to_numpy=True, show_progress_bar=True
    )
    embeddings_dict = dict(zip(word_list, embeddings_tensor))
    return embeddings_dict

class LogocceGame:
    def __init__(self, vocab, contextual_vecs):
        self.vocab = vocab
        self.word_list = list(vocab)
        self.emb_matrix = np.stack([contextual_vecs[word] for word in self.word_list])
        self.word_to_index = {word: idx for idx, word in enumerate(self.word_list)}
        self.target_word = random.choice(self.word_list)
        self.guess_history = []
        self.guessed_words = set()
        self.wrong_guesses = 0
        self.ai_improvement = 0.5
        self.revealed_indices = set()
        logger.info(f"Game initialized with target word: {self.target_word}")

    def semantic_distance_vec(self, word_vec, all_vecs):
        dot_products = np.dot(all_vecs, word_vec)
        norms = np.linalg.norm(all_vecs, axis=1) * np.linalg.norm(word_vec)
        cosine_similarities = dot_products / norms
        return 1 - cosine_similarities

    def ai_play_word(self, player_word):
        target_vec = self.emb_matrix[self.word_to_index[self.target_word]]
        player_vec = self.emb_matrix[self.word_to_index[player_word]]

        target_dist = float(1 - np.dot(target_vec, player_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(player_vec)))
        desired_distance = target_dist * (1 - self.ai_improvement)

        dist_to_target = self.semantic_distance_vec(target_vec, self.emb_matrix)
        dist_to_player = self.semantic_distance_vec(player_vec, self.emb_matrix)

        valid_mask = (dist_to_target < target_dist) & (dist_to_player < target_dist)
        guessed_indices = [self.word_to_index[word] for word in self.guessed_words.union({player_word})]
        valid_mask[guessed_indices] = False

        # Explicitly exclude the target word
        valid_mask[self.word_to_index[self.target_word]] = False

        if not valid_mask.any():
            return None

        diffs = np.abs(dist_to_target - desired_distance)
        diffs[~valid_mask] = np.inf

        best_idx = np.argmin(diffs)
        best_word = self.word_list[best_idx]

        self.ai_improvement = min(self.ai_improvement + 0.1, 0.9)

        return best_word

    def reveal_hint(self):
        while len(self.revealed_indices) < min(self.wrong_guesses - 1, len(self.target_word)):
            unrevealed = set(range(len(self.target_word))) - self.revealed_indices
            self.revealed_indices.add(random.choice(list(unrevealed)))
        hint = ' '.join(self.target_word[i] if i in self.revealed_indices else '_' for i in range(len(self.target_word)))
        return hint

    def player_guess(self, guess_word):
        try:
            if guess_word in self.guessed_words:
                return {"message": f"'{guess_word}' has already been guessed. Try another word."}

            if guess_word not in self.vocab:
                return {"message": f"'{guess_word}' is not in the valid word list. Try another word."}

            self.guessed_words.add(guess_word)
            player_vec = self.emb_matrix[self.word_to_index[guess_word]]
            target_vec = self.emb_matrix[self.word_to_index[self.target_word]]
            distance = float(self.semantic_distance_vec(player_vec, np.array([target_vec]))[0])

            self.guess_history.append({'player': 'human', 'word': guess_word, 'distance': round(distance, 3)})

            if distance < 0.1:
                return {
                    "message": f"Correct! The word was '{self.target_word}'.",
                    "player_guess": guess_word,
                    "player_distance": float(round(distance, 3)),
                    "guess_history": self.guess_history
                }

            self.wrong_guesses += 1

            ai_word = self.ai_play_word(guess_word)
            if ai_word is None:
                return {"message": "AI has no valid words left to guess. Game over."}

            ai_vec = self.emb_matrix[self.word_to_index[ai_word]]
            ai_distance = float(self.semantic_distance_vec(ai_vec, np.array([target_vec]))[0])

            self.guessed_words.add(ai_word)
            self.guess_history.append({'player': 'AI', 'word': ai_word, 'distance': round(ai_distance, 3)})

            response = {
                "message": "Guess processed",
                "player_guess": guess_word,
                "player_distance": float(round(distance, 3)),
                "ai_guess": ai_word,
                "ai_distance": float(round(ai_distance, 3)),
                "closer": 'player' if distance < ai_distance else 'ai',
                "guess_history": self.guess_history.copy()
            }

            if self.wrong_guesses > 1:
                response["hint"] = self.reveal_hint()

            return response
        except Exception as e:
            logger.error(f"Error in player_guess: {str(e)}")
            return {"message": f"An error occurred: {str(e)}"}

@app.post("/start")
async def start_game(request: Request):
    try:
        global game_state
        logger.info("Starting new game")
        vocab = build_filtered_vocab()
        contextual_vecs = precompute_contextual_embeddings(vocab)
        game_state = LogocceGame(vocab, contextual_vecs)
        logger.info("New game started successfully")
        return JSONResponse(
            content={"message": "Game started!"},
            headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
        )
    except Exception as e:
        logger.error(f"Error starting game: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
        )

@app.post("/guess")
async def make_guess(request: Request):
    try:
        if not game_state:
            return JSONResponse(
                status_code=400,
                content={"detail": "Game not started"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
        # Parse the request body
        body = await request.json()
        guess = body.get("guess")
        
        if not guess:
            return JSONResponse(
                status_code=400,
                content={"detail": "No guess provided"},
                headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
            )
        
        logger.info(f"Processing guess: {guess}")
        response = game_state.player_guess(guess)
        return JSONResponse(
            content=response,
            headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
        )
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid JSON in request body"},
            headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
        )
    except Exception as e:
        logger.error(f"Error processing guess: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={"Access-Control-Allow-Origin": "http://localhost:3000"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 