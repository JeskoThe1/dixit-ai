CREATE TABLE IF NOT EXISTS generated_clues(
    image_hash TEXT PRIMARY KEY, 
    captions TEXT, 
    interpretation TEXT, 
    association TEXT, 
    clue TEXT, 
    score INTEGER, 
    qna_session TEXT, 
    pre_qna_interpretation TEXT
);


CREATE TABLE IF NOT EXISTS guesses(
    image_grid BLOB, 
    guessed_image TEXT, 
    true_image TEXT, 
    final_answer TEXT,
    score INTEGER, 
    clue TEXT,
    clue_relations TEXT
);

CREATE TABLE IF NOT EXISTS guesses_from_hand(
    image_grid BLOB, 
    guessed_image TEXT, 
    final_answer TEXT,
    score INTEGER, 
    clue TEXT,
    clue_relations TEXT
);