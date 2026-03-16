import numpy as np
import pickle
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("next_word_lstm.h5")

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(text):

    max_sequence_len = model.input_shape[1] + 1
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return "Word not found"

interface = gr.Interface(
    fn=predict_next_word,
    inputs=gr.Textbox(
        label="Enter text sequence",
        value="To be or not to"
    ),
    outputs=gr.Textbox(label="Predicted Next Word"),
    title="Next Word Prediction using LSTM",
    description="Enter a sequence of words and the model predicts the next word."
)

interface.launch()