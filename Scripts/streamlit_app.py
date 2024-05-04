import streamlit as st
import time
import pandas as pd
import requests
from io import StringIO
import spacy
from spacy.tokens import Span
from spacy.language import Language
import re
import math
import random


@Language.component("recognize_events")
def recognize_events(d):
    i = 0
    for token in d:
        temp_tuple = (token.text.lower(), 'EVENT')
        if temp_tuple in events:
            # If the token is in events list, set its label to "EVENT"
            d.ents = list(d.ents) + [Span(d, i, i + 1, label="EVENT")]
        i += 1
    return d


@Language.component("recognize_cards")
def recognize_events(d):
    i = 0
    for token in d:
        temp_tuple = (token.text.lower(), 'CARD')
        if temp_tuple in cards:
            # If the token is in cards list, set its label to "CARD"
            d.ents = list(d.ents) + [Span(d, i, i + 1, label="CARD")]
        i += 1
    return d


class Model:
    @staticmethod
    def read_file_from_url(url):
        # Fetch the file content from the URL
        url_response = requests.get(url)
        # Check if request was successful
        if url_response.status_code == 200:
            # Read the content of the file
            content = url_response.text
            # Create a StringIO object to mimic a file object
            file_object = StringIO(content)
            # Read the first row to extract column names
            column_names = file_object.readline().strip().split('\t')
            # Initialize an empty list to hold rows
            rows = []
            # Read each line from the file and append to rows list
            for line in file_object:
                # Split the line based on the tab delimiter
                values = line.strip().split('\t')
                # Trim values if they exceed the number of columns in header
                if len(values) > len(column_names):
                    values = values[:len(column_names)]
                # Append the values to the rows list
                rows.append(values)
            # Create DataFrame from rows
            df = pd.DataFrame(rows, columns=column_names)
            return df
        else:
            print("Failed to fetch the file from URL.")
            return None

    def __init__(self, cp_file, ep_file, et_file):
        self.card_percentages_df = self.read_file_from_url(cp_file)
        self.event_percentages_df = self.read_file_from_url(ep_file)
        self.event_tagging_df = self.read_file_from_url(et_file)

        self.event_names = self.event_tagging_df.iloc[:, 0].tolist()
        self.events = []
        self.card_names = self.card_percentages_df.iloc[:, 0].tolist()
        self.cards = []

        self.choice_words = []
        self.word_counts = []
        self.choice_words_occurrences = []
        self.word_counts_per_event = []
        self.compute_choice_bigrams()

        self.small_val_flag = 0

        # Iterate over the event names and populate the events list
        for event_name in self.event_names:
            new_event_name = event_name.replace(" ", "_")
            self.events.append((new_event_name.lower(), "EVENT"))  # Converting to lowercase for consistency

        for card_name in self.card_names:
            new_card_name = card_name.replace(" ", "_")
            self.cards.append((new_card_name.lower(), "CARD"))

    def preprocess(self, t):
        t = t.lower()
        for event in self.event_names:
            t = t.replace(event.lower(), event.lower().replace(" ", "_"))
        for card in self.card_names:
            t = t.replace(card.lower(), card.lower().replace(" ", "_"))
        return t

    def process_text(self, t, n):
        t = self.preprocess(t)
        d = n(t)
        return d

    def compute_choice_bigrams(self):
        articles = ["a", "an", "the"]
        pronouns = ["i", "you", "he", "she", "they", "it", "we"]
        banned_words = articles + pronouns
        for index, row in self.event_tagging_df.iterrows():
            for column in self.event_tagging_df:
                val = row[column]
                if val is not None:
                    pattern = r'\[([^\[\]]*)\]'
                    val = re.sub(pattern, r'\1', val)
                    val = val.split()
                    for word in val:
                        check_word = word.lower()
                        if check_word not in self.choice_words:
                            if check_word not in banned_words:
                                self.choice_words.append(check_word)
                                self.choice_words_occurrences.append(0)

        self.word_counts = [[0] * len(self.choice_words) for _ in range(len(self.event_names))]
        self.word_counts_per_event = [0] * len(self.choice_words_occurrences)
        i = 0
        for index, row in self.event_tagging_df.iterrows():
            for column in self.event_tagging_df:
                val = row[column]
                if val is not None:
                    pattern = r'\[([^\[\]]*)\]'
                    val = re.sub(pattern, r'\1', val)
                    val = val.split()
                    for word in val:
                        check_word = word.lower()
                        if check_word not in banned_words:
                            index_of_word = self.choice_words.index(check_word)
                            self.choice_words_occurrences[index_of_word] += 1
                            self.word_counts[i][index_of_word] += 1
                            self.word_counts_per_event[i] += 1
            i += 1
        return self.word_counts

    def find_most_likely_event(self, d):
        event_probs = [0] * len(self.event_names)
        card_count = 0
        for ent in d.ents:
            if ent.label_ == "CARD":
                card_count += 1
            elif ent.label_ == "EVENT":
                return ent.text
        max_posterior = -1
        most_likely_event = None
        for i in range(len(event_probs)):
            for token in d:
                check_word = token.text.lower()
                if check_word in self.choice_words:
                    index_of_word = self.choice_words.index(check_word)
                    log_prob = math.log(((self.word_counts[i][index_of_word] + 1) / (
                            self.choice_words_occurrences[index_of_word] + self.word_counts_per_event[i] + len(self.word_counts))), math.e)
                    event_probs[i] += log_prob
            # noinspection PyTypeChecker
            event_probs[i] = math.e ** event_probs[i]
            if event_probs[i] > max_posterior:
                max_posterior = event_probs[i]
                most_likely_event = self.event_names[i]

        if max_posterior < 5 * 10 ** -15:
            self.small_val_flag = 1
        if card_count >= 3:
            return "card_selection"
        else:
            return most_likely_event

    def make_choice_from_event(self, event_name):
        event_row = self.event_percentages_df[self.event_percentages_df['Event Name'] == event_name]
        event_row = event_row.iloc[0]
        event_row = event_row.dropna()
        choice_scores = [0] * math.floor((len(event_row) / 2))
        choice_indexes = [0] * math.floor((len(event_row) / 2))
        i = 1
        while i < len(event_row):
            choice_total = float(event_row.iloc[i])
            choice_percent = float(event_row.iloc[i + 1]) / 100
            # noinspection PyTypeChecker
            choice_scores[math.floor(i / 2)] = choice_total * choice_percent
            choice_indexes[math.floor(i / 2)] = math.ceil(i / 2)
            i += 2
        norm_choice_scores = [float(i) / sum(choice_scores) for i in choice_scores]
        choice_index = random.choices(choice_indexes, weights=norm_choice_scores, k=1)[0]
        print("Choice Index: ", choice_index)
        print("Scores: ", norm_choice_scores)

        event_choice_row = self.event_tagging_df[self.event_tagging_df['Event Name'] == event_name]
        event_choice_row = event_choice_row.iloc[0]
        print("choice: ", event_choice_row.iloc[choice_index])
        return event_choice_row.iloc[choice_index]

    def form_response(self, d):
        response_event_name = self.find_most_likely_event(d)
        response_event_choice = self.make_choice_from_event(response_event_name)
        if not self.small_val_flag:
            prompt_response = random.choice(
                [
                    "For the event " + response_event_name + random.choice(
                        [
                            " I would suggest ",
                            " the best choice would be ",
                            " this choice is the best: ",
                            " my recommendation is: ",
                            " your best bet would be ",
                            " I would recommend ",
                            " you should do this: "
                        ]
                    ) + response_event_choice,
                    "If you are encountering " + response_event_name + random.choice(
                        [
                            " I would suggest ",
                            " the best choice would be ",
                            " this choice is the best: ",
                            " my recommendation is: ",
                            " your best bet would be ",
                            " I would recommend ",
                            " you should do this: "
                        ]
                    ) + response_event_choice,
                    random.choice(
                        [
                            "I would suggest ",
                            "the best choice would be ",
                            "this choice is the best: ",
                            "my recommendation is: ",
                            "your best bet would be ",
                            "I would recommend ",
                            "you should do this: "
                        ]
                    ) + response_event_choice
                ]
            )
        else:
            prompt_response = "I think you are talking about " + response_event_name + " so i will suggest " + response_event_choice + "\nIf this is not the correct event consider prompting again with the event name or more information on the choices you have."
        return prompt_response


def response_generator(response_model, p):
    doc = response_model.process_text(p, nlp)
    generated_response = response_model.form_response(doc)
    if not response_model.small_val_flag:
        random_num = random.choice([1, 2, 3, 4, 5, 6])
        if random_num == 1:
            generated_response += (
                "\nIf you think this is about the wrong event consider prompting again with the event name or "
                "more information on the choices you have.")
    response_model.small_val_flag = 0
    for word in generated_response.split():
        yield word + " "
        time.sleep(0.05)


nlp = spacy.load("en_core_web_sm")

card_percentages_file = "https://raw.githubusercontent.com/sethmeh/CS6320.002-Project/main/DataFiles/STSCardStats.txt"
event_percentages_file = "https://raw.githubusercontent.com/sethmeh/CS6320.002-Project/main/DataFiles/STSEventStats.txt"
event_tagging_file = "https://raw.githubusercontent.com/sethmeh/CS6320.002-Project/main/DataFiles/STSEventTagging.txt"

model = Model(card_percentages_file, event_percentages_file, event_tagging_file)
events = model.events
cards = model.cards

# Add the component to the pipeline
nlp.add_pipe("recognize_events", before="ner")
nlp.add_pipe("recognize_cards", before='ner')

st.title("Slay The Spire Helper")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(model, prompt))
st.session_state.messages.append({"role": "assistant", "content": response})
