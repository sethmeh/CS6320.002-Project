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
        if temp_tuple in model_events:
            # If the token is in events list, set its label to "EVENT"
            d.ents = list(d.ents) + [Span(d, i, i + 1, label="EVENT")]
        i += 1
    return d


@Language.component("recognize_cards")
def recognize_events(d):
    i = 0
    for token in d:
        temp_tuple = (token.text.lower(), 'CARD')
        if temp_tuple in model_cards:
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

    def __init__(self, cp_file, ep_file, et_file, cu_file):
        self.card_percentages_df = self.read_file_from_url(cp_file)
        self.event_percentages_df = self.read_file_from_url(ep_file)
        self.event_tagging_df = self.read_file_from_url(et_file)
        self.card_upgrades_df = self.read_file_from_url(cu_file)

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
                event_name = ent.text.replace('_', ' ')
                event_name_list = event_name.split(" ")
                event_name = ""
                for en in event_name_list:
                    if en != 'of' and en != 'and':
                        en = en.title()
                    event_name += en + " "
                return event_name[:-1]
        if card_count >= 2:
            for token in d:
                if token.text.lower() == "upgrade":
                    return "Upgrade Card"
            return "Card Selection"
        max_posterior = -1
        most_likely_event = None
        max_num_choice_words_in_doc = 0
        for i in range(len(event_probs)):
            current_num_choice_words_in_doc = 0
            for token in d:
                check_word = token.text.lower()
                if check_word in self.choice_words:
                    current_num_choice_words_in_doc += 1
                    index_of_word = self.choice_words.index(check_word)
                    log_prob = math.log(((self.word_counts[i][index_of_word] + 1) / (
                            self.choice_words_occurrences[index_of_word] + self.word_counts_per_event[i] + len(
                        self.word_counts))), math.e)
                    event_probs[i] += log_prob
            if event_probs[i] > 0:
                # noinspection PyTypeChecker
                event_probs[i] = math.e ** event_probs[i]
            if event_probs[i] > max_posterior:
                max_posterior = event_probs[i]
                most_likely_event = self.event_names[i]
            max_num_choice_words_in_doc = current_num_choice_words_in_doc
        if max_posterior <= 0:
            return None
        elif max_posterior < 5 * 10 ** -15 or max_num_choice_words_in_doc < 3:
            self.small_val_flag = 1
        return most_likely_event

    def make_choice_from_event(self, event_name, d):
        if event_name == "Card Selection":
            return self.choose_card(d)
        elif event_name == "Upgrade Card":
            return self.choose_upgrade(d)
        elif event_name is None:
            return "FAILED STATE"
        event_row = self.event_percentages_df[self.event_percentages_df['Event Name'] == event_name]
        event_row = event_row.iloc[0]
        event_row = event_row.dropna()
        choice_scores = [0] * math.floor((len(event_row) / 2))
        choice_indexes = [0] * math.floor((len(event_row) / 2))
        i = 1
        while i < len(event_row):
            choice_total = float(event_row.iloc[i])**0.2
            choice_percent = float(event_row.iloc[i + 1]) / 100
            # noinspection PyTypeChecker
            choice_scores[math.floor(i / 2)] = choice_total * choice_percent
            choice_indexes[math.floor(i / 2)] = math.ceil(i / 2)
            i += 2
        norm_choice_scores = [float(i) / sum(choice_scores) for i in choice_scores]
        choice_index = random.choices(choice_indexes, weights=norm_choice_scores, k=1)[0]

        event_choice_row = self.event_tagging_df[self.event_tagging_df['Event Name'] == event_name]
        event_choice_row = event_choice_row.iloc[0]

        return event_choice_row.iloc[choice_index]

    def choose_card(self, d):
        current_card_list = []
        for ent in d.ents:
            if ent.label_ == "CARD":
                current_card_list.append(ent.text.replace('_', ' ').title())
        card_list_scores = [0] * (len(current_card_list) + 1)
        i = 0
        for card in current_card_list:
            card_row = self.card_percentages_df[self.card_percentages_df['Card Name'] == card]
            choice_total = float(card_row.iloc[0, 1])**0.2
            choice_percent = float(card_row.iloc[0, 2]) / 100
            choice_skip_total = float(card_row.iloc[0, 3])**0.2
            choice_skip_percent = float(card_row.iloc[0, 4]) / 100
            # noinspection PyTypeChecker
            card_list_scores[i] += choice_total * choice_percent
            for j in range(len(card_list_scores)):
                if i != j:
                    card_list_scores[j] += choice_skip_total * choice_skip_percent
            i += 1

        current_card_list.append("Skip")
        norm_card_scores = [float(i) / sum(card_list_scores) for i in card_list_scores]
        final_card = random.choices(current_card_list, weights=norm_card_scores, k=1)[0]

        return final_card

    def choose_upgrade(self, d):
        current_card_list = []
        for ent in d.ents:
            if ent.label_ == "CARD":
                current_card_list.append(ent.text.replace('_', ' ').title())
        card_list_scores = [0] * len(current_card_list)
        i = 0
        for card in current_card_list:
            card_row = self.card_upgrades_df[self.card_upgrades_df['Card Name'] == card]
            nu_total = float(card_row.iloc[0, 1]) ** 0.2
            nu_percent = float(card_row.iloc[0, 2]) / 100
            u_total = float(card_row.iloc[0, 3]) ** 0.2
            u_percent = float(card_row.iloc[0, 4]) / 100
            card_list_scores[i] += nu_total * nu_percent - u_total * u_percent
            i += 1

        norm_card_scores = [float(i) / sum(card_list_scores) for i in card_list_scores]
        final_card = random.choices(current_card_list, weights=norm_card_scores, k=1)[0]

        return final_card

    def form_response(self, d):
        response_event_name = self.find_most_likely_event(d)
        response_event_choice = self.make_choice_from_event(response_event_name, d)
        if response_event_name == "Card Selection":
            if response_event_choice == "Skip":
                prompt_response = random.choice(
                    [
                        "Here I would skip.",
                        "In this case I would skip.",
                        "Do not take any of these cards.",
                        "Skip all of those.",
                        "I recommend you skip these cards."
                    ]
                )
            else:
                prompt_response = random.choice(
                    [
                        "The best card to take is " + response_event_choice,
                        "My suggestion here is " + response_event_choice,
                        response_event_choice + " is the card I would choose",
                        "The best choice here is " + response_event_choice,
                        "My suggestion is to take " + response_event_choice,
                        "I recommend " + response_event_choice
                    ]
                )
        elif response_event_name == "Upgrade Card":
            prompt_response = random.choice(
                [
                    "The best card to upgrade is " + response_event_choice,
                    "My suggestion here is " + response_event_choice,
                    response_event_choice + " is the card I would upgrade",
                    "The best choice here is " + response_event_choice,
                    "My suggestion is to upgrade " + response_event_choice,
                    "I recommend " + response_event_choice
                ]
            )
        elif response_event_choice == "FAILED STATE" or response_event_name is None:
            return "Sorry I didn't understand that. Please provide more information and keep your responses limited to question about Slay the Spire."
        else:
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
        if self.small_val_flag:
            prompt_response += ". \nI think you are talking about " + response_event_name + ". If this is not the correct event consider prompting again with the event name or more information on the choices you have."
            self.small_val_flag = 0
        return prompt_response


def response_generator(response_model, p):
    doc = response_model.process_text(p, nlp)
    generated_response = response_model.form_response(doc)

    for word in generated_response.split():
        yield word + " "
        time.sleep(0.05)


nlp = spacy.load("en_core_web_sm")

card_percentages_file = "https://raw.githubusercontent.com/sethmeh/CS6320.002-Project/main/DataFiles/STSCardStats.txt"
event_percentages_file = "https://raw.githubusercontent.com/sethmeh/CS6320.002-Project/main/DataFiles/STSEventStats.txt"
event_tagging_file = "https://raw.githubusercontent.com/sethmeh/CS6320.002-Project/main/DataFiles/STSEventTagging.txt"
card_upgrades_file = "https://raw.githubusercontent.com/sethmeh/CS6320.002-Project/main/DataFiles/STSCardStatsUpgrades.txt"

model = Model(card_percentages_file, event_percentages_file, event_tagging_file, card_upgrades_file)
model_events = model.events
model_cards = model.cards

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



