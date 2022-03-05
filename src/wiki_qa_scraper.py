from bs4 import BeautifulSoup
import torch
import requests

base_url = "https://en.wikipedia.org/wiki/"


def get_wiki_info_for_topic(topic):
    url = base_url + topic
    page = requests.get(url).content
    soup = BeautifulSoup(page, 'html.parser')
    return "".join([p.text for p in soup.find_all('p')])

def get_wiki_answer_for_topic(topic):
    return get_wiki_info_for_topic(topic)[:1000].split(".")[0]

def get_wiki_answer_for_topic_using_nlp(topic):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

    context = get_wiki_info_for_topic(topic)[:1000]
    query = f"what is {topic}?"
    print(len(context.split()))
    print((context, query))
    indexed_tokens = tokenizer.encode(context, query, add_special_tokens=True, max_length=512, truncation=True)
    sep_idx = indexed_tokens.index(tokenizer.sep_token_id)
    len_question, len_answer = sep_idx + 1, len(indexed_tokens) - (sep_idx + 1)
    segment_ids = [0] * len_question + [1] * len_answer
    segment_tensors = torch.tensor([segment_ids])
    tokens_tensor = torch.tensor([indexed_tokens])

    with torch.no_grad():
        out = model(tokens_tensor, token_type_ids=segment_tensors)

    answer = tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
    return f"{topic} is {answer}"
    
while True:
    topic = input("Input a topic of interest: ")
    ans = get_wiki_answer_for_topic(topic)
    print(ans)