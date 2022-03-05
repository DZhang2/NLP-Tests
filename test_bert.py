import torch
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

question_answering_model = torch.hub.load('huggingface/pytorch-transformers', 'modelForQuestionAnswering', 'bert-large-uncased-whole-word-masking-finetuned-squad')
question_answering_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-large-uncased-whole-word-masking-finetuned-squad')

# The format is paragraph first and then question
# text_1 = "Jim Henson was a puppeteer"
# text_2 = "Who was Jim Henson?"
text_1 = "Novak Djokovic is the GOAT"
text_2 = "Who is the GOAT?"
indexed_tokens = question_answering_tokenizer.encode(text_1, text_2, add_special_tokens=True)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
segments_ids = [0,0,0,0,0,0,0,1,1,1,1,1,1,1]
segments_tensors = torch.tensor([segments_ids])
print(segments_tensors.shape)
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor.shape)

# Predict the start and end positions logits
with torch.no_grad():
    out = question_answering_model(tokens_tensor, token_type_ids=segments_tensors)

# get the highest prediction
answer = question_answering_tokenizer.decode(indexed_tokens[torch.argmax(out.start_logits):torch.argmax(out.end_logits)+1])
print(answer)

# Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions (set model to train mode before if used for training)
# start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
# multiple_choice_loss = question_answering_model(tokens_tensor, token_type_ids=segments_tensors, start_positions=start_positions, end_positions=end_positions)