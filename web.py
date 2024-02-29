import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from   random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained = torch.load(r'C:\Users\Tairo Kageyama\Documents\GitHub\Python-fo-Natural-Language-Processing-main\lab5\model\S-BERT.pt')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.load_state_dict(trained)
model.eval()

def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool


def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_a = tokenizer(sentence_a, return_tensors='pt', truncation=True, padding=True).to(device)
    inputs_b = tokenizer(sentence_b, return_tensors='pt', truncation=True, padding=True).to(device)

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a['input_ids']
    attention_a = inputs_a['attention_mask']
    inputs_ids_b = inputs_b['input_ids']
    attention_b = inputs_b['attention_mask']

    # Extract token embeddings from BERT
    u = model(inputs_ids_a, attention_mask=attention_a)[0]  # all token embeddings A = batch_size, seq_len, hidden_dim
    v = model(inputs_ids_b, attention_mask=attention_b)[0]  # all token embeddings B = batch_size, seq_len, hidden_dim

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    # print(u, v)
    # print(u.reshape(1, -1), v.reshape(1, -1))
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score

# Example usage:
# sentence_a = 'Your contribution helped make it possible for us to provide our students with a quality education.'
# sentence_b = "Your contributions were of no help with our students' education."
# similarity = calculate_similarity(model, tokenizer, sentence_a, sentence_b, device)
# print(f"Cosine Similarity: {similarity:.4f}")


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='input-1', type='text', value='1st sentence'),
    dcc.Input(id='input-2', type='text', value='2nd sentence'),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output')
])


@app.callback(
    Output('output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('input-1', 'value')],
    [Input('input-2', 'value')]
)


def update_output(n_clicks, input1, input2):
    if n_clicks > 0:
        print(input1)
        print(input2)
        similarity = calculate_similarity(model, tokenizer, input1, input2, device)

        return f"You entered: {similarity}"
    else:
        return ""


if __name__ == '__main__':
    app.run_server(debug=True)
