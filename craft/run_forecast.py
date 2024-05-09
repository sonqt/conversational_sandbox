from convokit import Corpus, download
from torch import optim
import torch.nn.functional as F
import torch

from utils import *
from ContextRNN import ContextEncoderRNN, SingleTargetClf
from UtteranceRNN import EncoderRNN
def calculate_f1_score(labels, logits):
    preds = torch.sigmoid(logits) > 0.5
    # Calculating precision, recall, and F1 score using PyTorch
    TP = ((preds == 1) & (labels == 1)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1
def evaluate(encoder, context_encoder, attack_clf, device, val_utterances, val_conversationLength, val_labels):
    encoder.eval()
    context_encoder.eval()
    attack_clf.eval()
    val_loss = 0
    val_f1 = 0
    val_accuracy = 0
    val_batches = 0
    pos = 0
    for i in range(len(val_utterances)):
        batch_utterances = val_utterances[i]
        batch_conversationLength = val_conversationLength[i]
        batch_labels = val_labels[i]
        batch_size = len(batch_labels)
        if batch_size == 0:
            continue
        with torch.no_grad():
            utt_outputs, utt_hidden = encoder.forward(batch_utterances)
            context_features = prepare_context_batch(utt_hidden, batch_conversationLength)
            context_features = torch.from_numpy(context_features).to(device)
            context_outputs, context_hidden = context_encoder(context_features)
            logits = attack_clf(context_hidden)
            labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            # pos_weight = torch.tensor([1]).type_as(logits)
            # loss_fct = BCEWithLogitsLoss(pos_weight=pos_weight, reduction = 'sum')
            # loss = loss_fct(logits, labels)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            val_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            pos += preds.sum().item() / len(batch_labels)
            val_f1 += calculate_f1_score(labels.cpu().detach(), preds.cpu().detach())
            val_accuracy += (preds == labels).sum().item() / len(batch_labels)
            val_batches += 1
    return val_loss / val_batches, val_f1 / val_batches, val_accuracy/val_batches, pos/len(val_labels)
def main():
    corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
    train_utterances, train_conversationLength, train_comment_ids,\
        train_labels = load_data(corpus, split='train', last_only=True)
    valid_utterances, valid_conversationLength, valid_comment_ids,\
        valid_labels = load_data(corpus, split='val', last_only=True)

    # Define Model
    hidden_size = 500
    context_encoder_n_layers = 2
    encoder_n_layers = 2
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderRNN(device=device, hidden_size=hidden_size, 
                        embedding_dim=hidden_size, max_utterance_len = 300,
                            n_layers=encoder_n_layers, dropout=dropout)
    context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
    attack_clf = SingleTargetClf(hidden_size, dropout)
    checkpoint = torch.load("craft_pretrained.tar")
    context_sd = checkpoint['ctx']
    encoder_sd = checkpoint['en']
    embedding_sd = checkpoint['embedding']

    context_encoder.load_state_dict(context_sd)
    encoder.embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)

    # Move models to GPU
    # encoder = encoder.to(device) inside the UtteranceRNN.py
    context_encoder = context_encoder.to(device)
    attack_clf = attack_clf.to(device)


    # Training parameters
    learning_rate = 1e-5
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    context_encoder_optimizer = optim.Adam(context_encoder.parameters(), lr=learning_rate)
    attack_clf_optimizer = optim.Adam(attack_clf.parameters(), lr=learning_rate)

    num_steps = 0
    for epoch in range(30):
        for batch_idx in range(len(train_labels)):
            num_steps += 1
            if num_steps % 50 == 0:
                print(num_steps, epoch)
                val_loss, val_f1, val_accuracy, pos= evaluate(encoder, context_encoder, attack_clf, device, valid_utterances, valid_conversationLength, valid_labels)
                print("Validation loss: {:.2f} accuracy: {:.2f} f1: {:.2f} pos: {:.2f}".format(val_loss, val_accuracy * 100, val_f1 * 100, pos))
            encoder.train()
            context_encoder.train()
            attack_clf.train()
            
            batch_utterances = train_utterances[batch_idx]
            batch_conversationLength = train_conversationLength[batch_idx]
            batch_comment_ids = train_comment_ids[batch_idx]
            batch_labels = train_labels[batch_idx]
            encoder_optimizer.zero_grad()
            context_encoder_optimizer.zero_grad()
            attack_clf_optimizer.zero_grad()
            
            encoder_outputs, hidden = encoder.forward(batch_utterances)
            context_features = torch.from_numpy(prepare_context_batch(hidden, batch_conversationLength)).to(device)
            final_outputs, final_hidden = context_encoder.forward(context_features)
            logits = attack_clf(final_hidden)
            labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            loss.backward()
            clip = 50.0
            # Clip gradients: gradients are modified in place
            _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(attack_clf.parameters(), clip)

            # Adjust model weights
            encoder_optimizer.step()
            context_encoder_optimizer.step()
            attack_clf_optimizer.step()

if __name__ == "__main__":
    main()