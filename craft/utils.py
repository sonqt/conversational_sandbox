import numpy as np
def loadPairs(corpus, split=None, last_only=False):
    """
    Load context-reply pairs from the Corpus, optionally filtering to only conversations
    from the specified split (train, val, or test).
    Each conversation, which has N comments (not including the section header) will
    get converted into N-1 comment-reply pairs, one pair for each reply
    (the first comment does not reply to anything).
    Each comment-reply pair is a tuple consisting of the conversational context
    (that is, all comments prior to the reply), the reply itself, the label (that
    is, whether the reply contained a derailment event), and the comment ID of the
    reply (for later use in re-joining with the ConvoKit corpus).
    The function returns a list of such pairs.
    """
    pairs = []
    count_attack = 0
    count_convo = 0
    for convo in corpus.iter_conversations():
        # consider only conversations in the specified split of the data
        if split is None or convo.meta['split'] == split:
            count_convo += 1
            utterance_list = []
            for utterance in convo.iter_utterances():
                if utterance.meta['is_section_header']:
                    continue
                if utterance.meta['comment_has_personal_attack']:
                    count_attack += 1
                utterance_list.append({"text": utterance.text, 
                                        "is_attack": int(utterance.meta['comment_has_personal_attack']), 
                                        "id": utterance.id})
                
            iter_range = range(1, len(utterance_list)) if not last_only else [len(utterance_list)-1]
            for idx in iter_range:
                reply = utterance_list[idx]["text"]
                label = utterance_list[idx]["is_attack"]
                comment_id = utterance_list[idx]["id"]
                # gather as context all utterances preceding the reply
                context = [u["text"] for u in utterance_list[:idx]]
                pairs.append((context, reply, label, comment_id))

    return pairs
def conversations2utterances(conversations):
    """
    Convert list of conversations into list of utterances for UtteranceModel.
    INPUT:
        conversations: list of list of str
            List of conversations, each conversation is a list of utterances.
    OUTPUT:
        utterances: list of str
            List of utterances in the dataset.
        conversationLength: list of int
            List of number of utterances in conversations.
    """
    conversationLength = [len(convo) for convo in conversations]
    utterances = []
    for convo in conversations:
        for utterance in convo:
            utterances.append(utterance)
    # assert len(utterances) == sum(conversationLength)
    return utterances, conversationLength

def load_data(corpus, context_batch_size = 32, split=None, last_only=False):
    """
    Load data from corpus into the format ready for UtteranceModel.
    INPUT:
        corpus: convokit.Corpus
        split: str, optional
            If specified, only consider conversations in the specified split of the data.
        last_only: bool, optional
            If True, only consider the last utterance in each conversation.
    OUTPUT:
        utterances: list of str
            List of utterances in the dataset.
        conversationLength: list of int
            List of lengths of conversations in the dataset.
        comment_ids: list of str
            List of ids corresponding to the reply utterance.
        labels: list of int
            List of labels for each context if the next reply contains personal attack.
    """
    pairs = loadPairs(corpus, split, last_only)
    batch_labels = []
    batch_comment_ids = []
    batch_utterances = []
    batch_conversationLength = []
    conversations = []
    labels = []
    comment_ids = []
    for pair in pairs:
        if len(labels) == context_batch_size:
            utterances, conversationLength = conversations2utterances(conversations)
            batch_utterances.append(utterances)
            batch_conversationLength.append(conversationLength)
            batch_labels.append(labels)
            batch_comment_ids.append(comment_ids)
            assert len(conversationLength) == len(comment_ids) == len(labels)
            conversations = []
            labels = []
            comment_ids = []

        context, _, label, comment_id = pair
        conversations.append(context)
        labels.append(label)
        comment_ids.append(comment_id)
    if len(conversations) > 0:
        utterances, conversationLength = conversations2utterances(conversations)
        batch_utterances.append(utterances)
        batch_conversationLength.append(conversationLength)
        batch_labels.append(labels)
        batch_comment_ids.append(comment_ids)
    return batch_utterances, batch_conversationLength, batch_comment_ids, batch_labels

def prepare_context_batch(utt_hidden, batch_conversationLength, max_context_len=20):
    assert utt_hidden.shape[1] == sum(batch_conversationLength)
    utt_encoder_summed = utt_hidden[-2,:,:] + utt_hidden[-1,:,:]
    hidden_size = utt_encoder_summed.shape[1]
    context_features = np.zeros((len(batch_conversationLength), max_context_len, hidden_size), dtype=np.float32)

    current_utt_idx = 0
    for i, convo_len in enumerate(batch_conversationLength):
        if convo_len > max_context_len:
            current_utt_idx += convo_len - max_context_len
            convo_len = max_context_len
        context_features[i, -convo_len:, :] = np.array(utt_encoder_summed)[current_utt_idx:current_utt_idx+convo_len, :]
        current_utt_idx += convo_len
    return context_features