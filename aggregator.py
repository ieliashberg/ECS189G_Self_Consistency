from collections import Counter

def aggregate_answers(answer_list):
    if not answer_list:
        return None
    counts = Counter(answer_list)
    return counts.most_common(1)[0][0]
    