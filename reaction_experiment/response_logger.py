def label_responses(silence_times, response_times, margin=2.0):
    labels = []
    for s in silence_times:
        matched = any(s <= r <= s + margin for r in response_times)
        labels.append(1 if matched else 0)
    return labels
