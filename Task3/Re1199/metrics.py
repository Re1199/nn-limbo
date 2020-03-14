def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    len_predict = len(prediction)

    for i in range(len_predict):
        if prediction[i] == ground_truth[i]:
            if prediction[i]:
                tp += 1
            else:
                tn += 1
        elif prediction[i]:
            fp += 1
        else:
            fn += 1

    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if (tp + fn) != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    accuracy = (tp + tn) / len_predict

    if (precision + recall) != 0:
        f1 = (2 * recall * precision) / (precision + recall)
    else:
        f1 = 0

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    accuracy = 0
    len_predict = len(prediction)

    for i in range(len_predict):
        if prediction[i] == ground_truth[i]:
            accuracy += 1

    return accuracy / len_predict
