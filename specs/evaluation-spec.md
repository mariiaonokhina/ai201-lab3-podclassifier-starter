# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
[blank — write out the accuracy formula in plain English.
 What counts as "correct"? What do you divide by?]

 Accuracy is the number of predictions that exactly match the ground-truth labels divided by the total number of predictions. A prediction counts as correct only when the predicted label is identical to the corresponding ground-truth label, and you divide that count by the length of the list.
```

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. ...
 2. ...
 3. ...]


1. Initialize a counter for correct predictions.
2. Loop over each pair of `predicted` and `truth` labels.
3. For each pair, if they are exactly equal, increment the correct counter.
4. After the loop, divide the correct count by the total number of predictions.
5. Return that fraction as the accuracy.
```

---

**Edge case — what if both lists are empty?**

```
[blank — what should the function return? Why?]


If both lists are empty, return `0.0` because there are no predictions to evaluate; with no examples, accuracy should fall back to zero rather than causing a divide-by-zero or an undefined result.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

[blank — what does compute_accuracy() return for these inputs? Show your work.]


- `interview` vs `interview` -> correct
- `solo` vs `solo` -> correct
- `panel` vs `solo` -> incorrect
- `interview` vs `narrative` ->  incorrect

So 2 correct out of 4 total predictions, giving accuracy = `2 / 4 = 0.5`.
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
[blank — be precise. When does an episode count as correctly classified
 for the "interview" class, for example?]


 An episode is correct for a class only when its ground-truth label is that class and the prediction matches that same class exactly. For example, an episode counts as correctly classified for `interview` only if `ground_truth == "interview"` and `prediction == "interview"`.
```

---

**What does "total" mean for a given class?**

```
[blank — is "total" the total number of predictions, or something more specific?]


For a given class, `total` means the number of episodes whose ground-truth label is that class, not the total number of predictions. It’s the number of true examples of that class in the evaluation set, which is the denominator used to compute per-class accuracy.
```

---

**Step-by-step logic:**

```
[blank — describe the steps your code will take.
 1. Initialize ...
 2. Loop over ...
 3. For each pair (predicted, truth) ...
 4. After the loop ...
 5. Return ...]


1. Initialize a stats dict for each label in `VALID_LABELS`, with `correct = 0`, `total = 0`, and `accuracy = 0.0`.
2. Loop over each pair of `predicted` and `truth` labels.
3. For each pair, increment `total` for the class given by `truth`; if `predicted == truth`, also increment `correct` for that class.
4. After the loop, compute `accuracy` for each class as `correct / total` when `total > 0`, otherwise leave `accuracy` at `0.0`.
5. Return the dict keyed by label with each class’s `correct`, `total`, and `accuracy`.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
[blank — what should accuracy be set to? Why?
 Hint: look at the docstring in evaluate.py.]


 If a class has no examples in `ground_truth`, set its per-class accuracy to `0.0` because there is no data to evaluate that class, and the spec/docstring defines accuracy as `correct / total`, with `0.0` when `total == 0`.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

[blank — fill in the per-class results table below]

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

   ```
   Per-class accuracy is more informative because overall accuracy is an average across all labels.

   If one class is very weak but the others are strong, the overall score can still look okay.
   Per-class accuracy shows which labels are failing, so you can diagnose whether the classifier is biased or simply not learning certain categories.
   ```

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

   ```
   If panel episodes are consistently misclassified as interview, it suggests a label or prompt issue.

   The model may not be getting a strong enough distinction between “multiple guests with equal time” and “host plus guest.”
   That could mean your training examples for panel and interview are too similar, or the prompt definitions are not clear enough about the difference.
   ```

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
   ```
   More data usually makes evaluation more reliable.

   With 100 labeled training episodes, the classifier should have more examples to learn the differences between labels, which can improve generalization and reduce overfitting.
   With 200 test episodes, the reported accuracy becomes more stable and less sensitive to random noise, so you get a better estimate of true performance rather than a few lucky or unlucky examples.
   ```
