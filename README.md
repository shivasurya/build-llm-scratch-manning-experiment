# build-llm-scratch-manning-experiment
Building LLM from scratch experiment via Manning MEAP

### Trained Model

1. Text Spam Classifier - published in [huggingface](https://huggingface.co/shivasuryas/spam_classifier)


### Using Model

```python
import pytorch

....
....
model_state_dict = torch.load("spam_classifier.pth", map_location=torch.device('cpu')) // modify based on CPU/GPU
    model.load_state_dict(model_state_dict)
text_1 = (
    "I love to go to Waterloo university and summer break"
    ""
)
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))
```
