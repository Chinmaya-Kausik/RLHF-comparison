from datasets import load_dataset
dataset = load_dataset('openai/summarize_from_feedback', 'comparisons', split='train', streaming=True)
print(next(iter(dataset)))