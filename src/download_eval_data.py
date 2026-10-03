from datasets import load_dataset

ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="validation")
df = ds.to_pandas()
df.to_csv("data/tfns_validation.csv", index=False)
print(df.shape)
print(df.head())