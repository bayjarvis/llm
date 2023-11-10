from datasets import Dataset, load_dataset
from mistral.dpo.config import Config
import warnings
warnings.filterwarnings("ignore")

def dpo_data(dataset_id, split:str='train_prefs') -> Dataset:

    dataset = load_dataset(
        dataset_id,
        split = split,
        use_auth_token=True
    )

    original_columns = dataset.column_names

    def return_prompt_and_responses(samples):
        return {
            "prompt": samples["prompt"],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"]
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )
    
# Create triple (prompt, chosen, rejected) dataset
def create_dataset(dataset_id, split='train_prefs'):
    dataset =dpo_data(dataset_id, split=split)
    df = dataset.to_pandas()
    df["chosen"] = df["chosen"].apply(lambda x: x[1]["content"])
    df["rejected"] = df["rejected"].apply(lambda x: x[1]["content"])
    df = df.dropna()
    dataset = Dataset.from_pandas(df)
    return dataset

if __name__ == '__main__':
    config = Config()
    database_id = config.DATASET_ID
    data = dpo_data(database_id)
    print("Dataset loaded")
    print(data)
    df = data.to_pandas()
    print("Dataset converted to pandas dataframe")
    print(df.head(1))
    triple_dataset = create_dataset(database_id)
    print("Triple dataset created for DPO training")
    print(triple_dataset.to_pandas().head(1))