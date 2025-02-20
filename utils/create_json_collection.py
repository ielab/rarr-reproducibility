import argparse
import os
import json
from tqdm import tqdm
from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate JSON collections from shards."
    )
    parser.add_argument(
        "--shards_dir",
        type=str,
        required=True,
        help="Path to the directory containing shard folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the JSONL files will be written."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Extract paths from arguments
    SHARDS_DIR = args.shards_dir
    OUTPUT_COLLECTION_DIR = args.output_dir

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_COLLECTION_DIR, exist_ok=True)

    # Get all shard folders that match the pattern 'shard_'
    shard_folders = sorted(
        f for f in os.listdir(SHARDS_DIR)
        if f.startswith("shard_") and os.path.isdir(os.path.join(SHARDS_DIR, f))
    )

    for shard_dir in shard_folders:
        # Each shard gets its own subfolder under OUTPUT_COLLECTION_DIR
        shard_out_dir = os.path.join(OUTPUT_COLLECTION_DIR, shard_dir)
        os.makedirs(shard_out_dir, exist_ok=True)

        # The .jsonl file for this shard
        shard_jsonl_path = os.path.join(shard_out_dir, "collection.jsonl")

        # Load shard’s dataset
        shard_path = os.path.join(SHARDS_DIR, shard_dir)
        ds = load_from_disk(shard_path)

        # Write out documents for this shard
        with open(shard_jsonl_path, "w", encoding="utf-8") as out_f:
            for row in tqdm(ds, desc=f"Processing {shard_dir}"):
                doc = {
                    "doc_id": row["id"],
                    "id": row["chunked_id"],
                    "url": row["url"],
                    "title": row["title"],
                    "contents": row["text"]
                }
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Done! Each shard has its own JSONL file in: {OUTPUT_COLLECTION_DIR}")


if __name__ == "__main__":
    main()