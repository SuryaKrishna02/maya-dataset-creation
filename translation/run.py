import sys
sys.path.append("../")
import csv
import cohere
import logging
import asyncio
from tqdm import tqdm
from utils.helpers import load_json, save_json
from utils.constants import COHERE_API_KEY, TRANSLATION_BATCH_SIZE
from translation.processing import process_batch_gpu, process_item


async def main():
    """
    The main function that orchestrates the entire process of loading data,
    processing it in batches, and saving the results.
    """
    input_file = 'blip_laion_cc_sbu_558k.json'
    output_file = 'hindi.json'
    csv_file = 'translations.csv'

    data = load_json(input_file)
    total_count = len(data)
    print(f"Total number of IDs: {total_count}")

    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'GPT Value Original', 'GPT Value Modified'])

    co = cohere.AsyncClient(api_key=COHERE_API_KEY)

    batch_size = TRANSLATION_BATCH_SIZE

    logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

    for i in range(0, total_count, batch_size):
        batch = data[i:i + batch_size]

        try:
            batch = process_batch_gpu(batch)
            print("GPU preprocessing successful")
        except Exception as e:
            print(f"GPU processing failed, falling back to CPU: {str(e)}")

        progress_str = f"Progress: {min(i + batch_size, total_count)}/{total_count} IDs processed"
        pbar = tqdm(total=len(batch), desc=f"Processing batch {i // batch_size + 1}: {progress_str}",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

        tasks = [process_item(item, co, csv_file) for item in batch]
        results = await asyncio.gather(*tasks)
        for item, result in zip(batch, results):
            if not result:
                logging.error(f"Failed to process item {item['id']}")
        pbar.update(len(results))

        save_json(data, output_file)
        pbar.close()
        if i > 20:
            break

    print("Processing complete!")

if __name__ == "__main__":
    asyncio.run(main())