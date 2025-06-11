import json

import pandas as pd


def save_output(output_list: list | pd.DataFrame, output_file: str):
    if isinstance(output_list, pd.DataFrame):
        df = output_list
    else:
        df = pd.DataFrame(output_list)
    if output_file.endswith(".xlsx"):
        df.to_excel(output_file, index=False)
    elif output_file.endswith(".jsonl"):
        df.to_json(output_file, lines=True, force_ascii=False)
    elif output_file.endswith(".csv"):
        df.to_csv(output_file, index=False, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported output file format: {output_file}")


def jsonl_to_xlsx(jsonl_file, xlsx_file):
    jsonl_df = pd.read_json(jsonl_file, lines=True)
    jsonl_df.to_excel(xlsx_file, index=False)
    jsonl_df.to_csv(xlsx_file.replace(".xlsx", ".csv"), index=False)
    print(f"Finished convert {jsonl_file}")

