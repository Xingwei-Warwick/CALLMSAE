from nyt_parser import NYTArticle
from os import listdir
from os.path import isfile
import json
import pandas as pd


FILTER_LIST = ["bomb", "terrorism", "murder", "riots", "hijacking", "assassination", "kidnapping", "arson", "vandalism", "hate crime", "serial murder", "manslaughter", "extortion"]
FILTER_SET = set(FILTER_LIST)

def load_NYT():
    count_by_year = []
    data_dict = {}
    sentence_count = 0
    total_selected = 0
    descriptors_count = {}
    descriptors_doc = {}

    stats_table = {
        "doc_id": [],
        "word_count": [],
        "descriptors": [],
        "year": [],
        "month": []
    }
    dis_card_stats_table = {
        "doc_id": [],
        "word_count": [],
        "descriptors": [],
        "year": [],
        "month": []
    }
    for i in range(1987, 2008):
        count = 0

        for month_name in listdir(f"NYT_annotated/data/{i}"):
            if isfile(f"NYT_annotated/data/{i}/{month_name}"):
                continue
            for date_name in listdir(f"NYT_annotated/data/{i}/{month_name}"):
                if isfile(f"NYT_annotated/data/{i}/{month_name}/{date_name}"):
                    continue
                for file_name in listdir(f"NYT_annotated/data/{i}/{month_name}/{date_name}"):
                    if file_name.split('.')[-1] == 'xml':
                        f_id = file_name.split('.')[0]
                        with open(f"NYT_annotated/data/{i}/{month_name}/{date_name}/{file_name}", 'r', encoding='utf8') as f:
                            this_doc = NYTArticle.from_file(f)
                        this_doc = this_doc.as_dict()
                        if this_doc.get("abstract") is None:
                            continue
                        this_descriptors = []

                        for d in this_doc["descriptors"]:
                            this_descriptors.append(d)
                        this_descriptors = set(this_descriptors)

                        abstract = this_doc['abstract']
                        title = this_doc['title']
                        for des in this_descriptors:
                            if descriptors_count.get(des) is None:
                                descriptors_count[des] = 0
                            descriptors_count[des] += 1
                            if descriptors_doc.get(des) is None:
                                descriptors_doc[des] = []
                            
                        out_doc_dict = {
                            "title": title,
                            "abstract": abstract,
                            "text": '\n\n'.join(this_doc["paragraphs"]),
                            "descriptors": ";".join(this_descriptors),
                        }

                        word_count = len(out_doc_dict["text"].split())
                        stats_table["word_count"].append(word_count)
                        stats_table["descriptors"].append(out_doc_dict["descriptors"])
                        stats_table["year"].append(i)
                        stats_table["month"].append(month_name)
                        stats_table["doc_id"].append(f"{i}_{month_name}_{date_name}_{f_id}")
                        
                        if word_count < 100 or word_count > 10000:
                            dis_card_stats_table["word_count"].append(word_count)
                            dis_card_stats_table["descriptors"].append(out_doc_dict["descriptors"])
                            dis_card_stats_table["year"].append(i)
                            dis_card_stats_table["month"].append(month_name)
                            dis_card_stats_table["doc_id"].append(f"{i}_{month_name}_{date_name}_{f_id}")
                            continue

                        with open(f"raw_data/{i}_{month_name}_{date_name}_{f_id}.json", 'w') as f:
                            f.write(json.dumps(out_doc_dict, indent=4))
                        data_dict[f"{i}_{month_name}_{date_name}_{f_id}"] = this_doc
                        count += 1
                        total_selected += 1
                        sentence_count += len(this_doc["paragraphs"])
        count_by_year.append(count)

    print(count_by_year)
    print(sentence_count)
    print(total_selected)
    stats_df = pd.DataFrame(stats_table)
    stats_df.to_csv('data_stats/NYT_temp2_p_stats.csv', index=False, sep='\t', encoding='utf-8')
    stats_df.to_excel('data_stats/NYT_temp2_p_stats.xlsx', index=False)
    dis_card_stats_df = pd.DataFrame(dis_card_stats_table)
    dis_card_stats_df.to_csv('data_stats/NYT_temp2_p_discard_stats.csv', index=False, sep='\t', encoding='utf-8')
    dis_card_stats_df.to_excel('data_stats/NYT_temp2_p_discard_stats.xlsx', index=False)
    #print(descriptors_count)

    print(f"Load {len(data_dict)} files!")
    return data_dict

if __name__ == "__main__":
    data_dict =  load_NYT()



    
