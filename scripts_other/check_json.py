import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compare_answers(file1, file2):
    data1 = load_json(file1)
    data2 = load_json(file2)

    # id 기준으로 매핑
    data1_dict = {item["id"]: item["output"]["answer"] for item in data1}
    data2_dict = {item["id"]: item["output"]["answer"] for item in data2}

    common_ids = set(data1_dict.keys()) & set(data2_dict.keys())
    total = len(common_ids)
    same = 0

    for id_ in sorted(common_ids, key=lambda x: int(x)):
        answer1 = data1_dict[id_].strip()
        answer2 = data2_dict[id_].strip()
        if answer1 == answer2:
            same += 1
        else:
            print(f"[불일치] ID: {id_}\n - file1: {answer1}\n - file2: {answer2}\n")

    print(f"\n총 비교 항목 수: {total}")
    print(f"일치한 항목 수: {same}")
    print(f"불일치한 항목 수: {total - same}")

# 사용 예시
compare_answers(
    "output/2025-07-27_10-30-56_kakaocorp_kanana-1.5-8b-instruct-2505_r_128_ra_256_rd_0.2_float16_sft/test_result/test_results_896cb8a9_2025-07-27_12-54-01.json",
    "output/2025-07-27_10-30-56_kakaocorp_kanana-1.5-8b-instruct-2505_r_128_ra_256_rd_0.2_float16_sft/test_result/test_results_896cb8a9_2025-07-31_15-26-35.json"
)
