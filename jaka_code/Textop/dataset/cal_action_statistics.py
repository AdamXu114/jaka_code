import joblib
from pathlib import Path
import json
import argparse

def cal_action_statistics(data_folder, json_filename):
    dataset = data_folder + '/train.pkl'
    data = joblib.load(dataset)
    action_statistics = {}
    for seg in data:
        frame_labels = seg['frame_ann']
        for label in frame_labels:
            act_cat_list = label[3]
            duration = label[1] - label[0]
            for act_cat in act_cat_list:
                if act_cat not in action_statistics:
                    action_statistics[act_cat] = {
                        'total_weight': 1,
                        'total_len': 0
                    }
                action_statistics[act_cat]['total_len'] += duration

    for act_cat in action_statistics:
        action_statistics[act_cat]['weight'] = action_statistics[act_cat]['total_weight'] / action_statistics[act_cat]['total_len']

    export_path = Path(json_filename)

    with export_path.open('w') as f:
        json.dump(action_statistics, f, indent=4)

    print(f"Action statistics saved to {export_path}")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_folder', type=str, required=True)
    parse.add_argument('--trg_filename', type=str, required=True, help='json filename')
    args = parse.parse_args()
    cal_action_statistics(args.data_folder, args.trg_filename)