# import dataloader
from torch.utils.data import Dataset
import json, jsonlines


class FavaBenchDataset(Dataset):
    def __init__(self, json_file="./data/rarr-input_fava_bp.jsonl",
                       claim_field="claim", artificial_fact_path=None):
        # self.data = []
        if artificial_fact_path is not None:
            self.artificial_facts = []
            self.artificial_evidence = []
            with open(artificial_fact_path) as f:
                for line in f:
                    item = json.loads(line)
                    question_list =  item["result"]["question_gen"]['questions']
                    self.artificial_facts.append(question_list)
                    self.artificial_evidence.append(item["result"]["search"]['used_evidence'])
        else:
            self.artificial_facts = None
            self.artificial_evidence = None

        self.data = list(jsonlines.open(json_file))
        self.claim_field = claim_field


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_claim(self, idx):
        return self.data[idx]["input_info"][self.claim_field]

    def load_rarr_results(self, output_file):
        finished_results = {
            l["input_info"][self.claim_field]: l["result"]
            for l in jsonlines.open(output_file)
        }
        return finished_results


class WikibibDataset(Dataset):
    def __init__(self, json_file="./data/wikibib_halluc/labeled/ChatGPT.jsonl",
                       artificial_fact_path=None, per_sentence=False):
        if artificial_fact_path is not None:
            self.artificial_facts = []
            self.artificial_evidence = []
            with open(artificial_fact_path) as f:
                for line in f:
                    item = json.loads(line)
                    question_list =  item["result"]["question_gen"]['questions']
                    self.artificial_facts.append(question_list)
                    self.artificial_evidence.append(item["result"]["search"]['used_evidence'])
        else:
            self.artificial_facts = None
            self.artificial_evidence = None

        self.data = []
        with open(json_file) as f:
            for line in f:
                self.data.append(json.loads(line))

        if per_sentence:
            self.question_key = "question"
            self.llm_reply_key = "decon_sentences"
        else:
            self.question_key = "input"
            self.llm_reply_key = "output"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]

        question  = current_sample[self.question_key]
        llm_reply = current_sample[self.llm_reply_key]
        topic = current_sample["topic"]
            # annotations = current_sample["annotations"]# ["human-atomic-facts"]
            # annotations represent the list of sentences, where each sentence
            #   is a dictionary with the "text" and "label" keys, Labels being "S", "NS" or "IR"
        return {
            "question": question,
            "llm_reply": llm_reply,
            "topic": topic,
            # "annotations": annotations,
        }

    def load_rarr_results(self, output_file):
        finished_results = {}
        for l in jsonlines.open(output_file):
            finished_results[l[self.llm_reply_key]] = l["result"]
        return finished_results

    def get_claim(self, idx, per_sentence=False):

        current_sample = self.data[idx]
        return current_sample[self.llm_reply_key]

    def get_gt_facts(self, idx):
        """
        Get the facts for a given index
        Args:
            idx:

        Returns:
            evidence_list: List of evidence
            label_list: List of labels
            topic: The name of the Bibliography topic
        """
        item = self.data[idx]
        # print(item.keys(), item['topic'])

        label_mapping = {"S": 1, "NS": 0, "IR": -1}
        evidence_list, label_list = [], []

        # if annotations are null, return empty lists
        if item["annotations"] is not None:
            # print(item["annotations"])
            for sentence in item["annotations"]:
                atomic_facts = sentence["human-atomic-facts"]
                #print("Atomic Facts: ", atomic_facts)
                if atomic_facts is not None:
                    for fact in atomic_facts:
                        evidence_list.append(fact["text"])
                        label_list.append(label_mapping[fact["label"]])
        return evidence_list, label_list, item["topic"]



if __name__ == "__main__":
    json_file = "./data/wikibib_halluc/labeled/ChatGPT.jsonl"
    dset = WikibibDataset(json_file)
    print("Wiki Size: ", len(dset))

    print("Question", dset[0]["question"])
    print("Reply", dset[0]["llm_reply"])
    for sentence in dset[0]["annotations"]:
        print("\nNew Sentence: ",sentence["text"])
        atomic_facts = sentence["human-atomic-facts"]
        for fact in atomic_facts:
            print(fact["text"], fact["label"])