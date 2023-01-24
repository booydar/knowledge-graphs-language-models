import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import json
import time


class HitsCalculator:
    def __init__(self, emb_model='all-MiniLM-L6-v2', index_path="faiss/entities.index", 
    entities_path='faiss/large_verbalized_inference_entities.json', drop_description=False):
        self.emb_model = SentenceTransformer(emb_model)
        self.index=faiss.read_index(index_path)
        
        if entities_path is not None:
            with open(entities_path, 'r') as f:
                self.entities = json.load(f)
        else:
            self.entities = None

    def hits(self, outputs, labels, p_entity, y_entity, tokens_to_replace=(" [SEP-2]", " [SEP-3]")):
        for token in tokens_to_replace:
            outputs = list(map(lambda x: x.replace(token, ""), outputs))

        vectors = self.emb_model.encode(outputs)
        _, indices = self.index.search(vectors, 10)

        hits = {"Hits@1": 0, "Hits@3": 0, "Hits@5": 0, "Hits@10": 0}

        for i, label in enumerate(labels):
            target = int(label[1:]) 

            if target == indices[i][0]:
                hits['Hits@1'] += 1
                hits['Hits@3'] += 1
                hits['Hits@5'] += 1
                hits['Hits@10'] += 1
            
            elif target in indices[i][:3]:
                hits['Hits@3'] += 1
                hits['Hits@5'] += 1
                hits['Hits@10'] += 1

            elif target in indices[i][:5]:
                hits['Hits@5'] += 1
                hits['Hits@10'] += 1
            
            elif target in indices[i][:10]:
                hits['Hits@10'] += 1
            
        return { metric: hits[metric]/len(labels) for metric in hits.keys() }
        

    def hits_pipeline(self, outputs, labels, p_entity, y_entity, tokens_to_replace=(" [SEP-2]", " [SEP-3]")):
        if self.entities is None:
            return self.hits(self, outputs, labels, p_entity, y_entity, tokens_to_replace)
            
        for token in tokens_to_replace:
            outputs = list(map(lambda x: x.replace(token, ""), outputs))

        vectors = self.emb_model.encode(outputs)
        _, indices = self.index.search(vectors, 10)

        hits = {"Hits@1": 0, "Hits@3": 0, "Hits@5": 0, "Hits@10": 0}

        for i, label in enumerate(labels):
            target = int(label[1:]) 

            if p_entity[i] in self.entities:
                if p_entity[i] == y_entity[i]:
                    hits['Hits@1'] += 1
                    hits['Hits@3'] += 1
                    hits['Hits@5'] += 1
                    hits['Hits@10'] += 1

            elif target == indices[i][0]:
                hits['Hits@1'] += 1
                hits['Hits@3'] += 1
                hits['Hits@5'] += 1
                hits['Hits@10'] += 1
            
            elif target in indices[i][:3]:
                hits['Hits@3'] += 1
                hits['Hits@5'] += 1
                hits['Hits@10'] += 1

            elif target in indices[i][:5]:
                hits['Hits@5'] += 1
                hits['Hits@10'] += 1
            
            elif target in indices[i][:10]:
                hits['Hits@10'] += 1
            
        return { metric: hits[metric]/len(labels) for metric in hits.keys() }
