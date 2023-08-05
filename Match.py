import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity

class Facematch():
    def __init__(self):
        pass

    def get_similarity(self, x: torch.Tensor, y: torch.Tensor):

        similarity_obj = nn.CosineSimilarity(dim=0)
        similarity = similarity_obj(x, y)
        return similarity

    def get_confidante_score(self, x: torch.Tensor, y: torch.Tensor):
        similarity = torch.tensor(cosine_similarity(x.detach().numpy(), y.detach().numpy()))
        adjusted_cosine_similarity = (similarity + 1) / 2
        # print(adjusted_cosine_similarity)
        return adjusted_cosine_similarity.item()