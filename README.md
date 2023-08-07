# Face Match

Objective:- Identify the similarity between two faces

### Approach:- 
- Use VGG 16 pre-trained weights to get facial feature embeddings. 
- Use cosine similarity to calculate the similarity

### [Progress as of 8/Aug/2023]
- Moved away from the TinyVGG architecture to the pre-trained VGG16 weights to get better feature embedding.
Mostly it works if we keep the threshold to 70% similarity.

### [Next step]
- Moving from VGG16 to FaceNet or to OpenFace weights to get a better facial feature embedding 

### Result:-
<img align="left" src="https://github.com/deepakpillai/CNNFaceMatch/blob/main/Result.gif?raw=true" width="100%"/>

