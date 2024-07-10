Basic Implementation of Text and Code Embeddings by Contrastive Pre-Training
https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf

Implementation Details:
    1. Use pretrained GPT2 (117M) as Encoder.
    2. Instead of training on the entire web, we use MS MARCO.
    3. We train a linear probe to measure the quality of our representations on the SentEval SUBJ dataset.