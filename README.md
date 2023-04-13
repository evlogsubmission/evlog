## Implementation package

This is an implementation of ICSE 2024 submission #698: EvLog: Identifying Anomalous Logs over Software Evolution

The content includes:
- `datasets/` contains the *dataloader* and multi-level representation extractor for rich representation and abstract representation.
- `networks/` contains the *anomalous log discriminator*. Theoretically, this module be replaced by any neural network architecture, demonstrating the extensibility of our approach. 
- `optim/` contains the training and evaluating process.
- We also release samples of our *collected dataset* in `sampled_data/sample.pkl`. The full dataset will be released upon acceptance.
