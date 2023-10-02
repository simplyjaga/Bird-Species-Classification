# Multi-label Classification of Bird Species

As part of ornithology, bird species classification is vital to understand species distribution,
habitat requirements and environmental changes that affecting bird populations. It is possible for
ornithologists to assess the health of a certain habitat by tracking changes in bird species distributions. In
this paper, a method for labelling and classifying multiple bird species in real-time recordings using transfer
learning technique and transformers in deep learning. For this purpose, Wav2vec is proposed as
classification method for labelling the bird calls using only their raw audio waveform. It is expected that the
proposed algorithm classifies multi-species of bird calls that present in input audio file based to distinguish
its characteristics. A clipping technique with an aggregation strategy on the overlapping audio recordings
was used to determine multiple labels for the provided test data. Using transfer learning approaches, the
features of audio recordings have been automatically extracted for classification. Then, each probability
outcomes of audio segment through clipping approach are aggregated to represent multiple species of bird
call. Finally, the probability scores were used to find the presence of predominant bird species in the audio
recording for multi-labelling. The proposed Wav2vec approach achieves F1-score of 0.87 using Xeno-
Canto dataset in which the performance has been improved over other multi-label classifiers.
