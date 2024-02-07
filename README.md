# Towards Accurate Dst Index Prediction: A Deep Learning Framework Using RNNs

{\\
  @misc{\\
      title={Towards Accurate Dst Index Prediction: A Deep Learning Framework Using RNNs}, \\
      author={Jorge Enciso},\\
      year={2024},\\
      eprint={...},\\
      archivePrefix={...},\\
      primaryClass={...}\\
}\\
}

## Documentation

### Models


### Preprocessing
Automated data collection methods for:

- SWARM 
- Deep Space Climate Observatory (DSCOVR)
- WIND
- Solar and Heliospheric Observatory (SOHO)
- Advanced Composition Explorer (ACE)
- World Data Center for Geomagnetism (WDCG) Dst index

### Datasets
Training oriented dataset creation for:

- Seq2Seq: DSCOVR -> SWARM (First Stage)
- Seq2Seq: WIND, SOHO, ACE -> DSCOVR (Synthetic Data generation)
- Seq2Multiclass: SWARM -> Dst (Second Stage)

### Models
Seq2Seq RNN model:

### First Stage
Seq2Seq: DSCOVR -> SWARM