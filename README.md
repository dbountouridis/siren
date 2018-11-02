# Siren: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments


A simulation framework for the visualization and analysis of the effects of different recommenders systems in an online news environment. This simulation draws mainly on the e-commerce simulation work of Fleder and Hosanagar. However, SIREN also accounts for the specificities of news consumption, such as evolving users preferences and editorial priming (promoted articles on a news website). 

SIREN's interface currently offers recommendations based the [MyMediaLite](www.mymedialite.net/) toolbox and visualizations for two diversity metrics (long-tail and unexpectedness). SIREN can be used by content providers (news outlets) to investigate which recommendation strategy fits better their diversity needs. At the same time, SIREN's code can be adapted/expanded by researchers to analyse various recommender effects in a news environment.

This documentation provides information for:
1. [An overview of SIREN's model](docs/Overview.md)
2. [Setting up and running SIREN](docs/Setup.md)
3. [Using SIREN as researcher](docs/Setup.md)
4. [Using SIREN as content-provider](docs/UsageContentProvider.md)


## Citation

If you use this code, please cite the following publication:

```
@inproceedings{Bountouridis2018SIREN,
    author    = { Dimitrios Bountouridis and Jaron Harambam and Mykola Makhortykh and Monica Marrero and Nava Tintarev and Claudia Hauff},
    title     = { SIREN: A Simulation Framework for Understanding the Effects of Recommender Systems in Online News Environments },
    booktitle = { Proceedings of the ACM Conference on Fairness, Accountability, and Transparency  (FAT*) },
    year      = 2018
}
```

