
# Distill Training - EquiformerV2

- node2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/eqv2-painn-kd/painn_h512_n2ndistill_eqv2.yml
```

- edge2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/eqv2-painn-kd/painn_h512_e2ndistill_eqv2.yml
```

- vec2vec Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/eqv2-painn-kd/painn_h512_v2vdistill_eqv2.yml
```

- node2node & edge2node & vec2vec Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/eqv2-painn-kd/painn_h512_trinity_distill_eqv2.yml
```


# Distill Training - Gemnet-oc

- Baseline: training from scratch

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_goc/painn_h512.yml logger=wandb
```


- node2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_goc/painn_h512_n2ndistill.yml logger=wandb
```


- edge2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_goc/painn_h512_e2ndistill.yml logger=wandb
```

- adversarial_attack Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_goc/painn_h512_adversarial_jitter.yml logger=wandb
```