- Baseline: training from scratch

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/painn_h512.yml
```

# Distill Training - EquiformerV2-PaiNN

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

- node2node & edge2node & vec2vec (trinity) Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/eqv2-painn-kd/painn_h512_trinity_distill_eqv2.yml
```

# Distill Training - EquiformerV2-EquiformerV2_small

- Baseline: training EquiformerV2_small from scratch

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@1_M@2.yml
```

- node2node Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/n2n_distill.yml
```

- edge2node Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/e2n_distill.yml
```

- vec2vec Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/v2v_distill.yml

- node2node & edge2node & vec2vec (trinity) Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/trinity_distill.yml
```

# Distill Training - Gemnet-oc-PaiNN

### use --task.strict=False to ignore the scaling factors of Gemnet-OC which are uncessary for the distillation.


- node2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/goc-painn-kd/painn_h512_n2ndistill.yml --task.strict=False
```


- edge2node Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/goc-painn-kd/painn_h512_e2ndistill.yml --task.strict=False
```



- vec2vec Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/goc-painn-kd/painn_h512_v2vdistill.yml --task.strict=False
```


- node2node & edge2node & vec2vec (trinity) Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/eqv2-painn-kd/painn_h512_trinity_distill.yml --task.strict=False
```


- adversarial_attack Distill

```
python main.py --mode train --config-yml configs/s2ef/200k/painn/goc-painn-kd/painn_h512_adversarial_jitter.yml --task.strict=False
```