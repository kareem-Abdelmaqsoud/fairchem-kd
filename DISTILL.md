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
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/equiformer_v2_N@8_L@1_M@1.yml --optim.batch_size=1 --optim.eval_batch_size=1
```

- node2node Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/n2n_distill.yml --optim.batch_size=1 --optim.eval_batch_size=1 
```

- edge2edge Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/e2e_distill.yml --optim.batch_size=1 --optim.eval_batch_size=1 
```

- node2node & edge2edge Distill

python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/n2n_e2e_distill.yml --optim.batch_size=1 --optim.eval_batch_size=1 

- vec2vec Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/equiformer_v2/v2v_distill.yml

- node2node & edge2edge & vec2vec (trinity) Distill

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



# Distill Training - Gemnet-oc-Gemnet-oc



- node2node Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/gemnet/gemnet-oc_small_distill_n2n.yml --optim.batch_size=4 --optim.eval_batch_size=4 --task.strict=False
```


- edge2edge Distill

```
python main.py --mode train --config-yml configs/s2ef/2M/gemnet/gemnet-oc_small_distill_e2e.yml
```