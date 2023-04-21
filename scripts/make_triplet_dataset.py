import sys
sys.path.insert(0,'..')

import ruamel.yaml as yaml
import pickle
from itertools import product
import random
import argparse


random.seed(0)


def sample_equal(base_l, sample_l):
    q, mod = divmod(len(base_l), len(sample_l))
    n_samples = q * [len(sample_l)] + [mod]
    samples = []
    for n in n_samples:
        samples.extend(random.sample(sample_l, n))
    return samples


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="dataset")
    parser.add_argument("--pk", type=str, help="protected key to create dataset for")
    parser.add_argument("--n_repeats", type=int, default=1, help="how often to repeat dataset to achieve multiple random sampling operations")
    args = parser.parse_args()

    with open("../cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    data_cfg = f"data_config_{args.ds}"
    args = argparse.Namespace(**cfg[data_cfg], **vars(args))

    with open(args.train_pkl, 'rb') as file:
        data_dicts = pickle.load(file)

    protected_key = args.protected_key
    if isinstance(protected_key, str):
        protected_key = [protected_key]

    keys = [args.task_key, *protected_key, args.text_key]
    x = [[d[k] for k in keys] for d in data_dicts]

    pk = protected_key if args.pk is None else [args.pk]
    pk_id = [i for i,p in enumerate(keys) if p in pk]

    data = dict(zip(keys, zip(*x)))

    protected_key_combs = dict(enumerate(product(set(data[args.task_key]), *[set(data[k]) for k in pk])))
    comb_data_dict = {i:[t for t in x if t[0]==v[0] and all([t[id]==v[j+1] for j, id in enumerate(pk_id)])] for i, v in protected_key_combs.items()}

    del data
    
    triplet_ds = []
    for _ in range(args.n_repeats):

        triplet_subds = []
        for i, data in comb_data_dict.items():
            data_dict = dict(zip(keys, zip(*data)))
            v = protected_key_combs[i]
            tv, pvs = v[0], v[1:]

            other_pv = []
            for k, v in protected_key_combs.items():
                neq = [a!=b for a,b in zip(v[1:],pvs)]
                if v[0]==tv and any(neq):
                    other_pv.append((k, sum(neq)))
            
            other_pv_texts = [(t[-1], weight) for id, weight in other_pv for t in comb_data_dict[id]]
            samples_other_pv = sample_equal(data_dict[args.text_key], other_pv_texts)

            other_tv_texts = [t[-1] for t in x if t[0]!=tv]
            samples_other_tv = sample_equal(data_dict[args.text_key], other_tv_texts)

            triplet_subds.append((data_dict[args.text_key], samples_other_pv, samples_other_tv, data_dict[args.task_key], *[data_dict[pk] for pk in protected_key]))

        triplet_subds = [[v for sub_l in l for v in sub_l] for l in zip(*triplet_subds)]

        idx_list = list(range(len(triplet_subds[0])))
        random.shuffle(idx_list)
        triplet_subds = [[sub_l[i] for i in idx_list] for sub_l in triplet_subds]

        triplet_ds.extend(triplet_subds)

    new_keys = [args.text_key, "input_other_pv", "input_other_tv", args.task_key, *protected_key]
    triplet_ds = [dict(zip(new_keys, x)) for x in zip(*triplet_ds)]
        

    with open(f"../train_triplet_{args.ds}_{'_'.join(pk)}_nrepeat{args.n_repeats}.pkl","wb") as f:
        pickle.dump(triplet_subds, f)


if __name__ == "__main__":

    main()

