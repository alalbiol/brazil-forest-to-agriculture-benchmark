"""Offline reproduction from released frozen features; no Earth Engine/backbone dependency.

verify: reproduce saved validation/test predictions and the primary result.
train: repeat the original two fits into a separate directory (never published/).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

HERE=Path(__file__).resolve().parent
PUBLISHED=HERE/'published/dataset_c'
OUT=PUBLISHED
DATA=HERE/'cache'
REFERENCE=PUBLISHED/'reference.csv'
CLASSES=['primary_forest','secondary_forest','agriculture','mining']
SEED=20261001
METHODS=['joint_top9','landuse_top9','spectral_joint_top9','change_top9','optical_fraction']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def rows():
    with REFERENCE.open() as f:
        return [r for r in csv.DictReader(f) if r['reference_label']=='accepted']

def dataset(split,key):
    xs,ys=[],[]
    metadata=[]
    for row in rows():
        if row["split"]!=split:
            continue
        path=DATA/"features"/f"{row['window_id']}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            x=f[key].astype(np.float32).reshape(-1,196).T
            y=f["token_labels"].reshape(-1)
        valid=y>=0
        xs.append(x[valid]);ys.append(y[valid]);metadata.append(row)
    if not xs:
        raise ValueError(f"No {split} examples")
    return torch.from_numpy(np.concatenate(xs)),torch.from_numpy(np.concatenate(ys)),metadata


def fit_one(key):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(8)
    train_x,train_y,train_meta=dataset("train",key)
    val_x,val_y,val_meta=dataset("validation",key)
    mean=train_x.mean(0)
    std=train_x.std(0).clamp_min(.05)
    train_x=(train_x-mean)/std
    val_x=(val_x-mean)/std
    count=torch.bincount(train_y,minlength=4)
    if (count==0).any():
        raise ValueError(f"Missing training class: {count.tolist()}")
    weights=count.float().sum().div(count.float()).sqrt()
    weights=weights/weights.mean()
    model=torch.nn.Linear(train_x.shape[1],4)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best=float("inf");history=[];stale=0
    checkpoint=OUT/f"{key}_landuse_head.pt"
    for epoch in range(1,101):
        model.train()
        indices=torch.randperm(len(train_y))
        total=0.
        for batch in indices.split(4096):
            optimizer.zero_grad()
            loss=F.cross_entropy(model(train_x[batch]),train_y[batch],weight=weights)
            loss.backward();optimizer.step()
            total+=loss.detach().item()*len(batch)
        model.eval()
        with torch.no_grad():
            val_losses=[]
            # Fixed training-derived weights also used for validation loss.
            for batch in torch.arange(len(val_y)).split(8192):
                val_losses.append(float(F.cross_entropy(model(val_x[batch]),val_y[batch],weight=weights,reduction="sum")))
            val_loss=sum(val_losses)/float(weights[val_y].sum())
        history.append({"epoch":epoch,"train_loss":total/len(train_y),"validation_loss":val_loss})
        if val_loss<best-1e-5:
            best=val_loss;stale=0
            torch.save({"state_dict":model.state_dict(),"mean":mean,"std":std,
                        "classes":CLASSES,"epoch":epoch,"validation_loss":best,
                        "train_token_counts":count.tolist(),"key":key,
                        "reference_sha256":sha256(REFERENCE),
                        "seed":SEED},checkpoint)
        else:
            stale+=1
        if epoch%10==0:
            print(key,"epoch",epoch,"validation loss",round(val_loss,5),flush=True)
        if epoch>=20 and stale>=12:
            break
    (OUT/f"{key}_training.json").write_text(json.dumps({
        "train_chips":len(train_meta),"validation_chips":len(val_meta),
        "train_tokens":len(train_y),"validation_tokens":len(val_y),
        "class_token_counts":count.tolist(),"class_weights":weights.tolist(),
        "architecture":"one linear 1x1 token classifier; frozen backbone",
        "optimizer":{"name":"AdamW","lr":.001,"weight_decay":.01,"batch_tokens":4096},
        "history":history,"checkpoint_sha256":sha256(checkpoint)},indent=2)+"\n")
    print("saved",checkpoint,"best validation loss",best,flush=True)


def train():
    fit_one("tokens")
    fit_one("spectral")


def load(key):
    state=torch.load(OUT/f"{key}_landuse_head.pt",map_location="cpu",weights_only=False)
    model=torch.nn.Linear(len(state["mean"]),4)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model,state


def top9(a):
    return float(np.sort(a.ravel())[-9:].mean())


def predict(split):
    models={key:load(key) for key in ("tokens","spectral")}
    output=[]
    for row in rows():
        if row["split"]!=split:
            continue
        path=DATA/"features"/f"{row['window_id']}.npz"
        if not path.exists():
            continue
        with np.load(path) as f:
            change=f["change"].reshape(-1)
            probabilities={}
            for key,(model,state) in models.items():
                x=torch.from_numpy(f[key].astype(np.float32).reshape(-1,196).T)
                with torch.no_grad():
                    probabilities[key]=model((x-state["mean"])/state["std"]).softmax(-1).numpy()
            ag=probabilities["tokens"][:,2]
            spectral=probabilities["spectral"][:,2]
            output.append({**row,"joint_top9":top9(ag*change),
                           "landuse_top9":top9(ag),
                           "spectral_joint_top9":top9(spectral*change),
                           "change_top9":top9(change),
                           "optical_fraction":float(f["optical_fraction"]),
                           "mean_agri_probability":float(ag.mean())})
    return output


def threshold(rows,name):
    relevant=[r for r in rows if r["endpoint"]!="stable_forest"]
    y=np.array([r["endpoint"]=="agriculture" for r in relevant])
    s=np.array([r[name] for r in relevant])
    if len(set(y))!=2:
        raise ValueError("Validation requires agriculture and nonagri loss")
    fpr,tpr,t=roc_curve(y,s)
    valid=np.flatnonzero(fpr<=.1)
    best=valid[np.argmax(tpr[valid])]
    return {"threshold":float(t[best]),"validation_recall":float(tpr[best]),
            "validation_fpr":float(fpr[best]),"positive_n":int(y.sum()),
            "negative_n":int((~y).sum()),"comparison":"score >= threshold"}


def wilson(k,n):
    if n==0:return [None,None]
    z=1.96;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [round(c-h,4),round(c+h,4)]


def group(rows,thresholds):
    y=np.array([r["endpoint"]=="agriculture" for r in rows])
    out={"n":len(rows),"positive_n":int(y.sum()),"negative_n":int((~y).sum()),"methods":{}}
    if min(out["positive_n"],out["negative_n"])<5 and len(set(y))==2:
        out["caution"]="Fewer than five examples in one class: descriptive metrics only; bootstrap intervals suppressed."
    for name in METHODS:
        s=np.array([r[name] for r in rows])
        p=s>=thresholds[name]["threshold"]
        tp=int((y&p).sum());fn=int((y&~p).sum());fp=int((~y&p).sum());tn=int((~y&~p).sum())
        recall=tp/(tp+fn) if tp+fn else None
        fpr=fp/(fp+tn) if fp+tn else None
        out["methods"][name]={"auroc":float(roc_auc_score(y,s)) if len(set(y))==2 else None,
                              "tp":tp,"fn":fn,"fp":fp,"tn":tn,
                              "recall":recall,"recall_ci95_wilson":wilson(tp,tp+fn),
                              "fpr":fpr,"fpr_ci95_wilson":wilson(fp,fp+tn),
                              "accuracy":float((p==y).mean()) if len(y) else None,
                              "balanced_accuracy":(recall+1-fpr)/2 if recall is not None and fpr is not None else None}
    return out


def bootstrap(rows,iterations=2000):
    blocks=sorted({r["block_id"] for r in rows})
    lookup={b:[i for i,r in enumerate(rows) if r["block_id"]==b] for b in blocks}
    y=np.array([r["endpoint"]=="agriculture" for r in rows])
    scores={name:np.array([r[name] for r in rows]) for name in METHODS}
    draws={name:[] for name in METHODS}
    diffs=[]
    rng=np.random.default_rng(20261001)
    for _ in range(iterations):
        indices=np.array([i for b in rng.choice(blocks,len(blocks),replace=True) for i in lookup[b]])
        if len(set(y[indices]))!=2:continue
        values={name:roc_auc_score(y[indices],s[indices]) for name,s in scores.items()}
        for name,value in values.items():draws[name].append(value)
        diffs.append(values["joint_top9"]-values["change_top9"])
    result={name:[round(float(x),4) for x in np.percentile(vals,[2.5,97.5])] if vals else [None,None]
            for name,vals in draws.items()}
    difference={"estimate":float(roc_auc_score(y,scores["joint_top9"])-roc_auc_score(y,scores["change_top9"])),
                "ci95":[round(float(x),4) for x in np.percentile(diffs,[2.5,97.5])],
                "bootstrap_units":"10km geographic blocks","valid_draws":len(diffs)}
    return result,difference


def verify(destination):
    torch.set_num_threads(4)
    manifest=json.loads((HERE/'manifest.json').read_text())
    for item in manifest['files']:
        path=HERE/item['path']
        if not path.exists() or sha256(path)!=item['sha256']:
            raise ValueError(f"Missing or changed artifact: {item['path']}")
    thresholds=json.loads((PUBLISHED/'operating_point.json').read_text())['thresholds']
    errors={}
    for split in ['validation','test']:
        fresh=predict(split)
        with (PUBLISHED/f'{split}_predictions.csv').open() as f:
            saved={r['window_id']:r for r in csv.DictReader(f)}
        assert {r['window_id'] for r in fresh}==set(saved)
        errors[split]={name:max(abs(r[name]-float(saved[r['window_id']][name])) for r in fresh) for name in METHODS}
        assert max(errors[split].values())<1e-6,errors
        if split=='validation':
            calculated={name:threshold(fresh,name) for name in METHODS}
            assert all(abs(calculated[name]['threshold']-thresholds[name]['threshold'])<1e-6 for name in METHODS)
        else:
            loss=[r for r in fresh if r['endpoint']!='stable_forest']
            primary=group(loss,thresholds)
            ci,diff=bootstrap(loss)
    published=json.loads((PUBLISHED/'report.json').read_text())['metrics']['loss_endpoints']
    for name in METHODS:
        a=primary['methods'][name];b=published['methods'][name]
        assert all(a[k]==b[k] for k in ['tp','fn','fp','tn'])
        assert abs(a['auroc']-b['auroc'])<1e-6
    assert abs(diff['estimate']-published['joint_minus_change_auroc']['estimate'])<1e-6
    assert diff['ci95']==published['joint_minus_change_auroc']['ci95']
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps({'status':'passed','verified_files':len(manifest['files']),
        'maximum_score_errors':errors,'loss_endpoint_metrics':primary,
        'auroc_ci95_block_bootstrap':ci,'joint_minus_change_auroc':diff},indent=2)+'\n')
    print('PASS: manifest hashes, validation/test scores, frozen thresholds, confusion counts and primary paired bootstrap.')
    print(destination)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['verify','train'])
    parser.add_argument('--output',type=Path,default=HERE/'rerun')
    args=parser.parse_args()
    if args.command=='verify':
        verify(args.output/'verification.json')
    else:
        OUT=args.output.resolve()
        if OUT==PUBLISHED.resolve() or PUBLISHED.resolve() in OUT.parents:
            raise ValueError('Choose an output outside published/ to preserve published results.')
        OUT.mkdir(parents=True,exist_ok=True)
        train()
        print('Fits saved separately; published heads and predictions were not modified.')

if __name__=='__main__':main()
