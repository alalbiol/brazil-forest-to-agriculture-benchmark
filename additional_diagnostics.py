"""Post hoc diagnostics of frozen predictions; never train or replace primary thresholds.

Workspace: python additional_diagnostics.py
Released archives: python additional_diagnostics.py --bundle . --output rerun/diagnostics
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

HERE = Path(__file__).resolve().parent
METHODS = ['joint_top9', 'landuse_top9', 'spectral_joint_top9', 'change_top9', 'optical_fraction']
NAMES = ['TerraFM joint', 'TerraFM endpoint only', 'Spectral joint', 'Existing change head', 'Optical index change']
COLORS = ['#0072B2', '#009E73', '#D55E00', '#777777', '#CC79A7']


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(y, score, threshold):
    p = score >= threshold
    tp, fn = int((p & y).sum()), int((~p & y).sum())
    fp, tn = int((p & ~y).sum()), int((~p & ~y).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, recall=tp/(tp+fn),
                fpr=fp/(fp+tn), accuracy=(tp+tn)/len(y))


def save(fig, path, raster=False):
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    if raster:
        fig.savefig(path.with_suffix('.tif'), dpi=300, bbox_inches='tight', pil_kwargs={'compression':'tiff_lzw'})
    else:
        fig.savefig(path.with_suffix('.eps'), bbox_inches='tight')
    plt.close(fig)


def comparison_examples(test, thresholds, published, data, gen, figdir):
    models = {}
    for key in ['tokens', 'spectral']:
        state = torch.load(published/f'{key}_landuse_head.pt', map_location='cpu', weights_only=False)
        model = torch.nn.Linear(len(state['mean']), 4)
        model.load_state_dict(state['state_dict']); model.eval()
        models[key] = model, state
    chosen = []
    for endpoint, terra_correct in [('agriculture', True), ('secondary_forest', True), ('mining', False)]:
        def match(r):
            y = r['endpoint'] == 'agriculture'
            a = (float(r['joint_top9']) >= thresholds['joint_top9']['threshold']) == y
            b = (float(r['spectral_joint_top9']) >= thresholds['spectral_joint_top9']['threshold']) == y
            return r['endpoint'] == endpoint and a == terra_correct and a != b
        candidates = sorted(filter(match, test), key=lambda r:(float(r['joint_top9']), r['window_id']))
        if candidates:
            chosen.append(candidates[(len(candidates)-1)//2])
    assert len(chosen) == 3
    fig, axes = plt.subplots(3, 5, figsize=(10, 6.5), layout='constrained')
    cmap = ListedColormap(['#eeeeee', '#16652f', '#7eaf6b', '#e8b33b', '#88419d'])
    for i, row in enumerate(chosen):
        wid = row['window_id']
        for j, year in enumerate([2020, 2024]):
            with np.load(data['imagery']/f'{wid}_{year}.npz') as f:
                rgb = f['imagery'][[5,4,3]].transpose(1,2,0)
            axes[i,j].imshow(np.clip(rgb/3500,0,1)**.8, interpolation='nearest')
        with np.load(data['labels']/f'{wid}.npz') as f:
            after = f['after']; mapped = np.zeros_like(after)
            for codes, target in [([1],1),([2],2),([3,4],3),([10],4)]:
                mapped[np.isin(after,codes)] = target
        axes[i,2].imshow(mapped,cmap=cmap,vmin=0,vmax=4,interpolation='nearest')
        with np.load(data['features']/f'{wid}.npz') as f:
            for j, (key, method) in enumerate([('tokens','joint_top9'),('spectral','spectral_joint_top9')],3):
                model, state = models[key]
                x = torch.from_numpy(f[key].astype('float32').reshape(-1,196).T)
                with torch.no_grad():
                    ag = model((x-state['mean'])/state['std']).softmax(-1)[:,2].numpy().reshape(14,14)
                joint = ag * f['change']
                score = float(np.sort(joint.ravel())[-9:].mean())
                assert abs(score-float(row[method])) < 1e-6
                last = axes[i,j].imshow(joint,cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
                positive = score >= thresholds[method]['threshold']
                truth = row['endpoint'] == 'agriculture'
                category = ('TP' if positive else 'FN') if truth else ('FP' if positive else 'TN')
                axes[i,j].text(.5,-.10,f'{category}: {score:.3f}',transform=axes[i,j].transAxes,ha='center',fontsize=9)
        axes[i,0].set_ylabel(row['endpoint'].replace('_',' ').capitalize(),fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
    for ax, title in zip(axes[0],['2020 RGB','2024 RGB','Mapped 2024 class','TerraFM joint','Spectral joint']):
        ax.set_title(title,fontsize=10)
    fig.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(cmap.colors,
        ['Other/unknown','Primary forest','Secondary forest','Agriculture','Mining'])],
        loc='outside lower center',ncol=5,fontsize=8)
    fig.colorbar(last,ax=axes[:,3:].ravel().tolist(),shrink=.5,label='Uncalibrated token score')
    save(fig,figdir/'figureS2_comparison',raster=True)
    selection = {'criterion':'Lower median TerraFM joint score, ordered by score then ID, within each endpoint/direction disagreement category. Both methods winning are displayed.',
                 'thresholds':{k:thresholds[k]['threshold'] for k in ['joint_top9','spectral_joint_top9']},'examples':chosen}
    (gen/'diagnostic_examples.json').write_text(json.dumps(selection,indent=2)+'\n')


def supplementary_section(ap, sensitivity, disagreements):
    text = r'''\section{Post hoc precision--recall, threshold sensitivity and disagreements}
These descriptive analyses were added on 4 October 2026 after inspection of the original test results. They use the same fitted models and 113 loss-endpoint test chips (84 agricultural, 19 mining and ten secondary-forest endpoints). The original thresholds and primary results are retained.

\subsection{Precision--recall summaries}
Average precision (AP) is the noninterpolated sum $\sum_n(R_n-R_{n-1})P_n$ over distinct score thresholds, calculated with \texttt{sklearn.metrics.average\_precision\_score}. Agriculture-positive curves use the original scores; nonagriculture-positive curves reverse them as $1-S$. Both describe rankings of uncalibrated scores. The respective prevalences, 84/113 (74.3\%) and 29/113 (25.7\%), define the no-discrimination precision reference lines; finite-sample random-ranking AP can differ from these values. Table~\ref{tab:ap} reports descriptive point estimates without additional uncertainty intervals or significance tests. Their interpretation is conditional on the purposively sampled class mixture.
\begin{table}[ht]\centering\small
\caption{Post hoc average precision on the original loss-test predictions. Each column treats the named class as positive.}
\begin{tabular}{lrr}\toprule
Method & Agriculture AP & Nonagriculture AP\\\midrule
'''
    for r in ap:
        text += f"{r['method_name']} & {r['agriculture_ap']:.3f} & {r['nonagriculture_ap']:.3f}" + r' \\' + '\n'
    text += r'''\bottomrule\end{tabular}\label{tab:ap}\end{table}
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{figures/figureS1_pr.pdf}
\caption{Post hoc precision--recall curves, with agriculture (left) and nonagricultural loss (right) treated as positive. Dashed lines mark the respective test prevalences. Legends report noninterpolated AP. The right panel uses reversed scores.}\label{fig:pr}
\end{figure}
\clearpage
\subsection{Validation-derived threshold sensitivity}
For each method, thresholds maximize agricultural validation recall subject to FPR caps of 5\%, 10\% and 20\%, using the original ROC ordering and tie rule. Threshold selection uses 72 agricultural chips and 32 nonagricultural loss controls from validation; stable controls are excluded. Although the additional caps were chosen after the test was examined, their thresholds use validation scores exclusively. The 10\% rows reproduce the original thresholds and confusion counts. With 32 validation controls, FPR steps are 3.125 percentage points, allowing at most one, three and six false positives under the three caps. ROC ties can give identical thresholds across caps. Table~\ref{tab:sensitivity} reports percentage rates on the fixed test sample; validation caps do not guarantee the same test FPR, and threshold-selection uncertainty is not propagated.
\begin{table}[ht]\centering\small
\caption{Post hoc sensitivity to the validation FPR cap. Exact thresholds and confusion counts are supplied in \texttt{threshold\_sensitivity.csv}. The 10\% rows are the unchanged original analysis.}
\begin{tabular}{lrrrrr}\toprule
Method & Cap & Val. FPR & Test recall & Test FPR & Test accuracy\\\midrule
'''
    for r in sensitivity:
        text += f"{r['method_name']} & {100*r['validation_cap']:.0f} & {100*r['validation_fpr']:.3f} & {100*r['recall']:.1f} & {100*r['fpr']:.1f} & {100*r['accuracy']:.1f}" + r' \\' + '\n'
    text += r'''\bottomrule\end{tabular}\label{tab:sensitivity}\end{table}
\subsection{Paired correctness disagreements}
Table~\ref{tab:disagreements} compares correctness under the two methods' original validation thresholds. TerraFM alone correctly classifies eight agricultural chips and one secondary-forest chip; the spectral baseline alone correctly classifies one mining chip. The counts are conditional on these thresholds and the mapped references.
\begin{table}[H]\centering\small
\caption{Post hoc paired correctness counts at the original thresholds.}
\begin{tabular}{lrrrr}\toprule
Reference endpoint & Both correct & TerraFM only & Spectral only & Both wrong\\\midrule
'''
    for r in disagreements:
        text += r['endpoint'].replace('_',' ').capitalize()+' & '+' & '.join(str(r[k]) for k in ['both_correct','terra_only','spectral_only','both_wrong'])+r' \\'+'\n'
    text += r'''\bottomrule\end{tabular}\label{tab:disagreements}\end{table}
Figure~\ref{fig:comparison} shows both directions of disagreement. Within each endpoint/direction category, selection takes the lower median TerraFM joint score, ordered by score then identifier. The composites, reference grids and cached features for all three test chips are supplied, with RGB scaling identical to the main figure. Top-nine scores recomputed from the displayed maps agree with saved scores to within $10^{-6}$. The examples describe model differences; identifying their cause would require controlled ablations.
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{figures/figureS2_comparison.pdf}
\caption{Post hoc comparison examples showing TerraFM-only correct agriculture and secondary-forest cases, and a spectral-only correct mining case. TP/FN/FP/TN refer to agriculture as positive and to the mapped reference. TerraFM and spectral chip thresholds are 0.25337934494018555 and 0.7250828742980957; token maps themselves are not thresholded.}\label{fig:comparison}
\end{figure}
\clearpage
'''
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.bundle:
        published = args.bundle/'published/dataset_c'
        data = {k:args.bundle/'cache'/('example_imagery' if k=='imagery' else k) for k in ['features','labels','imagery']}
    else:
        bench = HERE.parents[1]/'use_case_2/brazil_benchmark'
        published = bench/'results/last_effort'
        data = {k:bench/'data/processed/last_effort'/k for k in ['features','labels','imagery']}
    output = args.output or (args.bundle/'rerun/diagnostics' if args.bundle else HERE)
    gen, figdir = output/'generated', output/'figures'
    gen.mkdir(parents=True,exist_ok=True); figdir.mkdir(parents=True,exist_ok=True)
    guarded = ['validation_predictions.csv','test_predictions.csv','operating_point.json','report.json',
               'tokens_landuse_head.pt','spectral_landuse_head.pt']
    original = {n:sha(published/n) for n in guarded}
    thresholds = json.loads((published/'operating_point.json').read_text())['thresholds']
    report = json.loads((published/'report.json').read_text())
    test = [r for r in read(published/'test_predictions.csv') if r['endpoint']!='stable_forest']
    val = [r for r in read(published/'validation_predictions.csv') if r['endpoint']!='stable_forest']
    y = np.array([r['endpoint']=='agriculture' for r in test])
    vy = np.array([r['endpoint']=='agriculture' for r in val])
    assert (len(y),int(y.sum()),len(vy),int(vy.sum())) == (113,84,104,72)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    torch.set_num_threads(4)
    fig, axes = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    ap, sensitivity = [], []
    for method, name, color in zip(METHODS,NAMES,COLORS):
        score = np.array([float(r[method]) for r in test])
        aps = [float(average_precision_score(y,score)),float(average_precision_score(~y,1-score))]
        ap.append(dict(method=method,method_name=name,agriculture_ap=aps[0],nonagriculture_ap=aps[1]))
        for ax, truth, s, value in zip(axes,[y,~y],[score,1-score],aps):
            precision, recall, _ = precision_recall_curve(truth,s)
            ax.step(recall,precision,where='post',color=color,lw=1.4,label=f'{name}: {value:.3f}')
        vs = np.array([float(r[method]) for r in val])
        fpr, tpr, ts = roc_curve(vy,vs)
        for cap in [.05,.10,.20]:
            valid = np.flatnonzero(fpr<=cap)
            best = valid[np.argmax(tpr[valid])]; threshold = float(ts[best])
            c = counts(y,score,threshold)
            sensitivity.append(dict(method=method,method_name=name,validation_cap=cap,threshold=threshold,
                validation_recall=float(tpr[best]),validation_fpr=float(fpr[best]),**c))
            if cap==.10:
                assert threshold == thresholds[method]['threshold']
                original_metrics = report['metrics']['loss_endpoints']['methods'][method]
                assert all(c[k]==original_metrics[k] for k in ['tp','fn','fp','tn'])
    for ax, truth, title in zip(axes,[y,~y],['Agriculture positive','Nonagricultural loss positive']):
        ax.axhline(truth.mean(),color='#444444',ls='--',lw=1,label=f'Prevalence: {truth.mean():.3f}')
        ax.set(xlabel='Recall',ylabel='Precision',title=title,xlim=(0,1),ylim=(0,1.02))
        ax.legend(fontsize=7.3,loc='lower left',framealpha=1); ax.grid(alpha=.2)
    save(fig,figdir/'figureS1_pr')
    disagreements = []
    for endpoint in ['agriculture','secondary_forest','mining','total']:
        rows = [r for r in test if endpoint=='total' or r['endpoint']==endpoint]
        d = dict(endpoint=endpoint,both_correct=0,terra_only=0,spectral_only=0,both_wrong=0)
        for r in rows:
            truth = r['endpoint']=='agriculture'
            a = (float(r['joint_top9'])>=thresholds['joint_top9']['threshold'])==truth
            b = (float(r['spectral_joint_top9'])>=thresholds['spectral_joint_top9']['threshold'])==truth
            d['both_correct' if a and b else 'terra_only' if a else 'spectral_only' if b else 'both_wrong'] += 1
        disagreements.append(d)
    comparison_examples(test,thresholds,published,data,gen,figdir)
    for name, rows in [('average_precision',ap),('threshold_sensitivity',sensitivity),('paired_disagreements',disagreements)]:
        write_csv(gen/f'{name}.csv',rows)
    (gen/'diagnostics_section.tex').write_text(supplementary_section(ap,sensitivity,disagreements))
    result = dict(added='2026-10-04',status='passed',analysis='Post hoc descriptive diagnostics; original test already examined',
        original_artifact_sha256=original,primary_result_unchanged=True,threshold_source='Validation only; original ROC/tie rule',
        test_n=len(y),agriculture_n=int(y.sum()),agriculture_prevalence=float(y.mean()),
        ap=ap,sensitivity=sensitivity,disagreements=disagreements)
    assert original == {n:sha(published/n) for n in guarded}
    (gen/'additional_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','primary_unchanged':True,'ap':ap,'disagreements':disagreements[-1]},indent=2))


if __name__=='__main__':
    main()
