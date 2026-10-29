"""Source-aware labels and fixed, report-grouped partitions; runs on Kaggle CPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .contracts import TARGETS, UID, sha256, write_json
from .prepare import competition_root

SOURCES = {
    'steven': ('stevenleehans/rsna-knee-llm-report-labels','llm_labels_full.csv','CC0-1.0'),
    'pilkwang': ('pilkwang/rsna-knee-llm-labels','report_labels_v2.csv','CC0-1.0'),
    'lixin': ('lixin73/rsna-knee-llm-report-labels-sol56','report_labels_gpt56sol.csv','CC0-1.0'),
    'dread': ('dreaddevelopment/rsna-knee-labels','labels_llm_soft.csv','CC0-1.0'),
    'jev': ('riadmohamed42/jev-knee-labels','HYBRID_labels_scores.csv','CC0-1.0'),
    'omer98': ('omerzalman/rsna-knee-98-accuracy-11-soft-labels','soft_labels.csv','CC-BY-NC-4.0'),
    'omer99': ('omerzalman/rsna-knee-99-accuracy-11-soft-labels','soft_labels.csv','CC-BY-NC-4.0'),
}


def source_file(slug, filename):
    for path in (Path('/kaggle/input/datasets')/slug/filename,
                 Path('/kaggle/input')/slug.split('/')[-1]/filename):
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing required public source {slug}/{filename}')


def normalized_report(text):
    text = unicodedata.normalize('NFKC',str(text)).casefold()
    return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',text)).strip()


def grouped_partitions(train):
    texts = [normalized_report(x) if pd.notna(x) else '' for x in train.Report]
    parents = list(range(len(texts)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i,j):
        a,b = find(i),find(j)
        parents[max(a,b)] = min(a,b)
    exact = {}
    for i,text in enumerate(texts):
        if not text:
            continue  # Empty reports are not evidence that the patients are duplicates.
        if text in exact:
            union(i,exact[text])
        else:
            exact[text] = i
    long_indices = [i for i,text in enumerate(texts) if len(text)>=80]
    if long_indices:
        features = TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=40000,
                                    min_df=2,dtype=np.float32).fit_transform([texts[i] for i in long_indices])
        neighbors = NearestNeighbors(metric='cosine',algorithm='brute',n_jobs=2).fit(features)
        for start in range(0,len(long_indices),128):
            distances,indices = neighbors.radius_neighbors(features[start:start+128],radius=.03)
            for offset,(ds,js) in enumerate(zip(distances,indices)):
                i = long_indices[start+offset]
                for distance,j in zip(ds,js):
                    union(i,long_indices[j])
    groups = np.array([find(i) for i in range(len(texts))])
    gold = train[TARGETS].notna().any(axis=1).to_numpy()
    excluded_groups = set(groups[gold])
    partitions = []
    folds = []
    group_keys = {}
    for g in np.unique(groups):
        members = sorted(train.loc[groups==g,UID].tolist())
        key = hashlib.sha256(('independent-split-20261022|'+'|'.join(members)).encode()).hexdigest()
        group_keys[g] = key
    for g,is_gold in zip(groups,gold):
        number = int(group_keys[g][:12],16)
        partitions.append('gold' if is_gold else 'excluded_gold_duplicate' if g in excluded_groups
                          else 'comparison' if number%8==0 else 'monitor' if number%8==1 else 'train')
        folds.append(number%3)
    return groups,np.array(partitions),np.array(folds),gold


def parse_source(table, index, source):
    if table[UID].duplicated().any():
        raise ValueError(f'Duplicate IDs in public label source {source}')
    table = table.set_index(UID).reindex(index)
    probabilities = np.full((len(index),12),np.nan,np.float32)
    status = np.full((len(index),12),'missing',dtype='<U12')
    confidence = np.zeros((len(index),12),np.float32)
    for j,target in enumerate(TARGETS):
        if target not in table:
            continue
        p = pd.to_numeric(table[target],errors='coerce').to_numpy(float)
        if np.any(np.isfinite(p)&((p<0)|(p>1))):
            raise ValueError(f'Out-of-range source probabilities: {source}/{target}')
        probabilities[:,j] = p
        known = np.isfinite(p)
        status[known,j] = 'uncertain'
        status[known&(p>=.65),j] = 'positive'
        status[known&(p<=.35),j] = 'negative'
        if source == 'steven':
            status[known&np.isclose(p,.5),j] = 'unmentioned'  # Explicitly documented source convention.
        verdict = target+'__verdict'
        if verdict in table:
            v = table[verdict].astype(str).str.upper().to_numpy()
            status[np.isin(v,['UNK','UNKNOWN','UNMENTIONED']),j] = 'unmentioned'
            status[np.isin(v,['YES','POSITIVE']),j] = 'positive'
            status[np.isin(v,['NO','NEGATIVE']),j] = 'negative'
        c = np.abs(2*p-1)
        if target+'__conf' in table:
            supplied = pd.to_numeric(table[target+'__conf'],errors='coerce').to_numpy(float)
            # Source confidence is retained only if it actually has a probability scale.
            supplied = np.where((supplied>=0)&(supplied<=1),supplied,np.nan)
            c = np.where(np.isfinite(supplied),supplied,c)
        confidence[:,j] = np.where(np.isfinite(c),np.clip(c,0,1),0)
        confidence[np.isin(status[:,j],['missing','unmentioned','uncertain']),j] = 0
    return probabilities,status,confidence


def fuse(probabilities,status,confidence,train_mask):
    # Main fusion excludes dependent JEV hard swaps and Gold-optimized, image-teacher-derived Omer labels.
    # They remain fully represented in the source-aware audit, not claimed as independent voters.
    p = probabilities[:4]
    valid = np.isin(status[:4],['positive','negative']) & np.isfinite(p)
    safe = np.nan_to_num(p,nan=.5)
    count = valid.sum(0)
    mean = (safe*valid).sum(0)/count.clip(1)
    mean[count==0] = .5
    signs = ((safe>.5)&valid).sum(0)/count.clip(1)
    agreement = np.maximum(signs,1-signs)
    agreement[count==0] = 0
    variance = (((safe-mean)**2)*valid).sum(0)/count.clip(1)
    known = (count>0)&((mean<=.35)|(mean>=.65))
    # Dread/Lixin numeric scores do not independently expose a mention verdict.
    # Do not reinterpret their low prior probabilities as explicit negatives when
    # BOTH mention-aware primary readers decline to address a finding.
    mention_supported = valid[0] | valid[1]
    known &= mention_supported
    l1 = known.astype(np.float32)
    strong_count = (valid&(confidence[:4]>=.7)).sum(0)
    l2 = (known&(strong_count>=2)&(agreement>=.9)&(np.abs(mean-.5)>=.3)).astype(np.float32)
    source_confidence = (confidence[:4]*valid).sum(0)/count.clip(1)
    l3 = l1*np.clip(2*np.abs(mean-.5),0,1)*agreement*np.clip(1-4*variance,0,1)*source_confidence
    # L4 source reliability is estimated ONLY within the fitting partition from leave-one-source-out agreement.
    reliability = np.zeros((4,12),np.float32)
    for s in range(4):
        other_count = count-valid[s]
        other_mean = ((safe*valid).sum(0)-safe[s]*valid[s])/other_count.clip(1)
        for j in range(12):
            rows = train_mask & valid[s,:,j] & (other_count[:,j]>0)
            reliability[s,j] = max(.05,float(1-np.abs(safe[s,rows,j]-other_mean[rows,j]).mean())) if rows.any() else .05
    weight = valid*reliability[:,None,:]
    l4_p = (safe*weight).sum(0)/weight.sum(0).clip(1e-6)
    l4_p[weight.sum(0)==0] = .5
    l4_w = l3*((l4_p<=.35)|(l4_p>=.65))
    return np.stack([mean,mean,mean,l4_p]),np.stack([l1,l2,l3,l4_w]),agreement,reliability


def main():
    root = competition_root()
    train = pd.read_csv(root/'train.csv',dtype={UID:str}).sort_values(UID).reset_index(drop=True)
    groups,partition,folds,gold = grouped_partitions(train)
    ps,ss,cs,provenance = [],[],[],[]
    for source,(slug,filename,license_name) in SOURCES.items():
        path = source_file(slug,filename)
        table = pd.read_csv(path,dtype={UID:str})
        p,status,confidence = parse_source(table,train[UID],source)
        if source.startswith('omer'):
            accepted = pd.read_csv(source_file(slug,'accepted_mask.csv'),dtype={UID:str}).set_index(UID).reindex(train[UID])
            supplied_conf = pd.read_csv(source_file(slug,'confidence.csv'),dtype={UID:str}).set_index(UID).reindex(train[UID])
            for j,target in enumerate(TARGETS):
                if target in accepted:
                    take = pd.to_numeric(accepted[target],errors='coerce').fillna(0).to_numpy()>0
                    status[(~take)&np.isfinite(p[:,j]),j] = 'uncertain'
                    confidence[:,j] = pd.to_numeric(supplied_conf[target],errors='coerce').fillna(0).to_numpy()
                    confidence[~take,j] = 0
        ps.append(p); ss.append(status); cs.append(confidence)
        provenance.append({'source':source,'url':'https://www.kaggle.com/datasets/'+slug,
                           'file':filename,'sha256':sha256(path),'license':license_name,
                           'primary_independent_voter':source in ['steven','pilkwang','lixin','dread'],
                           'gold_selection_exposure':source in ['jev','omer98','omer99'],
                           'notes': 'Gold-selected accepted accuracy is NOT MRI classification accuracy; no Synovitis; teacher exposure disclosed' if source.startswith('omer')
                                    else 'Hard-swap contains Steven/Pilkwang and was selected on Gold; not another independent vote' if source=='jev'
                                    else 'Model-generated report targets; not clinical gold'})
    probabilities,status,confidence = np.stack(ps),np.stack(ss),np.stack(cs)
    targets,weights,agreement,reliability = fuse(probabilities,status,confidence,partition=='train')
    group_sizes = pd.Series(groups).value_counts()
    duplicate_weight = np.array([1/group_sizes[g] for g in groups],np.float32)
    weights *= duplicate_weight[None,:,None]
    weights[:,np.isin(partition,['gold','excluded_gold_duplicate'])] = 0
    out = Path('/kaggle/working')
    np.savez_compressed(out/'supervision.npz',uids=train[UID].to_numpy(str),targets=targets,weights=weights,
                        groups=groups,partitions=partition,folds=folds,gold=gold,
                        gold_labels=train[TARGETS].to_numpy(np.float32),source_p=probabilities,
                        source_status=status,source_confidence=confidence,agreement=agreement)
    # Source-aware long form, WITHOUT reports.
    long = pd.DataFrame({UID:np.repeat(train[UID].to_numpy(),12),'target':np.tile(TARGETS,len(train)),
                         'partition':np.repeat(partition,12),'group':np.repeat(groups,12),
                         'agreement':agreement.reshape(-1),'confidence_weight':weights[2].reshape(-1)})
    for s,source in enumerate(SOURCES):
        long[source+'_p'] = probabilities[s].reshape(-1)
        long[source+'_status'] = status[s].reshape(-1)
        long[source+'_available'] = np.isfinite(probabilities[s]).reshape(-1)
        long[source+'_status_origin'] = ('raw YES/NO/UNK verdict' if source=='pilkwang'
                                        else 'polarity of score; documented 0.5=unmentioned' if source=='steven'
                                        else 'score polarity with released acceptance mask' if source.startswith('omer')
                                        else 'predicted score polarity, not an explicit mention verdict')
    long.to_csv(out/'source_aware_supervision.csv.gz',index=False)
    coverage = {}
    for policy in range(4):
        coverage[f'L{policy+1}'] = {target:{'known':int((weights[policy,:,j]>0).sum()),
            'positive':int(((weights[policy,:,j]>0)&(targets[policy,:,j]>.5)).sum()),
            'negative':int(((weights[policy,:,j]>0)&(targets[policy,:,j]<.5)).sum()),
            'conflict_cells':int(((agreement[:,j]<.75)&(agreement[:,j]>0)).sum())}
            for j,target in enumerate(TARGETS)}
    manifest = {'format':'independent-source-aware-supervision-v1','sources':provenance,
                'implementation_sha256':sha256(__file__),
                'policies':['L1 uniform masked fusion','L2 supported high confidence','L3 confidence weighted',
                            'L4 train-only source-agreement reliability'],
                'coverage':coverage,'partition_counts':pd.Series(partition).value_counts().to_dict(),
                'groups':len(set(groups)),'gold_rows':int(gold.sum()),'gold_used_to_fit_source_weights':False,
                'near_duplicate_cosine_distance':.03,'seed':'independent-split-20261022',
                'l4_reliability':reliability.tolist(),'supervision_sha256':sha256(out/'supervision.npz'),
                'source_table_sha256':sha256(out/'source_aware_supervision.csv.gz'),
                'clinical_ground_truth':False,'report_text_exported':False,
                'primary_mention_mask':'Steven original or Pilkwang explicitly addresses the finding; silence remains masked',
                'source_independence_warning':'Labels share original reports; voting does not certify correctness'}
    write_json(out/'supervision_manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
