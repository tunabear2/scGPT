#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scGPT Reference Mapping
- Mode 1: 커스텀 레퍼런스 데이터셋으로 매핑
- Mode 2: CellXGene 아틀라스로 매핑
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import sklearn.metrics
import scanpy as sc
from tqdm import tqdm

sys.path.insert(0, "../")
import scgpt as scg
from build_atlas_index_faiss import load_index, vote

warnings.filterwarnings("ignore", category=ResourceWarning)


# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────
MODEL_DIR = Path("../save/scGPT_human")
CELL_TYPE_KEY = "Celltype"
GENE_COL = "index"
K_CUSTOM = 10   # 커스텀 레퍼런스 이웃 수
K_ATLAS = 50    # CellXGene 아틀라스 이웃 수


# ──────────────────────────────────────────────
# faiss 미설치 시 fallback 유사도 함수
# ──────────────────────────────────────────────
def l2_sim(a, b):
    return -np.linalg.norm(a - b, axis=1)

def get_similar_vectors(vector, ref, top_k=10):
    sims = l2_sim(vector, ref)
    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]


# ──────────────────────────────────────────────
# Mode 1: 커스텀 레퍼런스 데이터셋으로 매핑
# ──────────────────────────────────────────────
def run_custom_reference_mapping():
    print("=== Mode 1: 커스텀 레퍼런스 매핑 ===")

    # 레퍼런스 임베딩
    ref_adata = sc.read_h5ad("../data/annotation_pancreas/demo_train.h5ad")
    ref_embed = scg.tasks.embed_data(
        ref_adata,
        MODEL_DIR,
        gene_col=GENE_COL,
        obs_to_save=CELL_TYPE_KEY,
        batch_size=64,
        return_new_adata=True,
    )

    # 레퍼런스 UMAP 시각화
    sc.pp.neighbors(ref_embed, use_rep="X")
    sc.tl.umap(ref_embed)
    sc.pl.umap(ref_embed, color=CELL_TYPE_KEY, frameon=False, wspace=0.4)

    # 쿼리 임베딩
    test_adata = sc.read_h5ad("../data/annotation_pancreas/demo_test.h5ad")
    test_embed = scg.tasks.embed_data(
        test_adata,
        MODEL_DIR,
        gene_col=GENE_COL,
        obs_to_save=CELL_TYPE_KEY,
        batch_size=64,
        return_new_adata=True,
    )

    # KNN 기반 레이블 전파 (faiss fallback)
    ref_X = ref_embed.X
    test_X = test_embed.X
    preds = []
    for i in range(test_X.shape[0]):
        idx, _ = get_similar_vectors(test_X[i][np.newaxis, ...], ref_X, top_k=K_CUSTOM)
        top_label = ref_embed.obs[CELL_TYPE_KEY][idx].value_counts().index[0]
        preds.append(top_label)

    gt = test_adata.obs[CELL_TYPE_KEY].to_numpy()
    acc = sklearn.metrics.accuracy_score(gt, preds)
    print(f"Accuracy: {acc:.4f}")
    return acc


# ──────────────────────────────────────────────
# Mode 2: CellXGene 아틀라스로 매핑
# ──────────────────────────────────────────────
def run_cellxgene_atlas_mapping(index_dir: str, test_embed_X: np.ndarray, gt: np.ndarray):
    print("=== Mode 2: CellXGene 아틀라스 매핑 ===")

    index, meta_labels = load_index(
        index_dir=index_dir,
        use_config_file=False,
        use_gpu=False,  # faiss GPU 미사용
    )
    print(f"로드된 셀 수: {index.ntotal:,}")

    distances, idx = index.search(test_embed_X, K_ATLAS)

    voting = []
    for preds in tqdm(meta_labels[idx], desc="Voting"):
        voting.append(vote(preds, return_prob=False)[0])
    voting = np.array(voting)

    print(f"\n정답 레이블 (상위 10개): {gt[:10]}")
    print(f"예측 레이블 (상위 10개): {voting[:10]}")

    # 예시: endothelial 세포 결과 확인
    ids_m = np.where(gt == "endothelial")[0]
    if len(ids_m) > 0:
        print(f"\nEndothelial 세포 {len(ids_m)}개 발견")
        print(f"  예측: {voting[ids_m]}")
        print(f"  정답: {gt[ids_m]}")

    return voting


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Mode 1
    run_custom_reference_mapping()

    # Mode 2 — index_dir 경로를 실제 경로로 변경하세요
    # test_adata와 test_embed는 Mode 1에서 생성된 것을 재사용할 수 있습니다
    # run_cellxgene_atlas_mapping(
    #     index_dir="path_to_faiss_index_folder",
    #     test_embed_X=test_embed.X,
    #     gt=test_adata.obs[CELL_TYPE_KEY].to_numpy(),
    # )
