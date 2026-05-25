# 在 FinReflectKG (2000-sample) 上跑 TTransE

這份文件說明如何用 **INK-USC/RE-Net** 的 TTransE baseline 跑
`FinReflect/finreflectkg_subset/three_relations_sample2000.tsv` 資料集。

---

## 0. 資料對應關係

原始 TSV 有 12 個欄位，本管線只取 4 個：

| TSV 欄位        | 對應到 TTransE 的角色 | 備註                                        |
|----------------|--------------------|---------------------------------------------|
| `entity`       | head（主體）         | 公司 ticker，例如 `amzn`                      |
| `relationship` | relation（關係）     | 3 種：`operates_in`, `produces`, `supplies`  |
| `target`       | tail（受體）         | 產品、地點、segment 等                        |
| `year`         | time（時間）         | 2014–2024 共 11 個時間步                      |

`start_date / end_date / triplet_id / source_file / chunk_id` 等欄位在這個
最小可跑範例中暫不使用（之後若要做「時段」建模可再延伸）。

---

## 1. 環境

RE-Net 原作者建議的環境：

```bash
conda create -n renet python=3.6 numpy
conda activate renet
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html
# 跑 TTransE 不需要 dgl，跑 RE-Net 主模型才需要
```

更新版的 PyTorch（>= 1.8）也能跑——本資料夾的 `model.py` 已將
`xavier_uniform` 改成新版的 `xavier_uniform_`，並把 `total_loss[0]`
換成 `.item()`，相容到 PyTorch 2.x。

CPU 也能跑（此資料集只有 ~2000 筆），但 GPU 較快。

---

## 2. 一鍵執行

```bash
cd Methods/RE-Net/baselines

# 快速 smoke run：10 epoch、dim=50，~1 分鐘以內可跑完
bash run_finreflectkg.sh

# 完整訓練：500 epoch、dim=100
FULL=1 bash run_finreflectkg.sh

# 指定 GPU 0
CUDA=0 FULL=1 bash run_finreflectkg.sh
```

執行完會在 `data/FinReflectKG_TTransE/` 看到資料、在 `model/FinReflectKG/`
看到 checkpoint，並在 stdout 印出 Hits@1/3/10、MeanRank、MRR 等指標。

---

## 3. 各檔案的角色

```
Methods/RE-Net/baselines/
├── preprocess_finreflectkg.py    # TSV -> RE-Net 的 *2id.txt 格式
├── verify_finreflectkg_data.py   # 純 Python 驗證資料正確性（無 torch 依賴）
├── run_finreflectkg.sh           # 一鍵串接前處理+訓練+評估
├── README_FinReflectKG.md        # 你正在讀的這份
│
├── TTransE.py                    # 訓練主程式（已小修補相容性）
├── model.py                      # TTransE / TA-TransE / TA-DistMult 模型定義
├── evaluation_TTransE.py         # link prediction 評估
├── data.py / utils.py / loss.py  # RE-Net 原始輔助程式
│
└── data/FinReflectKG_TTransE/    # 前處理產出
    ├── train2id.txt              # head\trel\ttail\ttime  (1580 筆)
    ├── valid2id.txt              # 197 筆
    ├── test2id.txt               # 198 筆
    ├── stat.txt                  # "1605\t3\t11"   (|E| |R| |T|)
    ├── entity2id.txt             # ticker / target → id  (查表用)
    ├── relation2id.txt
    └── time2id.txt
```

---

## 4. 評分函數

實作的就是 Leblay & Chekol 2018 提出的 **Vector-based TTransE**：

$$
\text{score}(s, p, o, t) \;=\; -\lVert \mathbf{s} + \mathbf{p} + \mathbf{t} - \mathbf{o} \rVert_{L1}
$$

時間步 t 與實體、關係共用同一個 embedding 空間，藉由「向頭實體與關係再加一個時間平移」來編碼時間資訊。

訓練目標：margin-based ranking loss，配合隨機負採樣（替換 head 或 tail）。

---

## 5. 統計

預處理完的 FinReflectKG_TTransE 規模：

| 指標 | 數值 |
|------|------|
| 實體數 (|E|) | 1605 |
| 關係數 (|R|) | 3 |
| 時間步 (|T|) | 11（2014–2024 各一年）|
| 唯一 quadruple 數 | 1975（去重後）|
| Train / Valid / Test | 1580 / 197 / 198（80/10/10 隨機切，seed=42）|

這個規模比 ICEWS14 (~12k 實體 / ~90k 事件) 小 50–100 倍，所以：
- 訓練幾分鐘就能收斂
- 但測試集的 ranking 噪音也比較大，建議多跑幾個 seed 看穩定性

---

## 6. 進階：常見調整

**改成「依時間切」而非隨機切**
編輯 `preprocess_finreflectkg.py`，把 `random.shuffle(quads_id)` 之前先按
time_id 排序，再用 `quads_id[:n_train]` 取早期年份做 train、後期年份做 test。
這比較符合 RE-Net 那種 *extrapolation*（外推未來）的評估場景。

**使用時段而非單點時間**
TSV 裡有 `start_date` 與 `end_date`。把它們轉成 `(year_start, year_end)`，
然後在前處理時為每個 quadruple 展開成 `[year_start, year_end]` 區間裡的多筆
時點，再用同一套 TTransE 訓練。這對應論文裡「temporal sampling, TS=N」的策略。

**換成更現代的 TKGE 模型**
同樣的資料格式可以直接餵給：
- `TATransE.py` / `TADistmult.py`（也在 baselines/ 內）
- TeRo / ATiSE：clone https://github.com/soledad921/ATISE，按它的資料夾結構放置即可
- RE-Net 本體：用 RE-Net repo 根目錄的 `data/<DATASET>/get_history_graph.py`
  先把我們的 train/valid/test 轉成它需要的 history 結構

---

## 7. Troubleshooting

* **`RuntimeError: context has already been set`**
  已在 TTransE.py 加上 try/except 處理，新跑就不會出現。

* **`AttributeError: 'Tensor' object has no attribute 'item'`**
  你的 PyTorch < 0.4。請升級到 1.x 以上。

* **`CUDA out of memory`**
  把 `BS=64 DIM=50 bash run_finreflectkg.sh` 即可。

* **Mean Rank 非常差（>1000）**
  正常。我們有 1605 個實體但只有 1975 筆三元組，KG 非常稀疏；先確認訓練 loss 在下降即可，再加大 epoch / 改成依時間切。

---

## 8. 引用

```bibtex
@inproceedings{leblay2018ttranse,
  title={Deriving Validity Time in Knowledge Graph},
  author={Leblay, Julien and Chekol, Melisachew Wudage},
  booktitle={Companion Proceedings of The Web Conference 2018},
  year={2018}
}

@inproceedings{jin2020renet,
  title={Recurrent Event Network: Autoregressive Structure Inference over Temporal Knowledge Graphs},
  author={Jin, Woojeong and Qu, Meng and Jin, Xisen and Ren, Xiang},
  booktitle={EMNLP},
  year={2020}
}
```

TTransE baseline 程式碼來自 https://github.com/INK-USC/RE-Net
