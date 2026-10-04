# CST: train lại math-only và kiểm tra cơ chế spectral

Ngày lập: 2026-10-03. Phạm vi chính: Qwen2.5-14B-Instruct → Qwen2.5-1.5B-Instruct; xác nhận cấu hình tốt trên Qwen3-8B → Qwen3-1.7B như bản thảo.

Đây là protocol đề xuất sau khi đọc bản thảo và source code; chưa generate dữ liệu hay chạy training. Các cấu hình dưới đây là ứng viên thực nghiệm, không phải kết quả đã xác nhận. Repo được đọc ở commit `127dc04c330ec47391fff5e6cd5b4674af5dfc32`.

## 1. Quyết định chính

1. Tạo một tập **prompt math chung**, khử trùng lặp và đối chiếu với hợp của tất cả benchmark đánh giá trước khi lấy mẫu.
2. Mỗi **teacher** generate một phiên bản response từ tập prompt đó. Mọi baseline của cùng teacher–student dùng đúng phiên bản dữ liệu đã đóng băng. Một model sau training được đánh giá trên cả năm benchmark; không tạo năm tập train riêng theo năm tập test.
3. Bảng Qwen2.5 cũ giữ làm lịch sử; mọi hàng dùng dữ liệu hoặc evaluation khác protocol phải chạy lại, không chuyển số sang bảng mới.
4. Trước hết chạy output-only đối chứng và CST chuẩn. Sau đó thử CST normalize response với hệ số cố định 1, rồi sweep weight, q, layer count, layer position và gamma.
5. Để nghiên cứu nuclear norm/rank, ưu tiên đo phổ trong **native student hidden states** và thêm đối chứng scalar. Chỉ thêm regularizer chống collapse khi probe thực sự cho thấy thiếu dimensionality ở các layer liên quan.

## 2. Những điểm trong repo cần xử lý trước khi chạy

| Điểm đã đọc được trong source | Hệ quả | Việc cần làm |
|---|---|---|
| `tools/generate_vllm.py` đọc toàn bộ `Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen` và lấy cột `instruction` | Chưa bảo đảm math-only; trang dataset có cả coding/logic | Đọc manifest prompt math đã lọc từ UltraInteract gốc |
| `tools/process_data_ultraInteract.py`: `valid_data = random.sample(raw_data, ...)`, nhưng `train = raw_data` | Nếu bật nhánh này, validation nằm trong training | Split theo problem group trước generation; train/dev không giao nhau |
| Preprocessor hiện nhận `prompt: string`, `generated_text: string` | Không nhận trực tiếp schema `instruction/prompt:list/response` trong ảnh | Viết adapter schema, không stringify cả list chat |
| Generation cắt prompt ở 1020 token; preprocessing cắt ở 512; train wrapper có `MAX_LENGTH=1025` | Có thể cắt mất câu hỏi, delimiter hoặc đáp án cuối | Chốt length policy sau pilot; thống kê đầy đủ mẫu vượt giới hạn |
| `cst_module.py` flatten `[B,T]` rồi lấy tổng cộng tối đa 64 token | Gram là của một tập token trong micro-batch, khác công thức per-example trong bản thảo | Khuyến nghị tính CST riêng từng example, rồi mean theo batch; sửa structural baselines tương ứng |
| Wrapper Qwen2.5 bật `--student-gen --type adaptive-sfkl` | Có thể dùng student rollouts/replay thay vì chỉ teacher responses | Chọn rõ controlled off-policy hoặc faithful on-policy, ghi đúng protocol |
| `cst_distance` chỉ có `l2`/`smooth_l1` | Chưa có normalized-response CST | Thêm variant có công thức cụ thể ở mục 5 |
| `bnmm` trả `nn_t - nn_s` trong `nnm_variants.py` | Teacher là hằng số đối với gradient student: đây không phải loss kéo về target | Dùng squared matching hoặc hinge có điểm dừng nếu muốn teacher-guided norm |
| `bnm`, `bnmm`, `erank` được dispatch qua projector và random projection | Metric tốt sau projection chưa chứng minh native student được cải thiện | Tạo scalar controls trực tiếp trên native hidden cho câu hỏi của CST |
| Hàm legacy trong `nnm_module.py` dùng `math.log(nn_s)` | Có thể tách giá trị student khỏi autograd nếu gọi hàm này | Main dispatcher đang gọi bản `nnm_variants.py` dùng `torch.log`; kiểm tra đúng call path và gradient |

Không kết luận rằng mọi training cũ đều bị các lỗi trên: các kết luận phụ thuộc script/call path thực sự được dùng. Source công khai chưa phải bằng chứng về config của từng run cũ.

## 3. Dữ liệu chuẩn dùng chung

### 3.1 Lọc prompt từ UltraInteract-SFT

- Nguồn: `openbmb/UltraInteract_sft`; pin revision.
- In histogram giá trị `task` trước khi lọc. Dataset card dùng `Math_Cot` và `Math_PoT`; không lọc bằng chuỗi trong văn bản paper như `Math-CoT` mà chưa kiểm tra.
- Giữ các bài toán thuộc Math-CoT và Math-PoT, bỏ Coding/Logic. Với PoT, lấy lại **đề toán gốc**, bỏ wrapper Python/tool để teacher generate lời giải văn bản theo prompt chung.
- Chỉ dùng initial problem. Không đưa lời giải trước đó, execution output, critique hay lượt sửa sai vào prompt.
- Nhóm theo `parent_id` khi hợp lệ; sau đó dedup bằng đề toán canonical vì cùng một bài có thể xuất hiện ở cả CoT và PoT hoặc nhiều parent.
- Giữ nguyên số, ký hiệu toán, điều kiện, hình thức câu hỏi và các lựa chọn đáp án. Chuẩn hóa để matching chỉ xử lý whitespace, Unicode và các wrapper đã biết; không xóa tùy tiện dấu toán học hoặc lowercase biến nhạy hoa/thường.
- Lưu `source_id`, `parent_id`, `source_dataset`, `source_task`, `problem_hash`, cùng lý do loại mẫu.

### 3.2 Decontamination trước sampling

Tạo một exclusion manifest gồm đề bài của tất cả benchmark định báo cáo: GSM8K test, GSM-Plus, MATH-500, MMLU-Pro phần math, OlympiadBench đúng subset evaluation và AIME 2024/2025/2026. Pin dataset revision, split, subset và ID. Với AIME, ghi rõ có dùng cả I và II hay không.

Thứ tự kiểm tra:

1. Nối bằng source ID/original problem ID khi có; theo dõi cả bài gốc của dữ liệu phái sinh.
2. Exact match trên đề toán canonical, sau khi bỏ prompt wrapper.
3. Tạo ứng viên near-duplicate bằng token/character shingles; kiểm tra lại các cặp tương đồng cao. Có thể bắt đầu từ token 5-gram Jaccard ≥ 0.8, nhưng phải kiểm tra ngưỡng trên các cặp được đánh nhãn, đặc biệt bài ngắn.
4. Dùng retrieval semantic để bổ sung ứng viên paraphrase nếu có tài nguyên; không tự động loại chỉ vì embedding giống nhau. Kiểm tra lại điều kiện, con số và câu hỏi đích.
5. Loại cả problem family/duplicate cluster của một mẫu trùng; không chỉ loại đúng một trajectory.

GSM-Plus cần chú ý quan hệ với câu GSM8K gốc. MATH-500 cần kiểm tra provenance về MATH. Nếu provenance không đáng tin, chọn quy tắc bảo thủ loại toàn bộ parent test pool có liên quan và công bố lượng dữ liệu bị loại.

Lọc bằng câu hỏi/ID, không đưa test answers vào prompt generation. Báo cáo số exact, near-duplicate và family matches; không tuyên bố loại được mọi paraphrase hoặc contamination trong pretraining của teacher.

### 3.3 Chọn số mẫu và split

- Giữ mục tiêu **35,000 train prompts** của bản thảo nếu pool sạch đủ lớn.
- Tách khoảng **1,000 dev prompts** theo duplicate cluster trước generation; dev và train không giao nhau.
- Seed chọn dữ liệu: `42`; sort ổn định theo ID/hash rồi lấy mẫu, có thể stratify theo source dataset. Dùng cùng prompt IDs cho các teacher.
- Nếu pool không đủ 36,000 prompt unique sạch, báo số thực tế; không bù bằng duplicate. Mục tiêu 35k chưa được kiểm tra trên dữ liệu trong phiên này.
- Dev accuracy phải dùng đáp án đã kiểm chứng từ nguồn gốc. Không mặc định teacher-generated answer là ground truth. Nếu không dựng được gold dev từ UltraInteract, dùng một tập dev riêng từ official training splits, rồi loại toàn bộ cluster tương ứng khỏi train.

### 3.4 Prompt và schema

Giữ nguyên câu trong ảnh, thêm đề bài sau một newline:

```text
Solve the following math problem step-by-step.
Simplify your answer as much as possible. Present your final answer as \boxed{Your Answer}.
{problem}
```

Chat message chỉ có user; dùng đúng `apply_chat_template` của model, không bọc template hai lần. Chuỗi canonical `instruction` phải bằng `prompt[0].content`.

```json
{
  "instruction": "Solve the following math problem step-by-step.\nSimplify your answer as much as possible. Present your final answer as \\boxed{Your Answer}.\n{problem}",
  "prompt": [{"role": "user", "content": "<same instruction>"}],
  "response": "<new response generated by the designated teacher>"
}
```

Đây là schema minh họa; metadata provenance có thể nằm trong sidecar manifest để giữ ba cột giống ảnh. Adapter cho preprocessor hiện tại ánh xạ `prompt ← instruction`, `generated_text ← response`, nhưng phải thay nhánh split sai trước khi dùng.

### 3.5 Generate theo teacher

| Corpus | Generator | Student và các baseline dùng chung |
|---|---|---|
| `qwen25_14b_math_train.jsonl` | Qwen2.5-14B-Instruct | Qwen2.5-1.5B-Instruct |
| `qwen3_8b_math_train.jsonl` | Qwen3-8B | Qwen3-1.7B |

- Pilot 256–512 prompt để chọn giới hạn độ dài; khởi đầu `temperature=0.85`, `top_p=0.95`, 1 response/prompt, seed cố định, `max_new_tokens=4096`. Đây là config đề xuất, không phải tái hiện nguyên xi TSD-KD.
- Chốt generation cap theo pilot, tỉ lệ truncation và budget. Training max length phải chứa được full prompt + response + EOS cho corpus chính; không cắt âm thầm cho vừa 1025 token.
- Giữ `finish_reason`, số token, parse status, boxed-answer status và teacher revision. Mẫu chạm length cap phải có policy retry/drop thống nhất trước khi freeze corpus.
- Với Qwen3, ghi rõ thinking mode và chat template. Dùng cùng quyết định cho các baseline; không trộn mode giữa các run.
- Generation xong một lần rồi tái sử dụng cùng bytes cho mọi baseline. Muốn lọc đáp án sai phải thực hiện ở cấp corpus chung và báo cáo retention; không lọc riêng cho CST.
- Lưu SHA-256 corpus, prompt manifest, tokenizer/chat-template hash, seed và generation config.

## 4. Train lại baseline công bằng

### 4.1 Hai nhóm so sánh

**Controlled experiment, ưu tiên để kết luận về CST:** tất cả dùng cùng teacher-generated prefixes, cùng SFKL output loss của Eq. 1, cùng token mask, optimizer, effective batch size, số update, data order và init seed. Bỏ `--student-gen` cho nhóm này. Output-only SFKL là đối chứng trực tiếp của `SFKL + CST`.

**Faithful method comparison:** DistiLLM/AMiD/CSD giữ thành phần sampling/on-policy đặc trưng nếu có trong implementation gốc. Tất cả vẫn dùng cùng base prompt corpus; ghi riêng student-rollout tokens, teacher queries, replay và training compute. Không gọi mọi run dùng SFKL trên fixed data là bản tái hiện đầy đủ của DistiLLM.

Nếu câu hỏi là CST có bổ sung được cho DistiLLM hiện tại không, chạy thêm cặp **DistiLLM vs DistiLLM+CST** có cùng on-policy scheduler. Khi đó mô tả CST được tính trên prefixes được chọn ở mỗi bước, thay vì tuyên bố chỉ teacher prefixes.

### 4.2 Danh sách cần có

| Nhóm | Phương pháp | Vai trò |
|---|---|---|
| Reference | Student; teacher | Eval lại bằng đúng evaluator mới |
| Data control | SeqKD/SFT trên teacher responses | Đo lợi ích chỉ từ corpus mới |
| Output control | SFKL off-policy | Đối chứng trực tiếp, λCST = 0 |
| Published methods | DistiLLM, CSD, AMiD | Giữ objective và sampling đặc trưng; báo compute |
| Native structural controls | CKA, Gram, normalized spectrum, direct spectrum | Cùng m, token selection, layer map và output backbone |
| Feature control | Hidden MSE + projector | Ghi projector architecture và optimizer |
| Spectral scalar | NuNo-KD/NNM; normalized nuclear matching | Phân biệt centroid/projector method với native scalar control |
| Ours | CST; normalized-response CST, coefficient = 1 | So sánh profile supervision |

Mỗi auxiliary baseline cần budget tune weight hợp lý; không dùng cùng một hệ số số học cho các loss có thang đo khác nhau rồi kết luận baseline kém. Loss phải được mean theo đúng q/layer/example; nếu sum thì layer count hoặc q sẽ đồng thời đổi strength.

Token-level KD yêu cầu kiểm tra teacher/student token IDs và vocabulary tương thích. Ghi rõ full fine-tuning hay LoRA; tên thư mục chứa `lora` không đủ xác nhận training mode.

Chạy seed `10` cho exploration. Sau khi chọn config bằng dev, chạy seed `10, 42, 2026` cho các so sánh chính. Tính Avg-5 riêng từng seed rồi report mean ± SD, không suy SD của Avg bằng cách lấy trung bình các SD theo task.

## 5. CST gốc, normalized CST và sweep

### 5.1 Định nghĩa hiện tại

Với từng example, lấy cùng m vị trí response cho teacher/student:

\[
X=H-\mathbf{1}\bar h^\top,\quad G=XX^\top/\|X\|_F^2,
\quad \Phi_H(\gamma)=\log\det(I+\gamma G).
\]

\[
L_{CST}=\operatorname{mean}_{b,\ell,j}(\Phi_S(\gamma_j)-\operatorname{sg}\Phi_T(\gamma_j))^2,
\quad L=L_{out}+\lambda L_{CST}.
\]

`q` là số gamma sample mỗi micro-batch; gamma được chia sẻ giữa teacher/student và các layer/example trong micro-batch. Tăng q với phép mean chủ yếu giảm nhiễu Monte Carlo và tăng cost; không tự động tăng độ mạnh hoặc đổi expected objective.

### 5.2 Đề xuất normalized-response CST với hệ số cố định 1

Gram của CST đã được trace-normalize. Variant mới cần normalize **response**, không chỉ lặp lại trace normalization. Một lựa chọn có diễn giải toán rõ là normalize theo khoảng khả dĩ của logdet.

Với `r=m−1`, giả sử cả hai hidden widths ≥ r, representation không suy biến và m ≥ 3:

\[
a(\gamma)=\log(1+\gamma),\qquad b_r(\gamma)=r\log(1+\gamma/r),
\]

\[
\psi_H(\gamma)=\frac{\Phi_H(\gamma)-a(\gamma)}{b_r(\gamma)-a(\gamma)},
\qquad 0\leq\psi_H\leq1.
\]

Lower bound là phổ rank 1; upper bound là phổ đều trên r chiều. Dùng cùng denominator cho S/T. Trong setting m=64 và hidden widths lớn, r=63.

\[
L_{nCST}=\operatorname{mean}_{b,\ell,j}(\psi_S-\operatorname{sg}\psi_T)^2,
\qquad L=L_{out}+L_{nCST}.
\]

- Đây là **variant đề xuất**, chưa có trong repo; không đồng nhất với baseline `normalized_spectrum`.
- Coefficient bằng 1, không sweep weight cho variant này như yêu cầu. Tuy nhiên vẫn là một quyết định về strength; loss bounded không bảo đảm gradient cân bằng với output loss.
- Normalization làm đổi weighting giữa các gamma. Báo như variant riêng, không coi là chỉ đổi tên objective gốc.
- Ở gamma rất nhỏ, cả numerator/denominator dễ cancellation. Thử trước trên `[0.1,10]`, kiểm tra bằng float64 và các spectrum biết trước; tránh bắt đầu từ gamma cực nhỏ. Giới hạn gamma→0 của ψ là `(1−tr(G²))/(1−1/r)`.
- Với m<3 hoặc energy gần 0: bỏ contribution đó và log tỉ lệ; không gán ψ bằng 0 như thể đó là một spectrum hợp lệ. Không dùng actual numerical rank làm r vì denominator sẽ thay đổi theo model.
- Nếu muốn warmup/ramp, chốt cùng schedule trước sweep; normalized variant chỉ cố định hệ số đích 1. Ghi rõ schedule, không gọi là hoàn toàn không có lựa chọn siêu tham số.

### 5.3 Sweep có thứ tự

Anchor đề xuất: m=64; q=2; L=4; relative depth `[0.20,0.85]`; gamma LogUniform(0.1,10); loss L2; per-example Gram. Không coi anchor này là best config Qwen2.5 chỉ vì từng có kết quả Qwen3.

| Thứ tự | Biến | Ứng viên | Giữ cố định/đánh giá |
|---|---|---|---|
| A | λ của CST gốc | `0, 1e-4, 1e-3, 3e-3, 5e-3, 1e-2` | λ=0 tái dùng output-only; mở rộng 3e-2 chỉ khi optimum nằm ở biên |
| A | Normalize response | nCST, coefficient=1 | Cùng q/L/m/gamma/data; log auxiliary/output gradient ratio |
| B | q | `1,2,4,8` | Giữ phép mean; chạy trên 1–2 loss configurations tốt từ A |
| C | Layer position, L=4 | Early `[0.10,0.40]`; middle `[0.30,0.65]`; late `[0.60,0.90]`; broad `[0.20,0.85]` | So sánh vị trí với cùng số layer |
| D | Layer count | `1,2,4,8` | Giữ range thắng ở C; L=1 lấy midpoint; log actual indices |
| E | Gamma range | `[0.1,10]`, `[1,100]`, `[10,1000]` | Thêm độ nhạy tail; monitor numerical stability và gradient strength |
| E | Sampling control | Log-uniform random vs fixed log-grid cùng q; tùy chọn stratified log-uniform | Không so 2 endpoints với q=8 rồi quy mọi chênh lệch cho stochasticity |
| F | m | `32,64,128` | Kiểm tra độ ổn định phổ; Cholesky có cost tăng theo m³ |
| G | Schedule | Không warmup vs warmup/ramp đã chốt | Chỉ cho top config; định nghĩa step là optimizer update |

Không chạy tích Descartes tất cả biến. Screening trên cùng subset sạch và cùng budget, sau đó chạy full budget cho top 2–3 cấu hình để kiểm tra thứ hạng. Dành budget tương đương cho weight/LR quan trọng của baseline. Sau coordinate sweep, kiểm tra nhỏ interaction `top 2 weights × top 2 layer ranges` vì hai biến có thể phụ thuộc nhau.

Với γ-range rộng hơn, CST raw có thể đổi strength đáng kể; cần kiểm tra lại một lân cận weight nhỏ. Với nCST, coefficient vẫn giữ 1.

Layer map dùng relative depth từ config thật của từng model, log indices chính xác và index convention. Nếu thử layer cuối, kiểm tra consistency pre/post final norm giữa hook student và `teacher_outputs.hidden_states`; các middle-layer sweep giảm rủi ro này.

## 6. Các thí nghiệm nuclear norm/rank đáng ưu tiên

### 6.1 Đo đúng mục tiêu

Nuclear norm của Gram PSD trace-normalized bằng `tr(G)=1`, nên **không thể dùng ||G||_* làm metric cần tăng**. Cần đo nuclear norm của centered hidden X hoặc X đã normalize.

Với p_i = σ_i(X)²/||X||F²:

\[
N_{norm}=\frac{\|X\|_*}{\sqrt r\|X\|_F}
=\frac{\sum_i\sqrt{p_i}}{\sqrt r},\quad
PR=\frac1{\sum_i p_i^2},\quad
eRank_{energy}=\exp(-\sum_i p_i\log p_i).
\]

Report `N_norm`, `PR/r`, `eRank_energy/r`, raw `||X||_*`, `||X||F`, và spectral profile gap. Effective rank trong repo `erank` dùng phân phối **σ/Σσ**, khác `eRank_energy`; nếu report cả hai phải đặt tên phân biệt. Algebraic rank thường gần max ở cả hai model và không đủ nhạy.

So sánh tại cùng layer, m, response positions và prompt/response text. Tăng raw nuclear norm cùng Frobenius norm, trong khi normalized metrics không đổi, chỉ cho thấy thay đổi scale. CST bất biến global scale nên raw norm tăng không phải hệ quả được bảo đảm của objective.

### 6.2 Thứ tự thử nghiệm và ý nghĩa

| Ưu tiên | Thí nghiệm | Câu hỏi kiểm tra |
|---|---|---|
| P0 | Output-only vs CST vs nCST, probe native hidden theo layer | CST đang sửa thiếu dimensionality ở đâu, hay chủ yếu tái phân bố phổ? |
| P1 | SFKL + normalized nuclear **matching** | Một scalar đã đủ hay cần profile nhiều scale? |
| P1 | SFKL + PR matching; tùy budget thêm eRank matching | Đối chứng single-statistic gần motivation trong paper |
| P1 | Gamma-tail và middle/late-layer sweeps của CST | Có thể cải thiện small-eigenvalue structure bằng chính CST hiện tại không? |
| P2 | CST + teacher-referenced one-sided PR hoặc normalized nuclear floor | Giảm thiếu dimensionality quá mức mà không ép isotropy tối đa |
| P2 | Output-only + cùng floor, cùng weight | Tách lợi ích riêng của floor khỏi CST; cần cho kết luận cơ chế |
| P2 | Teacher-free normalized-nuclear maximization | Phân biệt spread generic với transfer thông tin teacher |
| P3 | NuNo-KD faithfully; thêm frozen-projector hoặc no-centroid ablation nếu budget | Đánh giá ảnh hưởng projector/centroid; probe native hidden riêng |
| P3 | Logdet-slope matching thay cho hoặc thêm vào CST | Kiểm tra effective dimensionality nhiều scale gắn trực tiếp Proposition 2 |

**Scalar matching trên native hidden**, ví dụ:

\[
L_N=\operatorname{mean}(N_S-\operatorname{sg}N_T)^2.
\]

Đây là đối chứng cho lập luận profile cung cấp nhiều thông tin hơn một scalar. NNM/NuNo-KD có centroid + projector là một phương pháp khác, không nên đồng nhất hai hàng này.

**Teacher-referenced floor**, chỉ thử sau khi probe cho thấy deficit:

\[
L_{floor}=\operatorname{mean}[\operatorname{ReLU}(\tau\operatorname{sg}a_T-a_S)]^2,
\quad a\in\{N_{norm}, PR/r\}.
\]

Giá trị khởi đầu đề xuất: τ=0.9; chỉ thử τ=1 nếu cần. Tổng loss `Lout + λ LCST + β Lfloor`; sweep nhỏ β∈{0.01,0.1,1} trên dev và theo dõi gradient. Không thêm PR và nuclear floor đồng thời trong thí nghiệm đầu. Đây là extension có objective mới, phải báo riêng khỏi CST gốc.

Chạy factorial tối thiểu: output-only; output-only+floor; CST; CST+floor. Nếu chỉ floor đã cho cùng cải thiện thì không quy toàn bộ gain cho characteristic transform.

**Logdet-slope**, phù hợp lý thuyết CST:

\[
D_H(\gamma)=\partial\Phi_H/\partial\log\gamma
=\operatorname{tr}[\gamma G(I+\gamma G)^{-1}]
=\sum_i\frac{\gamma p_i}{1+\gamma p_i}.
\]

`D/r` là effective dimension phụ thuộc scale. Matching hoặc floor của D tạo một ablation có cơ sở; có thể tính bằng Cholesky solve, nhưng cần đo thêm training cost. Không coi finite-gamma D là algebraic rank chính xác.

### 6.3 Liên hệ lý thuyết giúp câu chuyện bài nhất quán

CST nhắm tới **teacher spectral organization**, không bảo đảm tăng rank cho mọi mẫu/layer. Với tổng energy cố định, phổ đều hơn làm logdet tăng; nhưng matching teacher có thể cần tăng hoặc giảm tùy layer. Do đó giả thuyết nên là: giảm spectral mismatch/thiếu dimensionality ở một số tầng có thể giúp downstream math.

Một liên hệ có thể tự suy ra từ Eq. 3 là:

\[
\frac{\|X\|_*}{\|X\|_F}=\operatorname{tr}\sqrt G
=\frac1{2\pi}\int_0^\infty\Phi_H(\gamma)\gamma^{-3/2}\,d\gamma.
\]

Đẳng thức theo từng eigenvalue và integration by parts. Nó cho thấy normalized nuclear norm là một functional của **toàn bộ** logdet curve. Training trên hữu hạn gamma trong một khoảng hẹp không bảo đảm match tích phân này. Đây là diễn giải lý thuyết, chưa phải kết quả thực nghiệm hay khẳng định novelty.

## 7. Probe, evaluation và tiêu chí chọn

- Freeze 256–512 held-out math prompts làm probe; giữ nguyên teacher-generated response và m token positions cho tất cả model/checkpoint. Không dùng response tự generate của mỗi model cho causal comparison chính vì text/length khác nhau.
- Probe all layers lúc init, giữa training và cuối; xem cả supervised lẫn unsupervised layers. Diagnostic eigendecomposition/SVD chỉ chạy ở probe, không làm mất claim training CST không cần SVD.
- Báo spectral profile gap trên một fixed dense gamma grid, không chỉ loss tại random gamma dùng trong training.
- Log student-parameter gradient từ auxiliary loss để xác nhận signal vào student; không chỉ giảm loss/projector metric. Với projected methods, log gradient và norm cả trước/sau projector.
- Primary selection bằng gold dev accuracy; metric spectral là diagnostic/secondary endpoint. Không chọn checkpoint vì rank đẹp nhưng math giảm.
- Chốt evaluator, prompt, chat template, greedy decoding, token budget và answer verifier trước final test. MMLU-Pro math giữ options và có extractor phù hợp; không ép một numeric-answer parser cho mọi task.
- Main score là Avg-5 = trung bình accuracy của GSM8K/GSM-Plus/MATH-500/Pro-Math/OlympiadBench. AIME báo riêng, không dùng để tune.
- Với chênh lệch nhỏ, report paired per-example comparisons và variability theo training seeds. Bootstrap GSM8K/GSM-Plus theo family nếu chúng chia sẻ bài gốc; không coi các biến thể là hoàn toàn độc lập.
- Ghi training GPU-hours, tokens/s, peak VRAM, teacher prepass/centroid time, rollout tokens, số trainable projector parameters. Cùng số epochs chưa chắc cùng cost.

Kết quả đủ hỗ trợ nhận định mong muốn khi downstream math tốt hơn trên test đã khóa, lặp qua seeds; normalized spectral metrics cải thiện ở layer đã nêu; không chỉ raw norm phình; và scalar/teacher-free controls chưa giải thích hết gain. Nếu rank tăng nhưng accuracy không tăng, đó là kết quả đối chứng có giá trị, không phải config thắng.

## 8. Bảng kết quả mới và thứ tự triển khai

Mọi ô dưới đây chờ chạy lại. Không điền số legacy có dữ liệu/evaluator khác.

| Method | GSM8K | GSM+ | MATH-500 | Pro-Math | Olympiad | Avg-5 | Data SHA | Seeds |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Student | — | — | — | — | — | — | N/A | — |
| Teacher | — | — | — | — | — | — | N/A | — |
| SeqKD/SFT | — | — | — | — | — | — | — | — |
| SFKL off-policy | — | — | — | — | — | — | — | — |
| DistiLLM | — | — | — | — | — | — | — | — |
| CSD | — | — | — | — | — | — | — | — |
| AMiD | — | — | — | — | — | — | — | — |
| Hidden MSE | — | — | — | — | — | — | — | — |
| Gram | — | — | — | — | — | — | — | — |
| CKA | — | — | — | — | — | — | — | — |
| Direct spectrum | — | — | — | — | — | — | — | — |
| Normalized spectrum | — | — | — | — | — | — | — | — |
| NuNo-KD/NNM | — | — | — | — | — | — | — | — |
| Normalized nuclear matching | — | — | — | — | — | — | — | — |
| PR matching | — | — | — | — | — | — | — | — |
| CST | — | — | — | — | — | — | — | — |
| nCST, coefficient=1 | — | — | — | — | — | — | — | — |
| CST+floor (nếu chạy) | — | — | — | — | — | — | — | — |

| Run ID | λ/variant | q | Layer IDs S/T | Gamma range | m | Dev acc | N_norm | PR/r | eRank/r | GPU-hours |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Pending | — | — | — | — | — | — | — | — | — | — |

Thứ tự thực thi:

1. Sửa data split/schema/length và quyết định per-example CST + off/on-policy.
2. Freeze math prompt manifest, decontam report, train/dev IDs; generate và freeze teacher corpora.
3. Smoke check masking, actual layer indices, teacher stop-gradient, student gradient và normalization. Kiểm tra batch-size>1 để bắt lỗi Gram pooling.
4. Chạy output-only và anchor CST trên Qwen2.5; probe native spectra.
5. Sweep A→D, sau đó gamma/m và các scalar controls; run floor chỉ khi có evidence deficit.
6. Chọn config bằng dev; full-budget rerun top configurations và các baseline với seeds đã chốt.
7. Final test + probe + compute report; xác nhận một config đã khóa trên Qwen3.
8. Cập nhật paper từ kết quả mới. Sửa đồng thời mô tả sampling/per-example loss, số dataset và provenance của mọi hàng; tính bảng/text từ một result manifest để tránh các Avg và mô tả cũ mâu thuẫn.

## Nguồn đã kiểm tra

- Bản thảo người dùng cung cấp: `_ARR_Oct_2027__Spectral_Transform_Distillation.pdf`, đặc biệt Eq. 2–9 và Experimental Setup.
- [UltraInteract-SFT dataset card và schema](https://huggingface.co/datasets/openbmb/UltraInteract_sft/blob/main/README.md).
- [Dataset tham khảo format Minsang](https://huggingface.co/datasets/Minsang/TSD-KD-Qwen2.5-1.5B-Instruct-Gen). Dùng để đối chiếu schema; chưa xác nhận mọi chi tiết training protocol của paper TSD-KD.
- [Repo NuNo-KD tại commit đã đọc](https://github.com/chiiipk/nuno-kd/tree/127dc04c330ec47391fff5e6cd5b4674af5dfc32): `cst_module.py`, `nnm_module.py`, `nnm_variants.py`, `structural_ablation_losses.py`, `finetune.py`, `arguments.py`, generation/preprocessing scripts và Qwen2.5 wrapper.
