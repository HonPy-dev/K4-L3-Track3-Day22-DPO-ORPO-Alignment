# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Hồng Phi
**Mã SV:** 2A202602750
**Khoá/lớp:** K4-L3B
**Tier đã chạy:** T4 (Colab, Tesla T4 14.56 GB)
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4 · 14.56 GB tổng VRAM; peak allocation chưa được ghi lại |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1,000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` · tiếng Việt · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9%; median 94 vs 86 tokens |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | Local RM panel: chỉ `Skywork-Reward-V2-Llama-3.2-3B` được giữ lại (sanity 12/12 = 100%); Qwen3 RM đạt 8/12 = 66.7% và bị loại |
| Chi phí | Colab miễn phí · $0 |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 28:08 cho 100 bước (log Trainer) |
| VRAM cao nhất | Chưa được ghi riêng; T4 có 14.56 GB tổng VRAM |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.09097 (chosen +0.36923, rejected +0.27826) |
| Độ chính xác reward trên held-out | 0.66 |
| Margin trên held-out | +0.07676 (chosen +0.37941, rejected +0.30265) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 526.3 → 520.8 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Trên train, `rewards/chosen` tăng từ gần 0 lên +0.369 và `rewards/rejected` cũng tăng nhưng thấp hơn, lên +0.278. Vì cả hai đều tăng, model chưa hạ log-prob chosen để tạo margin; đây không phải likelihood displacement. Margin train tăng dần qua các mốc log: khoảng +0.010 ở bước 25, +0.050 ở bước 50, +0.072 ở bước 75 và +0.091 ở bước 100. Held-out cuối cùng đi cùng hướng: chosen +0.379, rejected +0.303, margin +0.077 và reward accuracy 0.66. Khoảng cách train/eval còn gần nhau, nên chưa thấy dấu hiệu overfit rõ trong thí nghiệm nhỏ này, dù chỉ có 100 cặp held-out và chưa đủ để khẳng định khả năng tổng quát rộng. Loss đầu tiên là 0.69268, gần `log(2)=0.6931`, xác nhận policy khởi tạo gần reference SFT. Chẩn đoán `INTENDED` khớp với đường cong: chosen tăng, rejected tăng ít hơn, margin dương.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 100 | 12 | 13 | 75 | 0.495 [0.445, 0.545] | 0.473 (n=94) | 0.56 |
| hữu ích — helpfulness (4) | 4 | 1 | 1 | 2 | 0.50 [0.125, 0.875] | 0.667 (n=3) | 0.00 |
| an toàn — safety (4) | 4 | 1 | 1 | 2 | 0.50 [0.125, 0.875] | 0.333 (n=3) | 0.50 |

Giám khảo: `rm-panel:Skywork-Reward-V2-Llama-3.2-3B` · sanity 12/12 (100%); Qwen3 RM bị loại ở 8/12 (66.7%). `score_length_spearman`: Llama +0.107, Qwen +0.028; position consistency không áp dụng vì RM chấm mỗi câu trả lời riêng.

Khoảng tin cậy held-out [0.445, 0.545] chứa 0.5, nên kết luận bảo thủ là chưa có bằng chứng DPO thắng SFT. Llama judge pass sanity tiếng Việt, nhưng chỉ có 12 cặp kiểm tra nên đây không phải bảo đảm chất lượng tuyệt đối. Trên 100 held-out, DPO thắng 12, SFT thắng 13 và 75 hoà; length-matched win rate còn 0.473. Câu dài hơn thắng 56% số cặp có quyết định, trong khi DPO ngắn hơn trung bình khoảng 11 ký tự, nên không có dấu hiệu đơn giản rằng DPO thắng nhờ viết dài. Hai RM bất đồng một phần: Qwen cho DPO win rate 0.55 nhưng chỉ pass sanity 66.7%; Llama cho 0.495 và là judge được dùng. Agreement là 83.3%, nên kết quả phụ thuộc judge đáng lưu ý, không nên chọn riêng con số thuận lợi của Qwen. Ví dụ hữu ích: ở prompt quicksort, cả hai trả lời đúng và giải thích pivot/phân hoạch; DPO có chút diễn đạt khác nhưng còn sinh chuỗi `</tool_call>`, không phải cải thiện rõ rệt. Ví dụ an toàn: với yêu cầu pha chất nổ, cả SFT lẫn DPO đều từ chối và khuyên tránh hành vi nguy hiểm; DPO không cho thấy khác biệt thắng-thua rõ rệt.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

Chưa chạy beta-sweep. Tôi dự đoán β nhỏ hơn (0.05) cho phép policy lệch reference mạnh hơn, có thể tăng margin nhưng cũng tăng rủi ro làm giảm chất lượng/ngôn ngữ. β lớn hơn (0.5) sẽ kìm update, giữ đầu ra gần SFT và thường làm margin hoặc reward accuracy tăng chậm hơn. Quan hệ có thể không đơn điệu; cần so reward accuracy held-out, margin, độ dài và judge NB4 thay vì chỉ chọn β có margin lớn.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

Tôi chọn tăng `JUDGE_PROMPTS` lên 100 thay vì chỉ dùng vài câu ví dụ hoặc một judge API đơn lẻ. Phương án thay thế là chấm bằng một reward model duy nhất không sanity-check, hoặc dùng API judge (tốn phí/đòi hỏi key và có thể thiên vị vị trí). 100 prompt held-out giúp ước lượng win rate có độ bất định nhìn thấy được và kiểm tra riêng helpfulness/safety bằng tám prompt cố định. Kết quả xác nhận rằng cần sanity-check: Qwen3 RM chỉ đạt 66.7% trên 12 cặp kiểm tra nên bị loại, còn Llama RM đạt 100% và được dùng. Điều làm tôi bất ngờ là DPO có tín hiệu huấn luyện rõ (gap train +0.091, held-out accuracy 0.66) nhưng so sánh SFT–DPO gần như hoà: 75/100 tie và win rate DPO 49.5%, CI vẫn chứa 0.5. Nghĩa là reward trên cặp huấn luyện không tự động biến thành cải thiện hành vi nhìn thấy được. Nếu làm lại, tôi sẽ thêm đánh giá thủ công mù trên một mẫu nhỏ và kiểm tra/loại các token điều khiển thừa như `</tool_call>` trước khi kết luận. Tôi cũng sẽ ghi rõ thời gian và VRAM peak ngay lúc huấn luyện thay vì suy từ tổng VRAM của T4. Quyết định này làm kết luận kém “đẹp” hơn một win rate tích cực, nhưng minh bạch về judge bị loại và CI chứa 0.5 quan trọng hơn việc tối ưu con số báo cáo.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | 200/subtask | — | — | Chưa có kết quả |
| GSM8K | 250 | — | — | Chưa chạy |
| Global-MMLU-vi | 10/môn | — | — | Chưa chạy |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

NB6 chưa hoàn tất nên không có điểm benchmark/stderr để báo cáo và tôi không suy diễn kết quả. Cấu hình dự kiến giới hạn IFEval 200 mẫu mỗi subtask, GSM8K 250 mẫu, Global-MMLU-vi 10 mẫu/môn, chạy SFT và SFT+DPO với chat template. Tiến trình đầu tiên không trả về trong hơn 65 phút và notebook không hiển thị progress do `subprocess.run(capture_output=True)`; hiện chưa có traceback hay file kết quả để biết nó chậm vì sinh output hay bị kẹt khi tải dữ liệu. Không thể kết luận alignment tax hoặc so benchmark với NB4 từ trạng thái này. Đây là giới hạn của lượt chạy/tooling chưa chẩn đoán được, không phải bằng chứng rằng điểm mô hình bằng 0 hoặc DPO gây suy giảm. Cũng chưa thể xác nhận thư mục output có chứa sample trung gian hữu ích hay không khi cell còn chiếm kernel. Nếu tiếp tục, cần dừng cell hiện tại, kiểm tra output log/thư mục để tránh ghi đè, rồi thử IFEval với limit nhỏ (chẳng hạn 20) và generation cap rõ ràng, đồng thời stream log ra file để thấy tiến độ. Chỉ khi lượt thử nhỏ hoàn tất mới nên chạy đủ sáu lượt SFT/DPO và tính Δ cùng stderr. Vì đây là bonus, tôi ưu tiên giữ lại artifacts NB0–NB5 đã hoàn thành hơn là để một lệnh không có progress chạy vô thời hạn.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.59 | +0.0199 | 394.1 | INTENDED |
| RPO | 0.60 | +0.0282 | 374.7 | chosen/rejected rewards +0.509/+0.481 |
| DPO-norm | 0.54 | +0.0031 | 387.9 | FAILURE; rewards đều âm |
| LD-DPO | 0.56 | +0.0200 | 393.0 | LIKELIHOOD DISPLACEMENT |
| ORPO | 0.65 | N/A (log-odds ratio −0.623) | 351.1 | thang đo khác, không so trực tiếp margin |

Trong cùng probe 20 prompt, ORPO cho đầu ra ngắn nhất (351 ký tự trung bình), còn DPO dài nhất (394); LD-DPO gần DPO (393). Chênh lệch không lớn và mẫu probe nhỏ, nên chưa đủ để quy kết cho loss. ORPO thêm NLL trên chosen cùng odds-ratio, có thể giữ câu trả lời gần mục tiêu chosen mà không phải tăng xác suất mọi token dài; LD-DPO giảm trọng số token vượt phần chung nên lý thuyết nhắm trực tiếp đến khác biệt độ dài, nhưng ở run này câu trả lời vẫn dài tương đương baseline. Accuracy reward cao nhất là ORPO 0.65, nhưng log-odds ratio của ORPO không cùng thang với DPO margin.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | Chưa chạy |
| Sai số chuẩn ≈ √(p(1−p)/n) | Chưa có số liệu |

Chưa chạy NB7; không có accuracy trước/sau hoặc reward components để kết luận.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [x] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là reward accuracy held-out của DPO đạt 0.66 nhưng judge SFT-vs-DPO trên held-out gần như ngang nhau (49.5% DPO win, 75% tie). NB3b cũng cho thấy biến thể có accuracy cao hơn không tự động chứng minh chất lượng đầu ra tốt hơn; cần đọc reward trajectory và đánh giá độc lập.
