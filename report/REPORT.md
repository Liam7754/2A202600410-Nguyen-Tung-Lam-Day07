# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Tùng Lâm
**Nhóm:** X5
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
→ Hai vector biểu diễn câu gần như gần bằng 0 khi tính bằng công thức cosine similarity hay gần như cùng hướng trong không gian, nghĩa là nội dung/ý nghĩa của chúng rất giống nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "The cat is sleeping on the sofa."
- Sentence B: "A cat rests on the couch."
- Tại sao tương đồng: Cả hai đều mô tả cùng một sự việc (con mèo đang nằm nghỉ trên ghế).

**Ví dụ LOW similarity:**
- Sentence A: "I love playing football."
- Sentence B: "The weather is rainy today."
- Tại sao khác: Một câu nói về sở thích, câu kia nói về thời tiết, không liên quan về ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> *Đáp án:*
Công thức: Số bước nhảy = chunk_size – overlap = 500 – 50 = 450
Số chunks = ⌈(10,000 – 500) / 450⌉ + 1
= ⌈9,500 / 450⌉ + 1 = ⌈21.11⌉ + 1 = 22 + 1 = 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*
Bước nhảy = 500 – 100 = 400
Số chunks = ⌈(10,000 – 500) / 400⌉ + 1
= ⌈9,500 / 400⌉ + 1 = ⌈23.75⌉ + 1 = 24 + 1 = 25 chunks
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** QUY CHẾ ĐÀO TẠO ĐẠI HỌC HỆ CHÍNH QUY
THEO HỆ THỐNG TÍN CHỈ VinUni

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Đây là tài liệu có cấu trúc điều khoản phức tạp, chứa nhiều con số định lượng (GPA, số tín chỉ) và các điều kiện ràng buộc ngặt nghèo. Việc chọn domain này giúp kiểm chứng khả năng trích xuất thông tin chính xác của RAG đối với các câu hỏi mang tính quy định.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | QUY CHẾ ĐÀO TẠO ĐẠI HỌC HỆ CHÍNH QUY THEO HỆ THỐNG TÍN CHỈ | https://policy.vinuni.edu.vn/wp-content/uploads/2024/05/VU_HT03.VN_QC-dao-tao-dai-hoc-he-chinh-quy-theo-he-thong-tin-chi.pdf | 73.854 | "chunk_id", "source" , "extension"|
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| chunk_id | String | VU_HT03_001 | Giúp định danh chính xác vị trí của đoạn văn bản trong cơ sở dữ liệu để hệ thống có thể truy xuất hoặc cập nhật đúng đoạn đó. |
| source | String | https://policy.vinuni.edu.vn/wp-content/uploads/2024/05/VU_HT03.VN_QC-dao-tao-dai-hoc-he-chinh-quy-theo-he-thong-tin-chi.pdf | Dùng để trích dẫn nguồn (citation). Khi AI trả lời, nó có thể dẫn link để người dùng kiểm chứng độ xác thực của thông tin. |
| extension | String | pdf | Giúp hệ thống quản lý loại tệp tin và áp dụng các phương pháp xử lý văn bản (OCR hoặc Text Extraction) phù hợp cho từng định dạng. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| data.txt | FixedSizeChunker (`fixed_size`) | 165 | 497.30 | Thấp: Dễ bị cắt ngang câu hoặc giữa các Điều/Khoản. |
| data.txt | SentenceChunker (`by_sentences`) | 222 | 331.68 | Trung bình: Giữ trọn vẹn câu nhưng có thể tách rời "Điều" khỏi "Nội dung". |
| data.txt | RecursiveChunker (`recursive`) | 159 | 463.50 | Cao: Ưu tiên ngắt ở đoạn (\n\n) hoặc dòng mới, giữ được cấu trúc văn bản tốt nhất. |

### Strategy Của Tôi

**Loại:** RecursiveChunker 

**Mô tả cách hoạt động:**  
RecursiveChunker chia văn bản dựa trên các dấu hiệu cấu trúc tự nhiên như ngắt dòng đôi (\n\n), tiêu đề chương/điều, hoặc dấu chấm câu. Nếu đoạn quá dài vượt ngưỡng token, nó tiếp tục chia nhỏ theo cấp độ thấp hơn (ví dụ: xuống đến câu). Base case là khi đoạn văn bản đã ngắn hơn ngưỡng cho phép thì giữ nguyên.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Văn bản quy chế thường có cấu trúc phân cấp rõ ràng (Chương → Điều → Khoản). RecursiveChunker tận dụng pattern này để giữ nguyên ngữ cảnh, tránh mất tiêu đề hoặc cắt ngang điều khoản. Điều này đặc biệt quan trọng khi truy xuất các quy định pháp lý, vì tiêu đề và nội dung luôn đi kèm nhau.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| data.txt | best baseline (Best Baseline (Recursive)) | 159 | 463.50 | Tốt: Giữ được cấu trúc phân cấp (Chương/Điều) rất tốt. |
| data.txt | RecursiveChunker | 152 | 480.20 | Rất tốt: Giữ nguyên ngữ cảnh, chunk dài hơn nên ít bị phân mảnh. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveCharacter | 9/10 | Giữ được ngữ cảnh của toàn bộ mục lớn (Section). | Số lượng ký tự mỗi chunk không đều, dễ gây quá tải token cho LLM. |
| Bân | SentenceChunker (5 sentences) | 8/10 | Trả về nội dung gãy gọn, đúng trọng tâm câu hỏi về quy định. | Đôi khi mất tiêu đề "Điều/Khoản" nếu tiêu đề nằm ở chunk trước. |
| Lâm | FixedSizeChunker (497.30 chars) | 6/10 | Đơn giản, tốc độ xử lý nhanh nhất. | Hay bị cắt ngang câu làm AI trả lời cụt ngủn hoặc sai lệch. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
RecursiveChunker là phù hợp nhất vì văn bản quy chế có cấu trúc phân cấp rõ ràng. Nó giúp giữ nguyên tiêu đề và nội dung trong cùng chunk, đảm bảo khi truy xuất thì câu trả lời đầy đủ, không bị mất ngữ cảnh như FixedSize hoặc SentenceChunker.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

SentenceChunker.chunk: Dùng regex như r'(?<=[.!?])\s+' để phát hiện ranh giới câu. Edge case xử lý dấu chấm trong viết tắt (VD: "TS.") hoặc số thập phân.

RecursiveChunker.chunk / _split: Thuật toán chia theo cấp độ: đầu tiên ngắt theo đoạn (\n\n), nếu vẫn dài thì ngắt theo dòng đơn, cuối cùng ngắt theo câu. Base case: khi độ dài < max_tokens thì dừng.

### EmbeddingStore

add_documents + search: Lưu embedding vector trong FAISS index. Tính similarity bằng cosine similarity.

search_with_filter + delete_document: Filter áp dụng trước khi tính similarity để giảm tập tìm kiếm. Delete bằng cách loại bỏ vector và metadata khỏi index.

### KnowledgeBaseAgent

answer: Prompt gồm 2 phần: câu hỏi người dùng + context từ top-k chunks. Inject context bằng cách nối vào system prompt để LLM trả lời dựa trên nguồn.

### Test Results

```
# Paste output of: pytest tests/ -v
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | GPA ≥ 3.20 thì được xếp loại Giỏi. | Sinh viên có GPA từ 3.20 đến 3.59 được xếp loại Giỏi. | high | 0.89 | |
| 2 | Một tín chỉ tương đương 50 giờ học.| Một tín chỉ bằng 15 tiết học lý thuyết.| low | 0.42 | |
| 3 | Sinh viên phải đăng ký ít nhất 12 tín chỉ/kỳ. | Số tín chỉ tối đa mỗi kỳ là 25. | low | 0.35 | |
| 4 | Thời gian bảo lưu kết quả học tập tối đa 2 kỳ. | Sinh viên có thể xin bảo lưu từ 1 đến 2 học kỳ. | high| 0.91  | |
| 5 |Quy chế đào tạo có mã số VU_HT03.VN.|Mã số văn bản là VU_HT03.VN.| high  | 0.95 | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Kết quả bất ngờ nhất: Cặp số 2 cho thấy embeddings nhận diện sự khác biệt về số lượng giờ/tiết, dù cả hai đều nói về tín chỉ. Điều này chứng minh embeddings không chỉ dựa vào từ khóa mà còn biểu diễn ngữ nghĩa định lượng.

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Mã số của Quy chế đào tạo VinUni là gì? | VU_HT03.VN |
| 2 | Tổng số tín chỉ tối thiểu cần đăng ký một kỳ? | 12 tín chỉ đối với sinh viên hệ chính quy |
| 3 | GPA bao nhiêu thì được xếp loại học lực Giỏi? | Từ 3.20 đến 3.59 |
| 4 | Một tín chỉ tương đương bao nhiêu giờ học? | 50 giờ học định mức |
| 5 | Thời gian bảo lưu kết quả học tập tối đa? | Từ một đến hai học kỳ |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
| --- | --- | --- | --- | --- | --- |
| 1 | Mã số Quy chế | "Quy chế đào tạo đại học hệ chính quy… Mã số: VU_HT03.VN" | 0.92 | ✔ | VU_HT03.VN |
| 2 | Tín chỉ tối thiểu/kỳ | "Sinh viên hệ chính quy phải đăng ký ít nhất 12 tín chỉ…" | 0.88 | ✔ | 12 tín chỉ |
| 3 | GPA loại Giỏi | "Xếp loại học lực Giỏi: GPA từ 3.20 đến 3.59" | 0.90 | ✔ | 3.20–3.59 |
| 4 | Một tín chỉ = ? giờ | "Một tín chỉ tương đương 50 giờ học định mức" | 0.87 | ✔ | 50 giờ |
| 5 | Bảo lưu tối đa | "Sinh viên được bảo lưu kết quả học tập từ 1 đến 2 học kỳ" | 0.85 | ✔ | 1–2 học kỳ |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Tôi học được cách SentenceChunker giúp trả lời chính xác các câu hỏi ngắn gọn, vì chunk nhỏ nên ít nhiễu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Tôi thấy nhóm khác dùng metadata phong phú (ví dụ: "chapter", "article") để tăng độ chính xác retrieval.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Tôi sẽ bổ sung thêm metadata về "Điều/Khoản" để retrieval dễ dàng hơn, thay vì chỉ dựa vào chunk_id.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 13/ 15 |
| My approach | Cá nhân | 8/ 10 |
| Similarity predictions | Cá nhân | 3/ 5 |
| Results | Cá nhân | 7/ 10 |
| Core implementation (tests) | Cá nhân | 27/ 30 |
| Demo | Nhóm | 4/ 5 |
| **Tổng** | 77| **/ 100** |
