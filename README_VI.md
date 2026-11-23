# Hệ Thống Gợi Ý Hoạt Động Tình Nguyện & Phục Vụ Cộng Đồng (CS467)

> **ĐẠI HỌC DUY TÂN**  
> **Môn học:** CS467 - Trí Tuệ Nhân Tạo Nâng Cao (2026-2027)  
> **Bài tập 15:** Mini-Project Cá nhân: NLP & Recommendation Systems  

---

## 📌 Thông Tin Dự Án
| Mục | Chi tiết |
| :--- | :--- |
| **Họ và Tên** | **Lê Văn Hoài Rin** |
| **MSSV** | **29211332113** |
| **Lớp / Section** | **CS467C (STT 14)** |
| **Giảng viên** | **Puri Vikram** |
| **Mã đề tài** | **MP-C14** |
| **Tên đề tài** | **Volunteer and Community Service Recommender** |
| **Lĩnh vực** | **NGO and Student Civic Engagement** |
| **Nhánh demo** | `vietnamese-demo` |

---

## 🚀 Hướng Dẫn Chạy Demo Tiếng Việt

```bash
# Chuyển sang nhánh demo tiếng Việt
git checkout vietnamese-demo

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy bản demo Tiếng Việt hiển thị 3 ca kiểm thử
python src/demo_vi.py

# Chạy bản CLI đầy đủ
python src/main.py
```

---

## 💡 Cốt Lõi Kiến Trúc 4 Thuật Toán

1. **Popularity Baseline (Cơ sở Phổ biến):**
   - Công thức Bayesian Average (IMDb): xếp hạng theo điểm trung bình có trọng số lượt đánh giá, lọc bỏ các hoạt động tình nguyện viên đã tham gia.
2. **NLP Content-Based Recommender (Dựa trên Nội dung TF-IDF):**
   - Làm sạch văn bản, tách từ, loại bỏ stop-words.
   - Vector hóa TF-IDF (Unigram + Bigram, max 1.200 đặc trưng).
   - Xây dựng hồ sơ sở thích tình nguyện viên (User Interest Profile) bằng tổng trọng số vector các bài đã đánh giá $\ge 3.5\star$.
   - Tính độ tương đồng Cosine giữa hồ sơ/truy vấn với các hoạt động; trích xuất từ khóa giải thích (Explainable AI).
3. **User-Based Collaborative Filtering (Lọc Cộng tác Người dùng):**
   - Ma trận tương tác thưa (53.5% missing).
   - Centered-Cosine / Pearson correlation trừ trung bình người dùng, tìm $k$-láng giềng gần nhất ($k=5$) để dự đoán điểm.
4. **Weighted Hybrid Recommender (Mô hình Kết hợp Có Trọng số):**
   - Chuẩn hóa Min-Max điểm số Content và CF về cùng thang đo $[0, 1]$.
   - Điểm tổng: $\text{Score} = \alpha \cdot \text{Score}_{\text{Content}} + (1 - \alpha) \cdot \text{Score}_{\text{CF}}$ (mặc định $\alpha = 0.55$).
   - Cơ chế Fallback tự động: gặp item hoặc user Cold-Start (0 đánh giá) sẽ tự động chuyển sang thuần Content-Based.
