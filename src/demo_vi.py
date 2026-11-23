"""
Hệ thống Gợi ý Hoạt động Tình nguyện - Demo Tiếng Việt
Sinh viên: Lê Văn Hoài Rin (MSSV: 29211332113) - CS467
"""

import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_pipeline import VolunteerNLPPipeline
from src.recommender import (
    PopularityRecommender,
    ContentBasedRecommender,
    UserBasedCollaborativeFiltering,
    HybridRecommender
)

def chay_demo_tieng_viet():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    df_items = pd.read_csv(os.path.join(base_dir, "data", "items.csv"))
    df_ratings = pd.read_csv(os.path.join(base_dir, "data", "ratings.csv"))

    print("=" * 80)
    print("HỆ THỐNG GỢI Ý HOẠT ĐỘNG TÌNH NGUYỆN VÀ PHỤC VỤ CỘNG ĐỒNG")
    print("Sinh viên: Lê Văn Hoài Rin | MSSV: 29211332113 | Lớp: CS467C (STT 14)")
    print("=" * 80)
    print(f"Dữ liệu nạp: {len(df_items)} hoạt động, {len(df_ratings)} lượt đánh giá.\n")

    nlp = VolunteerNLPPipeline()
    nlp.fit_items(df_items)

    cb_rec = ContentBasedRecommender(nlp)
    cb_rec.fit(df_items)

    cf_rec = UserBasedCollaborativeFiltering()
    cf_rec.fit(df_ratings, df_items)

    hybrid_rec = HybridRecommender(cb_rec, cf_rec, alpha=0.55)

    # 1. Test case 1
    q1 = "weekend beach cleanup and environmental education for kids"
    print(">>> [CA KIỂM THỬ 1] Tìm kiếm theo sở thích ngôn ngữ tự nhiên:")
    print(f"    Truy vấn: '{q1}'")
    recs1 = cb_rec.recommend_for_query(q1, top_n=5)
    for r, it in enumerate(recs1, 1):
        print(f"    {r}. [{it['item_id']}] {it['activity_title']}")
        print(f"       Lĩnh vực: {it['cause_area']} | Điểm tương đồng: {it['score']:.4f}")
        print(f"       Từ khóa khớp: {', '.join(it['matching_keywords'])}")

    # 2. Test case 2
    u2 = "USR_02"
    print(f"\n>>> [CA KIỂM THỬ 2] Gợi ý cho tình nguyện viên quay lại ({u2} từng chấm 5 sao dạy học):")
    recs2 = hybrid_rec.recommend(u2, df_ratings, df_items, top_n=5)
    for r, it in enumerate(recs2, 1):
        print(f"    {r}. [{it['item_id']}] {it['activity_title']}")
        print(f"       Lĩnh vực: {it['cause_area']} | Điểm Hybrid: {it['score']:.4f} ({it['method']})")
        if it.get("matching_keywords"):
            print(f"       Trùng khớp hồ sơ: {', '.join(it['matching_keywords'])}")

    # 3. Test case 3
    print("\n>>> [CA KIỂM THỬ 3] Khởi đầu lạnh (Cold-Start ITEM_038 chưa có lượt đánh giá nào):")
    q3 = "rural community donation drive collecting children books and toys"
    print(f"    Truy vấn: '{q3}'")
    recs3 = cb_rec.recommend_for_query(q3, top_n=5)
    for r, it in enumerate(recs3, 1):
        dau_muc = " <--- [ĐÃ KHỚP HOẠT ĐỘNG MỚI COLD-START!]" if it["item_id"] == "ITEM_038" else ""
        print(f"    {r}. [{it['item_id']}] {it['activity_title']}{dau_muc}")
        print(f"       Lĩnh vực: {it['cause_area']} | Điểm tương đồng: {it['score']:.4f}")
        print(f"       Từ khóa khớp: {', '.join(it['matching_keywords'])}")

    print("\n" + "=" * 80)
    print("Xác nhận hoàn tất cả 3 ca kiểm thử.")
    print("=" * 80)

if __name__ == "__main__":
    chay_demo_tieng_viet()
