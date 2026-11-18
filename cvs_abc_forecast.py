"""
便利店物流中心：ABC 分類 + 鮮食需求預測 範例
=================================================
執行方式：python cvs_abc_forecast.py
需要套件：pandas numpy scikit-learn matplotlib
  （安裝：pip install pandas numpy scikit-learn matplotlib）

本範例使用「模擬資料」，實務上把第 1 步換成讀取公司的
出貨明細（例如 pd.read_csv("出貨明細.csv") 或用 SQL 撈出）即可。

輸出：
  abc_result.csv        每個 SKU 的 ABC 等級
  forecast_result.csv   鮮食每日預測 vs 實際
  abc_pareto.png        ABC 柏拉圖
  forecast.png          鮮食需求預測圖
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 中文字型（Windows 可改成 "Microsoft JhengHei"）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)

# =========================================================
# 第 1 步：產生模擬資料 — 物流中心一年的 SKU 出貨明細
# =========================================================
categories = {"鮮食": 20, "飲料": 25, "日用品": 35}   # 各品類 SKU 數
skus = []
for cat, n in categories.items():
    for i in range(1, n + 1):
        skus.append({
            "sku": f"{cat}-{i:03d}",
            "品類": cat,
            # 單價：鮮食 40~90、飲料 20~60、日用品 30~200
            "單價": {"鮮食": rng.integers(40, 90),
                     "飲料": rng.integers(20, 60),
                     "日用品": rng.integers(30, 200)}[cat],
            # 年出貨量用長尾分布，模擬「少數品項賣很多」
            "年出貨量": int(rng.pareto(1.2) * 3000 + 200),
        })
sku_df = pd.DataFrame(skus)

# =========================================================
# 第 2 步：ABC 分類（依出貨金額）
#   A：累積金額前 80%   → 重點管理、靠近出貨口儲位
#   B：80%~95%          → 一般管理
#   C：95%~100%         → 檢討是否下架、減少備貨
# =========================================================
sku_df["出貨金額"] = sku_df["單價"] * sku_df["年出貨量"]
abc = sku_df.sort_values("出貨金額", ascending=False).reset_index(drop=True)
abc["累積占比"] = abc["出貨金額"].cumsum() / abc["出貨金額"].sum()
abc["ABC"] = np.select(
    [abc["累積占比"] <= 0.80, abc["累積占比"] <= 0.95],
    ["A", "B"],
    default="C",
)

summary = abc.groupby("ABC").agg(
    SKU數=("sku", "count"),
    出貨金額=("出貨金額", "sum"),
)
summary["SKU占比"] = (summary["SKU數"] / summary["SKU數"].sum()).map("{:.0%}".format)
summary["金額占比"] = (summary["出貨金額"] / summary["出貨金額"].sum()).map("{:.0%}".format)

print("=" * 50)
print("【ABC 分類結果】")
print(summary)
print("\n【各品類的 A 類品項數】")
print(abc[abc["ABC"] == "A"].groupby("品類")["sku"].count())
abc.to_csv("abc_result.csv", index=False, encoding="utf-8-sig")

# 柏拉圖
fig, ax1 = plt.subplots(figsize=(11, 5))
colors = abc["ABC"].map({"A": "#2a6fdb", "B": "#f2a93b", "C": "#b0b0b0"})
ax1.bar(range(len(abc)), abc["出貨金額"] / 1e6, color=colors)
ax1.set_ylabel("出貨金額（百萬元）")
ax1.set_xlabel("SKU（依出貨金額排序）")
ax2 = ax1.twinx()
ax2.plot(range(len(abc)), abc["累積占比"] * 100, color="#333", lw=2)
ax2.axhline(80, ls="--", color="#2a6fdb", lw=1)
ax2.axhline(95, ls="--", color="#f2a93b", lw=1)
ax2.set_ylabel("累積占比（%）")
ax1.set_title("便利店物流中心 SKU ABC 柏拉圖（藍=A、橘=B、灰=C）")
plt.tight_layout()
plt.savefig("abc_pareto.png", dpi=130)
plt.close()

# =========================================================
# 第 3 步：產生模擬資料 — 某門市「鮮食」每日銷量（一年）
#   影響因素：星期幾、氣溫、假日、隨機波動
# =========================================================
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
df = pd.DataFrame({"日期": dates})
df["星期"] = df["日期"].dt.dayofweek            # 0=週一 … 6=週日
df["月"] = df["日期"].dt.month
# 氣溫：夏天高、冬天低
df["氣溫"] = 23 + 7 * np.sin((df["日期"].dt.dayofyear - 110) / 365 * 2 * np.pi) \
             + rng.normal(0, 2, len(df))
# 簡化的國定假日
holidays = pd.to_datetime(["2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30",
                           "2025-02-28", "2025-04-04", "2025-05-01", "2025-05-31",
                           "2025-10-06", "2025-10-10"])
df["假日"] = df["日期"].isin(holidays).astype(int)

# 「真實」需求公式：辦公商圈店 → 平日賣得好、週末和假日掉；天熱便當少、涼麵多（簡化為整體略降）
base = 180
weekday_effect = np.where(df["星期"] < 5, 1.0, 0.70)
holiday_effect = np.where(df["假日"] == 1, 0.65, 1.0)
temp_effect = 1 - (df["氣溫"] - 23) * 0.008
df["銷量"] = (base * weekday_effect * holiday_effect * temp_effect
             + rng.normal(0, 10, len(df))).round().astype(int)

# =========================================================
# 第 4 步：特徵工程 — 預測時「當下已知」的資訊
#   注意：只能用過去的銷量（lag），不能偷看當天答案
# =========================================================
df["上週同日銷量"] = df["銷量"].shift(7)
df["近7日平均"] = df["銷量"].shift(1).rolling(7).mean()
df = df.dropna().reset_index(drop=True)

features = ["星期", "月", "氣溫", "假日", "上週同日銷量", "近7日平均"]
train = df[df["日期"] < "2025-11-01"]
test = df[df["日期"] >= "2025-11-01"].copy()      # 用最後兩個月驗證

# =========================================================
# 第 5 步：訓練模型 vs 傳統做法比較
#   傳統做法：店長憑經驗，直接照「上週同一天」的量訂貨
# =========================================================
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(train[features], train["銷量"])
test["模型預測"] = model.predict(test[features]).round()
test["傳統做法"] = test["上週同日銷量"]

mae_model = mean_absolute_error(test["銷量"], test["模型預測"])
mae_old = mean_absolute_error(test["銷量"], test["傳統做法"])

# =========================================================
# 第 6 步：翻成「錢」— 報廢與缺貨成本
#   訂貨量 = 預測量；多訂的報廢（損失成本 35 元/個）
#   少訂的缺貨（損失毛利 20 元/個）
# =========================================================
COST, MARGIN = 35, 20

def money_loss(order, actual):
    waste = np.clip(order - actual, 0, None)
    short = np.clip(actual - order, 0, None)
    return waste.sum(), short.sum(), waste.sum() * COST + short.sum() * MARGIN

w_old, s_old, loss_old = money_loss(test["傳統做法"], test["銷量"])
w_new, s_new, loss_new = money_loss(test["模型預測"], test["銷量"])
days = len(test)

print("\n" + "=" * 50)
print(f"【鮮食需求預測】驗證期間：{days} 天")
print(f"平均誤差（MAE） 傳統做法：{mae_old:.1f} 個/天　模型：{mae_model:.1f} 個/天")
print(f"報廢量          傳統做法：{w_old:.0f} 個　　 模型：{w_new:.0f} 個")
print(f"缺貨量          傳統做法：{s_old:.0f} 個　　 模型：{s_new:.0f} 個")
print(f"損失金額        傳統做法：{loss_old:,.0f} 元　模型：{loss_new:,.0f} 元")
saving_month = (loss_old - loss_new) / days * 30
print(f"→ 單店每月約可減少損失 {saving_month:,.0f} 元，"
      f"若推到 300 家門市 ≈ 每月 {saving_month * 300 / 1e4:,.0f} 萬元")

imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n【影響鮮食銷量的因素（重要度）】")
print(imp.map("{:.0%}".format))

test[["日期", "銷量", "傳統做法", "模型預測"]].to_csv(
    "forecast_result.csv", index=False, encoding="utf-8-sig")

# 預測圖
plt.figure(figsize=(11, 4.5))
plt.plot(test["日期"], test["銷量"], label="實際銷量", color="#333", lw=2)
plt.plot(test["日期"], test["傳統做法"], label="傳統做法（上週同日）", color="#f2a93b", ls="--")
plt.plot(test["日期"], test["模型預測"], label="模型預測", color="#2a6fdb")
plt.title("門市鮮食每日需求：實際 vs 預測（2025/11–12）")
plt.ylabel("銷量（個）")
plt.legend()
plt.tight_layout()
plt.savefig("forecast.png", dpi=130)
plt.close()

print("\n已輸出：abc_result.csv、forecast_result.csv、abc_pareto.png、forecast.png")
