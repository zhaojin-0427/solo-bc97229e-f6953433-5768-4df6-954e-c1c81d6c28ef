"""对运行中的服务做端到端冒烟验证（默认 http://127.0.0.1:8000）。

用法：python3 scripts/smoke_test.py [base_url] [sample_data_dir]
覆盖：导入 -> 分组 -> 指定分区 -> 继承 -> 冲突拒绝 -> 幂等 -> 下载 -> 并发一致性。
"""
from __future__ import annotations

import csv
import io
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SAMPLES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data"
)

client = httpx.Client(base_url=BASE, timeout=30)
failures = []


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        failures.append(name)


def upload(path):
    with open(path, "rb") as f:
        return client.post("/imports", files={"file": (os.path.basename(path), f, "text/csv")})


def assign(image_id, partition):
    return client.post("/partitions/assign", json={"image_id": image_id, "partition": partition})


def get_image(image_id):
    return client.get(f"/images/{image_id}")


# 0. 健康检查
check("health", client.get("/health").json()["status"] == "ok")

# 1. 导入 batch1：IMG001-004 沿关联链成一组，IMG005 独立成组
r = upload(f"{SAMPLES}/batch1.csv")
check("导入 batch1", r.status_code == 200 and r.json()["inserted"] == 5, r.text[:200])
g1 = get_image("IMG001").json()["group_id"]
check(
    "关联链归为一组",
    len({get_image(i).json()["group_id"] for i in ("IMG001", "IMG002", "IMG003", "IMG004")}) == 1
    and get_image("IMG005").json()["group_id"] != g1,
)

# 2. 同内容重传：幂等
r = upload(f"{SAMPLES}/batch1.csv")
check("同内容重传不新增", r.status_code == 200 and r.json()["inserted"] == 0 and r.json()["skipped_duplicates"] == 5)

# 3. 异内容重传：整批拒绝
r = upload(f"{SAMPLES}/batch5_changed_content.csv")
check("异内容整批拒绝(409)", r.status_code == 409 and r.json()["detail"]["error"] == "content_conflict")

# 4. 按任一图片 ID 指定整组到 train
r = assign("IMG003", "train")
check("按任一图片ID指定整组", r.status_code == 200 and r.json()["images_in_group"] == 4)
check("整组生效", get_image("IMG001").json()["partition"] == "train")

# 5. 已指定不可改动
r = assign("IMG001", "val")
check("已指定不可改动(409)", r.status_code == 409 and r.json()["detail"]["error"] == "already_assigned")
r = assign("IMG001", "train")
check("同分区重复指定幂等", r.status_code == 200 and r.json()["changed"] is False)

# 6. 导入 batch2：IMG006 经 LOT-2 关联 train 组 -> 继承 train；IMG007 新组
r = upload(f"{SAMPLES}/batch2.csv")
check("导入 batch2", r.status_code == 200 and r.json()["inserted"] == 2, r.text[:200])
check("新图继承关联组分区", get_image("IMG006").json()["partition"] == "train")
check("无关联新图未分区", get_image("IMG007").json()["partition"] is None)

# 7. IMG005 组指定为 val
check("指定 val", assign("IMG005", "val").status_code == 200)

# 8. 导入 batch3：IMG008 连接 val 组(WP-C)与未分区组(IMG007/LOT-4) -> 合并并继承 val
r = upload(f"{SAMPLES}/batch3_link.csv")
check("导入 batch3 连接 val 与未分区组", r.status_code == 200 and r.json()["components"][0]["partition"] == "val", r.text[:200])
check("被合并组继承 val", get_image("IMG007").json()["partition"] == "val")

# 9. 导入 batch4：IMG009 同时关联 val(WP-C) 与 train(LOT-2) -> 整批拒绝
r = upload(f"{SAMPLES}/batch4_conflict.csv")
detail = r.json().get("detail", {})
check(
    "连通两个不同分区整批拒绝(409)",
    r.status_code == 409 and detail.get("error") == "partition_conflict"
    and set(detail["component"]["partitions"]) == {"train", "val"},
    r.text[:300],
)
check("冲突导入未留脏数据", get_image("IMG009").status_code == 404)

# 10. 未分区列表应为空（IMG007 已随组合并进 val）
un = client.get("/images/unpartitioned").json()
check("未分区列表", un["count"] == 0, f"count={un['count']}")

# 11. 分区 CSV 下载：train 5 张、val 3 张
train_rows = list(csv.DictReader(io.StringIO(client.get("/downloads/train").text)))
val_rows = list(csv.DictReader(io.StringIO(client.get("/downloads/val").text)))
check("train CSV 完整", sorted(r["image_id"] for r in train_rows) == ["IMG001", "IMG002", "IMG003", "IMG004", "IMG006"])
check("val CSV 完整", sorted(r["image_id"] for r in val_rows) == ["IMG005", "IMG007", "IMG008"])

# 12. HTTP 级并发：10 线程混合导入/指定分区/下载，结果必须一致
def worker(i):
    c = httpx.Client(base_url=BASE, timeout=30)
    try:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["image_id", "workpiece_no", "batch_no"])
        # 每个线程的新图都关联到 train 组（LOT-1），应全部继承 train
        w.writerow([f"CONC-{i:02d}", f"WP-CONC-{i}", "LOT-1"])
        r1 = c.post("/imports", files={"file": ("c.csv", buf.getvalue().encode(), "text/csv")})
        r2 = c.post("/partitions/assign", json={"image_id": "IMG002", "partition": "train"})
        r3 = c.get("/downloads/train")
        return r1.status_code, r2.status_code, r3.status_code
    finally:
        c.close()

with ThreadPoolExecutor(max_workers=10) as pool:
    outcomes = list(pool.map(worker, range(10)))
check("并发导入全部成功", all(o[0] == 200 for o in outcomes), str(outcomes))
check("并发指定分区均合法", all(o[1] in (200, 409) for o in outcomes))
check("并发下载均成功", all(o[2] == 200 for o in outcomes))
conc_ok = all(
    get_image(f"CONC-{i:02d}").json()["partition"] == "train" for i in range(10)
)
check("并发新图全部继承 train", conc_ok)
train_rows = list(csv.DictReader(io.StringIO(client.get("/downloads/train").text)))
check("并发后 train CSV 完整一致", len(train_rows) == 15, f"rows={len(train_rows)}")

stats = client.get("/stats").json()
print("最终状态:", stats)
# 5(batch1) + 2(batch2) + 1(batch3) + 10(并发) = 18；冲突与异内容导入被拒绝不计入
check("最终图片数=18", stats["images"] == 18)

# ===========================================================================
# 13. 模型评估：预测 / 真值上传、3x3 矩阵、修订与冲突、分区联动
# ===========================================================================

def upload_pred(model, path):
    with open(path, "rb") as f:
        return client.post(f"/models/{model}/predictions",
                           files={"file": (os.path.basename(path), f, "text/csv")})


def upload_truth(path):
    with open(path, "rb") as f:
        return client.post("/ground-truths",
                           files={"file": (os.path.basename(path), f, "text/csv")})


def mat(model, partition):
    return client.get(f"/models/{model}/matrices/{partition}").json()


# 13.1 未分区状态不计数（此刻全部图片已在 train/val，先在新图上验证未分区）
buf = io.StringIO()
w = csv.writer(buf, lineterminator="\n")
w.writerow(["image_id", "workpiece_no", "batch_no"])
w.writerow(["EVAL-1", "WP-EVAL", "LOT-EVAL"])  # 独立未分区组
r = client.post("/imports", files={"file": ("e.csv", buf.getvalue().encode(), "text/csv")})
check("评估演示新图导入", r.status_code == 200)
r = upload_pred("model-a", f"{SAMPLES}/predictions_model_a.csv")
check("预测上传成功", r.status_code == 200, r.text[:200])
check("同标签预测重传复用",
      upload_pred("model-a", f"{SAMPLES}/predictions_model_a.csv").json()["reused"] == 8)
# 未分区的 EVAL-1 不参与；先验证非法/冲突路径
r = upload_pred("model-a", f"{SAMPLES}/predictions_conflict.csv")
check("预测冲突整批拒绝(409)", r.status_code == 409
      and r.json()["detail"]["error"] == "prediction_conflict")
r = client.post("/models/model-a/predictions",
                files={"file": ("b.csv", b"image_id,label\nGHOST,normal\n", "text/csv")})
check("未登记图片预测整批拒绝(400)", r.status_code == 400
      and r.json()["detail"]["error"] == "unregistered_image")

# 13.2 真值上传（rev1）
r = upload_truth(f"{SAMPLES}/ground_truth_rev1.csv")
check("真值 rev1 接受", r.status_code == 200 and r.json()["accepted"] == 8, r.text[:200])
r = upload_truth(f"{SAMPLES}/ground_truth_same_revision_conflict.csv")
check("同修订异标签整批拒绝(409)", r.status_code == 409
      and r.json()["detail"]["error"] == "revision_conflict")

# 13.3 分区矩阵（此时 18 张老图均已分区；8 张评估图分布：train 5 张、val 3 张）
mt = mat("model-a", "train")
mv = mat("model-a", "val")
# train: IMG001 n/n, IMG002 s/n, IMG003 s/s, IMG004 d/s, IMG006 n/n
check("train 矩阵计数正确",
      mt["matrix"]["normal"]["normal"] == 2
      and mt["matrix"]["scratch"]["normal"] == 1
      and mt["matrix"]["scratch"]["scratch"] == 1
      and mt["matrix"]["dent"]["scratch"] == 1
      and mt["support"] == 5, str(mt["matrix"]))
# val: IMG005 d/d, IMG007 d/d, IMG008 n/d
check("val 矩阵计数正确",
      mv["matrix"]["dent"]["dent"] == 2
      and mv["matrix"]["normal"]["dent"] == 1
      and mv["support"] == 3, str(mv["matrix"]))

# 13.4 单次查询返回同一状态的完整双分区矩阵
both = client.get("/models/model-a/matrices").json()
check("双分区矩阵同状态返回",
      both["partitions"]["train"]["state_version"] == both["state_version"]
      and both["partitions"]["val"]["state_version"] == both["state_version"])

# 13.5 真值更高修订更正：旧贡献扣除、新贡献建立
r = upload_truth(f"{SAMPLES}/ground_truth_rev2_corrections.csv")
check("真值 rev2 更正接受", r.status_code == 200 and r.json()["accepted"] == 3)
mt2 = mat("model-a", "train")
# IMG002: scratch/normal -> dent/normal；IMG004: dent/scratch -> scratch/scratch
check("更正后 train 旧单元扣除",
      mt2["matrix"]["scratch"]["normal"] == 0 and mt2["matrix"]["dent"]["scratch"] == 0)
check("更正后 train 新单元建立且总数不变",
      mt2["matrix"]["dent"]["normal"] == 1
      and mt2["matrix"]["scratch"]["scratch"] == 2
      and mt2["support"] == 5)
mv2 = mat("model-a", "val")
# IMG008: normal/dent -> dent/dent
check("更正后 val 单元迁移",
      mv2["matrix"]["normal"]["dent"] == 0 and mv2["matrix"]["dent"]["dent"] == 3
      and mv2["support"] == 3)
# 低修订重传被忽略
low = io.StringIO()
cw = csv.writer(low, lineterminator="\n")
cw.writerow(["image_id", "revision", "label"])
cw.writerow(["IMG002", "1", "normal"])
r = client.post("/ground-truths",
                files={"file": ("low.csv", low.getvalue().encode(), "text/csv")})
check("低修订真值忽略", r.json()["ignored_lower_revision"] == 1 and r.json()["accepted"] == 0)

# 13.6 贡献明细持久化，且与矩阵聚合一致
contribs = client.get("/contributions", params={"model_id": "model-a"}).json()
check("逐图片贡献数=矩阵 support 之和",
      contribs["count"] == mt2["support"] + mv2["support"] == 8)

# 13.7 未分区图片补齐分区后自动计数（EVAL-1 先有预测+真值，再分区）
client.post("/models/model-a/predictions",
            files={"file": ("e.csv", b"image_id,label\nEVAL-1,scratch\n", "text/csv")})
client.post("/ground-truths",
            files={"file": ("e.csv", b"image_id,revision,label\nEVAL-1,1,dent\n", "text/csv")})
before = mat("model-a", "train")["support"]
client.post("/partitions/assign", json={"image_id": "EVAL-1", "partition": "train"})
after = mat("model-a", "train")
check("整组分区后增量计数",
      after["support"] == before + 1 and after["matrix"]["dent"]["scratch"] == 1)

# 13.8 并发评估上传：矩阵始终等于逐图片贡献聚合
def eval_worker(i):
    c = httpx.Client(base_url=BASE, timeout=30)
    try:
        img = f"EVALC-{i:02d}"
        buf2 = io.StringIO()
        w2 = csv.writer(buf2, lineterminator="\n")
        w2.writerow(["image_id", "workpiece_no", "batch_no"])
        w2.writerow([img, f"WP-EC-{i}", f"LOT-EC-{i}"])
        r1 = c.post("/imports", files={"file": ("c.csv", buf2.getvalue().encode(), "text/csv")})
        r2 = c.post("/models/model-a/predictions",
                    files={"file": ("p.csv", f"image_id,label\n{img},scratch\n".encode(), "text/csv")})
        r3 = c.post("/ground-truths",
                    files={"file": ("t.csv", f"image_id,revision,label\n{img},1,dent\n".encode(), "text/csv")})
        r4 = c.post("/partitions/assign",
                    json={"image_id": img, "partition": "val" if i % 2 else "train"})
        return [r.status_code for r in (r1, r2, r3, r4)]
    finally:
        c.close()


with ThreadPoolExecutor(max_workers=10) as pool:
    codes = list(pool.map(eval_worker, range(10)))
check("并发评估上传全部成功", all(all(x < 400 for x in cs) for cs in codes), str(codes))
final = client.get("/models/model-a/matrices").json()
total_support = sum(final["partitions"][p]["support"] for p in ("train", "val"))
contrib_count = client.get("/contributions", params={"model_id": "model-a"}).json()["count"]
check("并发后矩阵与贡献聚合一致", total_support == contrib_count,
      f"support={total_support} contrib={contrib_count}")
print("最终评估矩阵:", final)

print()
if failures:
    print(f"共 {len(failures)} 项失败: {failures}")
    sys.exit(1)
print("全部冒烟检查通过 ✔")
