"""评估 API 端到端测试：上传校验、修订机制、3x3 矩阵、分区联动、并发与重启。"""
from __future__ import annotations

import csv
import io
import threading

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _csv(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode()


def import_images(client, rows):
    payload = _csv(["image_id", "workpiece_no", "batch_no"], rows)
    return client.post("/imports", files={"file": ("d.csv", payload, "text/csv")})


def upload_pred(client, model, rows):
    payload = _csv(["image_id", "label"], rows)
    return client.post(
        f"/models/{model}/predictions",
        files={"file": ("p.csv", payload, "text/csv")},
    )


def upload_truth(client, rows):
    payload = _csv(["image_id", "revision", "label"], rows)
    return client.post("/ground-truths", files={"file": ("t.csv", payload, "text/csv")})


def matrix(client, model, partition):
    return client.get(f"/models/{model}/matrices/{partition}").json()


@pytest.fixture()
def seeded(client):
    # 组 A：IMG1..3（train 候选）；IMG4 独立（val 候选）
    assert import_images(client, [
        ("IMG1", "WP-A", "LOT-1"),
        ("IMG2", "WP-A", "LOT-1"),
        ("IMG3", "WP-B", "LOT-2"),
        ("IMG4", "WP-C", "LOT-3"),
    ]).status_code == 200
    return client


# ---------------------------------------------------------------------------
# 文件级校验
# ---------------------------------------------------------------------------

def test_prediction_csv_validation(seeded):
    # 非法标签 + 空 ID，一次返回全部行错误，整批拒绝
    payload = _csv(["image_id", "label"], [
        ("IMG1", "broken"),
        ("", "normal"),
        ("IMG2", "NORMAL"),  # 大小写不敏感，合法
    ])
    r = seeded.post("/models/m1/predictions", files={"file": ("p.csv", payload, "text/csv")})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "invalid_csv"
    assert len(r.json()["detail"]["errors"]) == 2
    # 整批拒绝：合法行也不写入
    assert seeded.get("/images/IMG2/evaluation").json()["predictions"] == []


def test_prediction_duplicate_ids_in_batch_rejected(seeded):
    payload = _csv(["image_id", "label"], [("IMG1", "normal"), ("IMG1", "dent")])
    r = seeded.post("/models/m1/predictions", files={"file": ("p.csv", payload, "text/csv")})
    assert r.status_code == 400
    assert any(e["problem"] == "duplicate_id_conflict" for e in r.json()["detail"]["errors"])


def test_prediction_duplicate_ids_same_label_rejected(seeded):
    # 批内 ID 必须唯一，即使标签相同也整批拒绝
    payload = _csv(["image_id", "label"], [("IMG1", "normal"), ("IMG1", "normal")])
    r = seeded.post("/models/m1/predictions", files={"file": ("p.csv", payload, "text/csv")})
    assert r.status_code == 400


def test_truth_revision_must_be_positive_integer(seeded):
    payload = _csv(["image_id", "revision", "label"], [
        ("IMG1", "0", "normal"),
        ("IMG2", "1.5", "dent"),
        ("IMG3", "x", "scratch"),
        ("IMG4", "-2", "dent"),
    ])
    r = seeded.post("/ground-truths", files={"file": ("t.csv", payload, "text/csv")})
    assert r.status_code == 400
    assert len(r.json()["detail"]["errors"]) == 4


def test_truth_duplicate_ids_in_batch_rejected(seeded):
    payload = _csv(["image_id", "revision", "label"], [
        ("IMG1", "1", "normal"), ("IMG1", "2", "dent"),
    ])
    r = seeded.post("/ground-truths", files={"file": ("t.csv", payload, "text/csv")})
    assert r.status_code == 400


def test_chinese_headers_accepted(seeded):
    payload = _csv(["图片ID", "预测标签"], [("IMG1", "normal")])
    r = seeded.post("/models/m1/predictions", files={"file": ("p.csv", payload, "text/csv")})
    assert r.status_code == 200 and r.json()["inserted"] == 1
    payload = _csv(["图片ID", "修订号", "真值标签"], [("IMG1", "1", "normal")])
    r = seeded.post("/ground-truths", files={"file": ("t.csv", payload, "text/csv")})
    assert r.status_code == 200 and r.json()["accepted"] == 1


def test_unregistered_images_reject_whole_batch(seeded):
    r = upload_pred(seeded, "m1", [("GHOST", "normal"), ("IMG1", "dent")])
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "unregistered_image"
    assert r.json()["detail"]["image_ids"] == ["GHOST"]
    r = upload_truth(seeded, [("GHOST", 1, "normal")])
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 预测语义
# ---------------------------------------------------------------------------

def test_prediction_conflict_rejects_whole_batch(seeded):
    assert upload_pred(seeded, "m1", [("IMG1", "normal"), ("IMG2", "dent")]).status_code == 200
    r = upload_pred(seeded, "m1", [("IMG1", "dent"), ("IMG3", "scratch")])
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["error"] == "prediction_conflict" and d["image_id"] == "IMG1"
    # 整批拒绝：IMG3 未写入，IMG1 保持原标签
    assert seeded.get("/images/IMG3/evaluation").json()["predictions"] == []
    assert seeded.get("/images/IMG1/evaluation").json()["predictions"][0]["label"] == "normal"


def test_same_prediction_reupload_is_idempotent(seeded):
    r1 = upload_pred(seeded, "m1", [("IMG1", "normal")])
    v1 = r1.json()["state_version"]
    r2 = upload_pred(seeded, "m1", [("IMG1", "normal")])
    body = r2.json()
    assert body["inserted"] == 0 and body["reused"] == 1 and body["changed"] is False
    assert body["state_version"] == v1  # 纯幂等不推进版本


def test_models_are_independent(seeded):
    upload_pred(seeded, "m1", [("IMG1", "normal")])
    upload_pred(seeded, "m2", [("IMG1", "dent")])
    ev = seeded.get("/images/IMG1/evaluation").json()
    assert {p["model_id"]: p["label"] for p in ev["predictions"]} == {
        "m1": "normal", "m2": "dent",
    }
    models = seeded.get("/models").json()["models"]
    assert {m["model_id"] for m in models} == {"m1", "m2"}


# ---------------------------------------------------------------------------
# 真值修订机制
# ---------------------------------------------------------------------------

def test_truth_revision_rules(seeded):
    r = upload_truth(seeded, [("IMG1", 3, "dent")])
    assert r.status_code == 200 and r.json()["accepted"] == 1

    r = upload_truth(seeded, [("IMG1", 2, "normal")])  # 低修订忽略
    assert r.json()["ignored_lower_revision"] == 1 and r.json()["accepted"] == 0
    assert seeded.get("/images/IMG1/evaluation").json()["ground_truth"] == {
        "revision": 3, "label": "dent",
    }

    r = upload_truth(seeded, [("IMG1", 3, "dent")])  # 同修订同标签幂等
    assert r.status_code == 200 and r.json()["ignored_same_revision"] == 1

    r = upload_truth(seeded, [("IMG1", 3, "scratch")])  # 同修订异标签
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "revision_conflict"
    assert seeded.get("/images/IMG1/evaluation").json()["ground_truth"]["label"] == "dent"


def test_truth_batch_mix_conflict_is_atomic(seeded):
    upload_truth(seeded, [("IMG1", 5, "normal"), ("IMG2", 1, "dent")])
    # 同修订冲突 + 一条本可接受的高修订：整批拒绝
    r = upload_truth(seeded, [("IMG1", 5, "dent"), ("IMG2", 9, "scratch")])
    assert r.status_code == 409
    ev2 = seeded.get("/images/IMG2/evaluation").json()["ground_truth"]
    assert ev2 == {"revision": 1, "label": "dent"}  # 高修订未生效


# ---------------------------------------------------------------------------
# 矩阵：计数规则、分区联动、更正扣旧贡献
# ---------------------------------------------------------------------------

def test_unpartitioned_not_counted(seeded):
    upload_pred(seeded, "m1", [("IMG1", "normal"), ("IMG2", "dent")])
    upload_truth(seeded, [("IMG1", 1, "normal"), ("IMG2", 1, "normal")])
    m = matrix(seeded, "m1", "train")
    assert m["support"] == 0 and m["accuracy"] is None
    assert seeded.get("/contributions").json()["count"] == 0


def test_full_matrix_lifecycle(seeded):
    upload_pred(seeded, "m1", [
        ("IMG1", "normal"), ("IMG2", "dent"), ("IMG3", "scratch"),
    ])
    upload_truth(seeded, [
        ("IMG1", 1, "normal"), ("IMG2", 1, "normal"), ("IMG3", 1, "scratch"),
    ])
    # 整组 A 指定 train：IMG1 正确、IMG2 真值normal预测dent、IMG3 正确
    r = seeded.post("/partitions/assign", json={"image_id": "IMG1", "partition": "train"})
    assert r.status_code == 200
    m = matrix(seeded, "m1", "train")
    assert m["matrix"]["normal"]["normal"] == 1
    assert m["matrix"]["normal"]["dent"] == 1
    assert m["matrix"]["scratch"]["scratch"] == 1
    assert m["support"] == 3 and m["correct"] == 2
    assert abs(m["accuracy"] - 2 / 3) < 1e-9

    # 缺预测的图片后续补齐：只增量加一个单元
    upload_pred(seeded, "m1", [("IMG4", "dent")])
    upload_truth(seeded, [("IMG4", 1, "dent")])
    assert matrix(seeded, "m1", "val")["support"] == 0  # IMG4 尚未分区
    seeded.post("/partitions/assign", json={"image_id": "IMG4", "partition": "val"})
    mv = matrix(seeded, "m1", "val")
    assert mv["matrix"]["dent"]["dent"] == 1 and mv["support"] == 1

    # 真值更正：IMG1 normal->dent（rev2），旧 normal/normal -1，新 dent/normal +1
    upload_truth(seeded, [("IMG1", 2, "dent")])
    mt = matrix(seeded, "m1", "train")
    assert mt["matrix"]["normal"]["normal"] == 0
    assert mt["matrix"]["dent"]["normal"] == 1
    assert mt["support"] == 3  # 总数不变，只移动单元

    # 贡献明细与矩阵一致
    contribs = seeded.get("/contributions", params={"model_id": "m1"}).json()["contributions"]
    assert len(contribs) == 4
    c1 = next(c for c in contribs if c["image_id"] == "IMG1")
    assert c1 == {
        "model_id": "m1", "image_id": "IMG1", "partition": "train",
        "truth_label": "dent", "pred_label": "normal",
    }


def test_single_query_returns_both_partitions_same_state(seeded):
    upload_pred(seeded, "m1", [("IMG1", "normal"), ("IMG4", "dent")])
    upload_truth(seeded, [("IMG1", 1, "normal"), ("IMG4", 1, "dent")])
    seeded.post("/partitions/assign", json={"image_id": "IMG1", "partition": "train"})
    seeded.post("/partitions/assign", json={"image_id": "IMG4", "partition": "val"})
    r = seeded.get("/models/m1/matrices")
    assert r.status_code == 200
    body = r.json()
    assert set(body["partitions"]) == {"train", "val"}
    assert body["partitions"]["train"]["state_version"] == body["state_version"]
    assert body["partitions"]["val"]["state_version"] == body["state_version"]


def test_partition_change_via_group_merge_updates_matrix(seeded):
    # IMG4 独立未分区组先有预测+真值（不计数）
    upload_pred(seeded, "m1", [("IMG4", "dent")])
    upload_truth(seeded, [("IMG4", 1, "dent")])
    assert matrix(seeded, "m1", "val")["support"] == 0
    # IMG3 组（组 A）指定 train，再导入 IMG5 桥接组 A 与 IMG4 -> 整组 train
    seeded.post("/partitions/assign", json={"image_id": "IMG3", "partition": "train"})
    r = import_images(seeded, [("IMG5", "WP-C", "LOT-2")])  # WP-C 连 IMG4，LOT-2 连组 A
    assert r.status_code == 200
    m = matrix(seeded, "m1", "train")
    assert m["matrix"]["dent"]["dent"] == 1  # IMG4 随并组进入 train 后才计数
    assert matrix(seeded, "m1", "val")["support"] == 0


def test_new_image_inheriting_partition_counts_after_uploads(seeded):
    seeded.post("/partitions/assign", json={"image_id": "IMG1", "partition": "train"})
    # 新图 WP-A 关联组 A -> 继承 train
    import_images(seeded, [("IMG9", "WP-A", "LOT-9")])
    upload_pred(seeded, "m1", [("IMG9", "scratch")])
    upload_truth(seeded, [("IMG9", 1, "dent")])
    m = matrix(seeded, "m1", "train")
    assert m["matrix"]["dent"]["scratch"] == 1 and m["support"] == 1


# ---------------------------------------------------------------------------
# 并发 / 重启 / 回滚
# ---------------------------------------------------------------------------

def test_concurrent_evaluation_uploads_remain_consistent(seeded):
    # 预置 20 张已分区图片（10 train / 10 val）
    rows = [(f"T{i}", f"WP-T{i}", f"LOT-T{i}") for i in range(10)]
    rows += [(f"V{i}", f"WP-V{i}", f"LOT-V{i}") for i in range(10)]
    payload = _csv(["image_id", "workpiece_no", "batch_no"], rows)
    seeded.post("/imports", files={"file": ("d.csv", payload, "text/csv")})
    for i in range(10):
        seeded.post("/partitions/assign", json={"image_id": f"T{i}", "partition": "train"})
        seeded.post("/partitions/assign", json={"image_id": f"V{i}", "partition": "val"})

    errors = []

    def worker(prefix, partition):
        try:
            for i in range(10):
                img = f"{prefix}{i}"
                r1 = upload_pred(seeded, "m1", [(img, "scratch")])
                r2 = upload_truth(seeded, [(img, 1, "dent")])
                if r1.status_code != 200 or r2.status_code != 200:
                    errors.append((img, r1.status_code, r2.status_code))
        except Exception as e:  # noqa: BLE001
            errors.append((prefix, repr(e)))

    t1 = threading.Thread(target=worker, args=("T", "train"))
    t2 = threading.Thread(target=worker, args=("V", "val"))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert errors == []
    mt = matrix(seeded, "m1", "train")
    mv = matrix(seeded, "m1", "val")
    assert mt["matrix"]["dent"]["scratch"] == 10
    assert mv["matrix"]["dent"]["scratch"] == 10
    # 贡献总数 = 矩阵 support
    contribs = seeded.get("/contributions", params={"model_id": "m1"}).json()
    assert contribs["count"] == 20


def test_restart_preserves_matrices(tmp_path, monkeypatch):
    db = tmp_path / "restart.db"
    monkeypatch.setenv("DATABASE_PATH", str(db))
    with TestClient(app) as c1:
        import_images(c1, [("IMG1", "WP-A", "LOT-1"), ("IMG2", "WP-B", "LOT-2")])
        c1.post("/partitions/assign", json={"image_id": "IMG1", "partition": "train"})
        upload_pred(c1, "m1", [("IMG1", "normal"), ("IMG2", "dent")])
        upload_truth(c1, [("IMG1", 1, "normal"), ("IMG2", 1, "scratch")])
        snapshot = c1.get("/models/m1/matrices").json()
    with TestClient(app) as c2:
        assert c2.get("/models/m1/matrices").json() == snapshot
        contrib = c2.get("/contributions").json()
        assert contrib["count"] == 1  # 仅 IMG1 已分区计数


def test_original_endpoints_unchanged(seeded):
    # 原接口保持可用
    assert seeded.get("/stats").status_code == 200
    assert seeded.get("/groups").status_code == 200
    assert seeded.get("/images/unpartitioned").status_code == 200
    assert seeded.get("/downloads/train").status_code == 200
    assert seeded.get("/health").status_code == 200
