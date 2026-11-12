"""模型评估核心逻辑：预测 / 真值上传与 3x3 混淆矩阵维护。

一致性设计
----------
1. 每批上传先在调用方完成纯文件级校验（编码、表头、取值、批内 ID 唯一），
   进入本模块后在**单个写事务**（BEGIN IMMEDIATE）内完成库内校验与写入，
   任何冲突都整体回滚，实现"校验失败整批拒绝、数据和统计一起生效或一起回滚"。

2. 矩阵从不全库重算。``reconcile_images`` 只重算给定图片集合的逐图片贡献：
   对比每张图片当前应有的贡献（所在组已分区 + 真值 + 预测齐备）与库中旧贡献，
   仅对发生变化的矩阵单元做增量增减，再写入/删除该图片的贡献行。

3. 分区变化（整组指定分区、新图继承、并组）由划分层在同一写事务内调用
   ``reconcile_images``，因此图片关系、分区、贡献与矩阵在同一事务中原子切换。
"""
from __future__ import annotations

from dataclasses import dataclass

from .db import bump_version

LABELS = ("normal", "scratch", "dent")
PARTITIONS = ("train", "val")


class PredictionConflictError(Exception):
    """同模型同图片已存在不同标签的预测。"""

    def __init__(self, model_id: str, image_id: str, existing: str, incoming: str):
        super().__init__(f"{model_id}:{image_id}")
        self.model_id = model_id
        self.image_id = image_id
        self.existing = existing
        self.incoming = incoming


class RevisionConflictError(Exception):
    """同修订号但标签不同。"""

    def __init__(self, image_id: str, revision: int, existing: str, incoming: str):
        super().__init__(image_id)
        self.image_id = image_id
        self.revision = revision
        self.existing = existing
        self.incoming = incoming


class UnregisteredImageError(Exception):
    """图片 ID 未在目录库登记。"""

    def __init__(self, image_ids: list[str]):
        super().__init__("unregistered images")
        self.image_ids = image_ids


@dataclass(frozen=True)
class PredRow:
    image_id: str
    label: str


@dataclass(frozen=True)
class TruthRow:
    image_id: str
    revision: int
    label: str


def _ph(items) -> str:
    return ",".join("?" * len(items))


# ---------------------------------------------------------------------------
# 贡献 / 矩阵增量维护（只调整受影响图片，不全库重算）
# ---------------------------------------------------------------------------

def _bump_cell(conn, model_id: str, partition: str, truth: str, pred: str, delta: int) -> None:
    if delta == 0:
        return
    if delta > 0:
        conn.execute(
            "INSERT INTO eval_matrices(model_id, partition, truth_label, pred_label, count) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id, partition, truth_label, pred_label) DO "
            "UPDATE SET count = eval_matrices.count + ?",
            (model_id, partition, truth, pred, delta, delta),
        )
    else:
        # 扣减只作用于已有单元（贡献存在则其单元必然存在），避免插入负计数
        conn.execute(
            "UPDATE eval_matrices SET count = count + ? "
            "WHERE model_id = ? AND partition = ? AND truth_label = ? AND pred_label = ?",
            (delta, model_id, partition, truth, pred),
        )
    # 计数归零的单元删除，保证矩阵持久化状态与"从零聚合贡献"严格一致
    conn.execute(
        "DELETE FROM eval_matrices WHERE model_id = ? AND partition = ? "
        "AND truth_label = ? AND pred_label = ? AND count = 0",
        (model_id, partition, truth, pred),
    )


def reconcile_images(conn, image_ids) -> list[str]:
    """重算给定图片的逐图片贡献并增量更新矩阵。

    在调用方的写事务内执行。每张 (model, image) 的目标贡献为：
      所在组已分区 + 该模型有预测 + 图片有真值 -> 对应矩阵单元 +1
    其他情况（缺真值 / 缺预测 / 未分区）-> 无贡献。

    返回实际发生贡献变化的图片 ID 列表。
    """
    if not image_ids:
        return []
    ids = sorted(set(image_ids))
    changed: list[str] = []

    # 受影响图片当前应有的全部目标贡献（只查涉及的图片，不扫全库）
    target_rows = conn.execute(
        f"SELECT p.model_id AS model_id, p.image_id AS image_id, "
        f"g.partition AS partition, t.label AS truth_label, p.label AS pred_label "
        f"FROM predictions p "
        f"JOIN images i ON i.image_id = p.image_id "
        f"JOIN groups g ON g.group_id = i.group_id "
        f"JOIN ground_truths t ON t.image_id = p.image_id "
        f"WHERE p.image_id IN ({_ph(ids)}) AND g.partition IS NOT NULL",
        ids,
    ).fetchall()
    targets = {(r["model_id"], r["image_id"]): r for r in target_rows}

    old_rows = conn.execute(
        f"SELECT model_id, image_id, partition, truth_label, pred_label "
        f"FROM eval_contributions WHERE image_id IN ({_ph(ids)})",
        ids,
    ).fetchall()
    olds = {(r["model_id"], r["image_id"]): r for r in old_rows}

    for key in olds.keys() | targets.keys():
        model_id, image_id = key
        old = olds.get(key)
        new = targets.get(key)
        old_sig = None if old is None else (
            old["partition"], old["truth_label"], old["pred_label"]
        )
        new_sig = None if new is None else (
            new["partition"], new["truth_label"], new["pred_label"]
        )
        if old_sig == new_sig:
            continue
        changed.append(image_id)
        if old is not None:
            _bump_cell(conn, model_id, old["partition"], old["truth_label"], old["pred_label"], -1)
            conn.execute(
                "DELETE FROM eval_contributions WHERE model_id = ? AND image_id = ?",
                (model_id, image_id),
            )
        if new is not None:
            _bump_cell(conn, model_id, new["partition"], new["truth_label"], new["pred_label"], +1)
            conn.execute(
                "INSERT INTO eval_contributions(model_id, image_id, partition, truth_label, pred_label) "
                "VALUES (?, ?, ?, ?, ?)",
                (model_id, image_id, new["partition"], new["truth_label"], new["pred_label"]),
            )
    return changed


# ---------------------------------------------------------------------------
# 预测上传
# ---------------------------------------------------------------------------

def upload_predictions(conn, model_id: str, rows: list[PredRow]) -> dict:
    """在调用方提供的写事务内写入一批预测。

    - 图片必须已登记，否则整批拒绝；
    - 同模型同图片预测不可改：异标签整批拒绝；同标签重传复用（幂等）。
    """
    ids = sorted({r.image_id for r in rows})
    registered = {
        r["image_id"] for r in conn.execute(
            f"SELECT image_id FROM images WHERE image_id IN ({_ph(ids)})", ids
        )
    }
    unknown = sorted(set(ids) - registered)
    if unknown:
        raise UnregisteredImageError(unknown)

    existing = {
        r["image_id"]: r["label"]
        for r in conn.execute(
            f"SELECT image_id, label FROM predictions "
            f"WHERE model_id = ? AND image_id IN ({_ph(ids)})",
            [model_id, *ids],
        )
    }

    inserted = 0
    reused = 0
    touched: list[str] = []
    for r in rows:
        old = existing.get(r.image_id)
        if old is None:
            conn.execute(
                "INSERT INTO predictions(model_id, image_id, label) VALUES (?, ?, ?)",
                (model_id, r.image_id, r.label),
            )
            inserted += 1
            touched.append(r.image_id)
        elif old != r.label:
            raise PredictionConflictError(model_id, r.image_id, old, r.label)
        else:
            reused += 1  # 同标签重传复用，不产生任何写入

    reconcile_images(conn, touched)
    if inserted:
        bump_version(conn)
    return {
        "model_id": model_id,
        "received": len(rows),
        "inserted": inserted,
        "reused": reused,
        "changed": bool(inserted),
    }


# ---------------------------------------------------------------------------
# 真值上传
# ---------------------------------------------------------------------------

def upload_ground_truths(conn, rows: list[TruthRow]) -> dict:
    """在调用方提供的写事务内写入一批真值（修订号机制）。

    - 图片必须已登记，否则整批拒绝；
    - 只接受更高修订：低修订忽略；同修订同标签忽略（幂等）；
      同修订异标签 -> 整批拒绝。
    - 接受更高修订时替换旧真值，并扣除该图片所有模型的旧矩阵贡献。
    """
    ids = sorted({r.image_id for r in rows})
    registered = {
        r["image_id"] for r in conn.execute(
            f"SELECT image_id FROM images WHERE image_id IN ({_ph(ids)})", ids
        )
    }
    unknown = sorted(set(ids) - registered)
    if unknown:
        raise UnregisteredImageError(unknown)

    current = {
        r["image_id"]: r
        for r in conn.execute(
            f"SELECT image_id, revision, label FROM ground_truths "
            f"WHERE image_id IN ({_ph(ids)})",
            ids,
        )
    }

    accepted = 0
    ignored_lower = 0
    ignored_same = 0
    touched: list[str] = []
    for r in rows:
        old = current.get(r.image_id)
        if old is None:
            conn.execute(
                "INSERT INTO ground_truths(image_id, revision, label) VALUES (?, ?, ?)",
                (r.image_id, r.revision, r.label),
            )
            accepted += 1
            touched.append(r.image_id)
        elif r.revision < old["revision"]:
            ignored_lower += 1  # 低修订忽略
        elif r.revision == old["revision"]:
            if r.label != old["label"]:
                raise RevisionConflictError(
                    r.image_id, r.revision, old["label"], r.label
                )
            ignored_same += 1  # 同修订同标签：幂等忽略
        else:
            # 更高修订：更正真值；reconcile 负责扣除旧贡献、建立新贡献
            conn.execute(
                "UPDATE ground_truths SET revision = ?, label = ? WHERE image_id = ?",
                (r.revision, r.label, r.image_id),
            )
            accepted += 1
            touched.append(r.image_id)

    reconcile_images(conn, touched)
    if accepted:
        bump_version(conn)
    return {
        "received": len(rows),
        "accepted": accepted,
        "ignored_lower_revision": ignored_lower,
        "ignored_same_revision": ignored_same,
        "changed": bool(accepted),
    }


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def read_matrix(conn, model_id: str, partition: str) -> dict:
    """单个模型 × 分区的 3x3 矩阵（行=真实标签，列=预测标签），含状态版本。"""
    counts = {
        (r["truth_label"], r["pred_label"]): r["count"]
        for r in conn.execute(
            "SELECT truth_label, pred_label, count FROM eval_matrices "
            "WHERE model_id = ? AND partition = ?",
            (model_id, partition),
        )
    }
    matrix = {t: {p: counts.get((t, p), 0) for p in LABELS} for t in LABELS}
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[l][l] for l in LABELS)
    return {
        "model_id": model_id,
        "partition": partition,
        "truth_labels": list(LABELS),
        "pred_labels": list(LABELS),
        "matrix": matrix,
        "support": total,
        "correct": correct,
        "accuracy": (correct / total) if total else None,
    }


def read_all_matrices(conn, model_id: str) -> dict:
    return {
        "model_id": model_id,
        "partitions": {p: read_matrix(conn, model_id, p) for p in PARTITIONS},
    }


def list_models(conn) -> list[dict]:
    return [
        {
            "model_id": r["model_id"],
            "prediction_images": r["prediction_images"],
            "contribution_images": r["contribution_images"],
        }
        for r in conn.execute(
            "SELECT p.model_id AS model_id, "
            "COUNT(DISTINCT p.image_id) AS prediction_images, "
            "COUNT(DISTINCT c.image_id) AS contribution_images "
            "FROM predictions p "
            "LEFT JOIN eval_contributions c ON c.model_id = p.model_id "
            "GROUP BY p.model_id ORDER BY p.model_id"
        )
    ]


def list_contributions(conn, model_id: str | None, partition: str | None, limit: int) -> dict:
    sql = (
        "SELECT model_id, image_id, partition, truth_label, pred_label "
        "FROM eval_contributions WHERE 1=1"
    )
    params: list = []
    if model_id is not None:
        sql += " AND model_id = ?"
        params.append(model_id)
    if partition is not None:
        sql += " AND partition = ?"
        params.append(partition)
    sql += " ORDER BY model_id, image_id LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return {"count": len(rows), "contributions": [dict(r) for r in rows]}


def image_evaluation(conn, image_id: str) -> dict:
    """单张图片的真值、各模型预测与其当前矩阵贡献（调试/核对用）。"""
    truth = conn.execute(
        "SELECT revision, label FROM ground_truths WHERE image_id = ?", (image_id,)
    ).fetchone()
    preds = [
        dict(r)
        for r in conn.execute(
            "SELECT model_id, label FROM predictions WHERE image_id = ? ORDER BY model_id",
            (image_id,),
        )
    ]
    contribs = [
        dict(r)
        for r in conn.execute(
            "SELECT model_id, partition, truth_label, pred_label "
            "FROM eval_contributions WHERE image_id = ? ORDER BY model_id",
            (image_id,),
        )
    ]
    return {
        "image_id": image_id,
        "ground_truth": None if truth is None else dict(truth),
        "predictions": preds,
        "contributions": contribs,
    }
