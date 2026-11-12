"""标准库验证脚本：不依赖 FastAPI，直接驱动 app.core / app.evaluation + sqlite3。

运行：python3 -I scripts/verify_core.py
覆盖：预测/真值上传校验语义、修订机制、3x3 矩阵增量维护、真值更正扣旧贡献、
整组分区/新图继承/并组触发的矩阵同步、原子回滚、重启一致性、并发、无全库重算。
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import core, evaluation  # noqa: E402
from app.db import connect, init_db, read_txn, state_version, write_txn  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not cond:
        failures.append(name)


def imp(rows):
    return core.import_rows([core.ImageRow(*r) for r in rows])


def pred(model, rows):
    conn = connect()
    try:
        with write_txn(conn):
            return evaluation.upload_predictions(
                conn, model, [evaluation.PredRow(i, l) for i, l in rows]
            )
    finally:
        conn.close()


def truth(rows):
    conn = connect()
    try:
        with write_txn(conn):
            return evaluation.upload_ground_truths(
                conn, [evaluation.TruthRow(i, rev, l) for i, rev, l in rows]
            )
    finally:
        conn.close()


def matrix(model, partition):
    conn = connect()
    try:
        with read_txn(conn):
            return evaluation.read_matrix(conn, model, partition)
    finally:
        conn.close()


def cell(model, partition, t, p):
    return matrix(model, partition)["matrix"][t][p]


def expect_matrix(model, partition, expected: dict[tuple[str, str], int]) -> None:
    m = matrix(model, partition)["matrix"]
    actual = {(t, p): m[t][p] for t in evaluation.LABELS for p in evaluation.LABELS if m[t][p]}
    check(
        f"{model}/{partition} 矩阵单元精确一致",
        actual == expected,
        f"actual={actual} expected={expected}",
    )
    check(
        f"{model}/{partition} support 与矩阵求和一致",
        matrix(model, partition)["support"] == sum(expected.values()),
    )


def main() -> None:
    tmp = tempfile.mkdtemp()
    os.environ["DATABASE_PATH"] = os.path.join(tmp, "verify.db")
    init_db()

    # 组1: IMG001..004（关联链），组2: IMG005 独立；IMG006 独立
    imp([
        ("IMG001", "WP-A", "LOT-1"),
        ("IMG002", "WP-A", "LOT-1"),
        ("IMG003", "WP-B", "LOT-1"),
        ("IMG004", "WP-B", "LOT-2"),
        ("IMG005", "WP-C", "LOT-3"),
        ("IMG006", "WP-Z", "LOT-9"),
    ])

    # ---- 1. 未登记图片：整批拒绝 ----
    try:
        pred("m1", [("GHOST", "normal"), ("IMG001", "dent")])
        check("预测未登记图片整批拒绝", False)
    except evaluation.UnregisteredImageError as e:
        check("预测未登记图片整批拒绝", e.image_ids == ["GHOST"], str(e.image_ids))
    try:
        truth([("GHOST", 1, "normal")])
        check("真值未登记图片整批拒绝", False)
    except evaluation.UnregisteredImageError:
        check("真值未登记图片整批拒绝", True)

    # ---- 2. 未分区：预测/真值可入库，但矩阵不计数 ----
    r = pred("m1", [("IMG001", "scratch"), ("IMG002", "normal"), ("IMG003", "dent")])
    check("预测首批写入", r["inserted"] == 3 and r["reused"] == 0, str(r))
    r = truth([("IMG001", 1, "scratch"), ("IMG002", 1, "dent"), ("IMG004", 1, "normal")])
    check("真值首批接受", r["accepted"] == 3, str(r))
    expect_matrix("m1", "train", {})
    expect_matrix("m1", "val", {})
    check("未分区无贡献行", _count("eval_contributions") == 0)

    # ---- 3. 同标签重传复用；异标签冲突整批拒绝 ----
    r = pred("m1", [("IMG001", "scratch"), ("IMG002", "normal")])
    check("同标签预测重传复用", r["inserted"] == 0 and r["reused"] == 2, str(r))
    before_ver = _version()
    try:
        pred("m1", [("IMG001", "dent"), ("IMG003", "normal")])
        check("预测冲突整批拒绝", False)
    except evaluation.PredictionConflictError as e:
        check("预测冲突整批拒绝", e.image_id == "IMG001" and e.existing == "scratch")
    check("冲突后 IMG003 预测未被写入", _pred_label("m1", "IMG003") == "dent")
    check("冲突不推进版本号", _version() == before_ver)

    # ---- 4. 整组指定 train：组1（IMG001..004）开始计数 ----
    core.assign_partition("IMG003", "train")
    # IMG001: t=scratch/p=scratch; IMG002: t=dent/p=normal; IMG003: 有预测无真值不计数;
    # IMG004: 有真值无预测不计数
    expect_matrix("m1", "train", {("scratch", "scratch"): 1, ("dent", "normal"): 1})
    check("贡献仅覆盖齐备的2张", _count("eval_contributions") == 2)
    m = matrix("m1", "train")
    check("accuracy 正确", abs(m["accuracy"] - 0.5) < 1e-9 and m["correct"] == 1, str(m["accuracy"]))

    # ---- 5. 缺预测/缺真值后续补齐：只增量加入 ----
    pred("m1", [("IMG004", "normal")])  # IMG004 t=normal p=normal
    expect_matrix("m1", "train", {
        ("scratch", "scratch"): 1, ("dent", "normal"): 1, ("normal", "normal"): 1,
    })
    truth([("IMG003", 2, "dent")])  # IMG003 t=dent p=dent
    expect_matrix("m1", "train", {
        ("scratch", "scratch"): 1, ("dent", "normal"): 1,
        ("normal", "normal"): 1, ("dent", "dent"): 1,
    })

    # ---- 6. 真值更高修订更正：扣除旧贡献，建立新贡献 ----
    truth([("IMG001", 3, "dent")])  # scratch/scratch -1, dent/scratch +1
    expect_matrix("m1", "train", {
        ("dent", "scratch"): 1, ("dent", "normal"): 1,
        ("normal", "normal"): 1, ("dent", "dent"): 1,
    })
    r = truth([("IMG001", 2, "normal")])  # 低修订忽略
    check("低修订真值忽略", r["ignored_lower_revision"] == 1 and r["accepted"] == 0, str(r))
    expect_matrix("m1", "train", {
        ("dent", "scratch"): 1, ("dent", "normal"): 1,
        ("normal", "normal"): 1, ("dent", "dent"): 1,
    })
    r = truth([("IMG001", 3, "dent")])  # 同修订同标签幂等
    check("同修订同标签幂等", r["ignored_same_revision"] == 1, str(r))
    try:
        truth([("IMG001", 3, "scratch")])  # 同修订异标签：整批拒绝
        check("同修订异标签整批拒绝", False)
    except evaluation.RevisionConflictError as e:
        check("同修订异标签整批拒绝", e.revision == 3 and e.existing == "dent")
    check("修订冲突后真值未变", _truth("IMG001") == (3, "dent"))

    # ---- 7. 新图继承分区：导入即入 train 组，随后上传预测/真值自动计数 ----
    imp([("IMG007", "WP-A", "LOT-77")])  # WP-A 关联 train 组 -> 继承 train
    check("新图继承 train", _partition("IMG007") == "train")
    pred("m1", [("IMG007", "normal")])
    truth([("IMG007", 1, "normal")])
    expect_matrix("m1", "train", {
        ("dent", "scratch"): 1, ("dent", "normal"): 1,
        ("normal", "normal"): 2, ("dent", "dent"): 1,
    })

    # ---- 8. 并组引起分区变化：未分区组并入 val 组，贡献同步建立 ----
    core.assign_partition("IMG005", "val")
    pred("m1", [("IMG005", "scratch"), ("IMG006", "dent")])
    truth([("IMG005", 1, "scratch"), ("IMG006", 1, "normal")])
    expect_matrix("m1", "val", {("scratch", "scratch"): 1})  # IMG006 未分区不计数
    check("IMG006 无贡献", _contrib_partition("IMG006") is None)
    # IMG008 把 IMG006（WP-Z/LOT-9 组）连到 val 组（WP-C）
    imp([("IMG008", "WP-C", "LOT-9")])
    check("并组后 IMG006 继承 val", _partition("IMG006") == "val")
    expect_matrix("m1", "val", {
        ("scratch", "scratch"): 1, ("normal", "dent"): 1,
    })
    check("IMG006 贡献已随并组建立", _contrib_partition("IMG006") == "val")

    # ---- 9. 多模型独立矩阵，互不影响 ----
    pred("m2", [("IMG001", "normal"), ("IMG002", "normal")])
    expect_matrix("m2", "train", {("dent", "normal"): 2})
    # m1 保持不变
    expect_matrix("m1", "train", {
        ("dent", "scratch"): 1, ("dent", "normal"): 1,
        ("normal", "normal"): 2, ("dent", "dent"): 1,
    })

    # ---- 10. 矩阵与逐图片贡献聚合始终一致（随机核对全部模型×分区） ----
    ok = _matrices_match_contributions()
    check("矩阵 = 贡献聚合（全量核对）", ok)

    # ---- 11. 事务回滚：写入中途失败，数据与统计一起回滚 ----
    conn = connect()
    try:
        try:
            with write_txn(conn):
                evaluation.upload_predictions(
                    conn, "m3", [evaluation.PredRow("IMG001", "normal")]
                )
                raise RuntimeError("simulated crash")
        except RuntimeError:
            pass
    finally:
        conn.close()
    check("回滚后预测不存在", _pred_label("m3", "IMG001") is None)
    check("回滚后矩阵无 m3", _count("eval_matrices", "model_id = 'm3'") == 0)
    check("回滚后贡献无 m3", _count("eval_contributions", "model_id = 'm3'") == 0)

    # ---- 12. 重启一致性：重新 init_db（同文件），状态保持 ----
    init_db()
    check("重启后矩阵一致", _matrices_match_contributions())
    expect_matrix("m1", "val", {("scratch", "scratch"): 1, ("normal", "dent"): 1})

    # ---- 13. 并发：多线程混合 预测/真值/导入/分区，最终矩阵仍等于贡献聚合 ----
    threads = []

    def worker(i):
        img = f"CP{i:02d}"
        # 模拟 HTTP 客户端：导入遇 503（重试用尽）时退避重试
        for _ in range(20):
            try:
                imp([(img, f"WP-CP-{i}", f"LOT-CP-{i}")])  # 独立未分区新组
                break
            except core.ConcurrentModificationError:
                import time as _t
                _t.sleep(0.02)
        else:
            failures.append(f"CP{i:02d} 导入重试耗尽")
        pred("m1", [(img, "scratch")])
        truth([(img, i + 1, "dent")])
        core.assign_partition(img, "val" if i % 2 else "train")

    for i in range(12):
        threads.append(threading.Thread(target=worker, args=(i,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("并发后矩阵=贡献聚合", _matrices_match_contributions())
    # 12 张并发图：真值 dent、预测 scratch -> 各分区累加 dent/scratch
    n_train = sum(1 for i in range(12) if i % 2 == 0)
    n_val = 12 - n_train
    # train 的 dent/scratch 基线含 IMG001 更正后的 1；此时 val 的 dent/scratch 基线为 0
    # （IMG005 仍是 scratch/scratch，其 dent/scratch 修订发生在下一步）
    check(
        "并发图片全部按分区计数",
        cell("m1", "train", "dent", "scratch") == 1 + n_train
        and cell("m1", "val", "dent", "scratch") == n_val,
        f"train d/s={cell('m1','train','dent','scratch')} val d/s={cell('m1','val','dent','scratch')}",
    )

    # ---- 14. 并发真值修订：同图并发更高修订，只接受最高且矩阵不重不漏 ----
    img = "IMG005"  # 当前 rev1 scratch/scratch @val
    outcomes = []

    def rev_worker(rev, label):
        try:
            outcomes.append((rev, truth([(img, rev, label)])))
        except Exception as exc:  # noqa: BLE001
            outcomes.append((rev, type(exc).__name__))

    ts = [
        threading.Thread(target=rev_worker, args=(5, "dent")),
        threading.Thread(target=rev_worker, args=(4, "normal")),
        threading.Thread(target=rev_worker, args=(6, "dent")),
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check("IMG005 最终修订为最高", _truth(img) == (6, "dent"), str(_truth(img)))
    check("并发修订后矩阵=贡献聚合", _matrices_match_contributions())
    check("IMG005 最终贡献 dent/scratch@val",
          _contrib("m1", img) == ("val", "dent", "scratch"), str(_contrib("m1", img)))

    # ---- 15. 矩阵零单元不落库；删除路径不留垃圾 ----
    check("无 count=0 的矩阵单元", _count("eval_matrices", "count = 0") == 0)

    print()
    if failures:
        print(f"{len(failures)} 项失败: {failures}")
        sys.exit(1)
    print("全部核心验证通过 ✔")


# ---------------------------------------------------------------------------
# 辅助查询
# ---------------------------------------------------------------------------

def _version() -> int:
    conn = connect()
    try:
        with read_txn(conn):
            return state_version(conn)
    finally:
        conn.close()


def _count(table: str, where: str = "") -> int:
    conn = connect()
    try:
        with read_txn(conn):
            return conn.execute(f"SELECT COUNT(*) AS c FROM {table} {('WHERE ' + where) if where else ''}").fetchone()["c"]
    finally:
        conn.close()


def _pred_label(model, image_id):
    conn = connect()
    try:
        with read_txn(conn):
            r = conn.execute(
                "SELECT label FROM predictions WHERE model_id=? AND image_id=?",
                (model, image_id),
            ).fetchone()
            return None if r is None else r["label"]
    finally:
        conn.close()


def _truth(image_id):
    conn = connect()
    try:
        with read_txn(conn):
            r = conn.execute(
                "SELECT revision, label FROM ground_truths WHERE image_id=?", (image_id,)
            ).fetchone()
            return None if r is None else (r["revision"], r["label"])
    finally:
        conn.close()


def _partition(image_id):
    conn = connect()
    try:
        with read_txn(conn):
            return conn.execute(
                "SELECT g.partition AS p FROM images i JOIN groups g ON g.group_id=i.group_id "
                "WHERE i.image_id=?", (image_id,)
            ).fetchone()["p"]
    finally:
        conn.close()


def _contrib_partition(image_id):
    conn = connect()
    try:
        with read_txn(conn):
            r = conn.execute(
                "SELECT partition FROM eval_contributions WHERE image_id=? LIMIT 1", (image_id,)
            ).fetchone()
            return None if r is None else r["partition"]
    finally:
        conn.close()


def _contrib(model, image_id):
    conn = connect()
    try:
        with read_txn(conn):
            r = conn.execute(
                "SELECT partition, truth_label, pred_label FROM eval_contributions "
                "WHERE model_id=? AND image_id=?", (model, image_id)
            ).fetchone()
            return None if r is None else (r["partition"], r["truth_label"], r["pred_label"])
    finally:
        conn.close()


def _matrices_match_contributions() -> bool:
    conn = connect()
    try:
        with read_txn(conn):
            agg = {
                (r["model_id"], r["partition"], r["truth_label"], r["pred_label"]): r["c"]
                for r in conn.execute(
                    "SELECT model_id, partition, truth_label, pred_label, COUNT(*) AS c "
                    "FROM eval_contributions GROUP BY 1,2,3,4"
                )
            }
            stored = {
                (r["model_id"], r["partition"], r["truth_label"], r["pred_label"]): r["count"]
                for r in conn.execute(
                    "SELECT model_id, partition, truth_label, pred_label, count FROM eval_matrices"
                )
            }
            return agg == stored
    finally:
        conn.close()


if __name__ == "__main__":
    main()
