"""预测掩码 -> 骨架化 -> 剪枝 -> 交叉点续接配链，估算 根数量 / 各根长度 / 总长度。

流程：
1. 轻度腐蚀（disk=1，默认 1 次）：消除 5px 画线/预测线的厚度伪影；
2. 二值掩码骨架化（skimage），在原图分辨率上进行；
3. 剪枝：移除长度短于 spur(像素) 的末梢（噪声）；
4. 链收缩：度数=2 的骨架像素折叠成带权链边（直连1、对角 sqrt(2)），
   只保留 叶端(度1)/分叉点(度>=3)；邻近分叉点(<=6px)合并成单节点，
   消除线条交叉处厚度产生的簇状伪分叉；
5. 分叉点续接：在每个分叉点把"方向最连贯"的两条臂配成一对
   （十字交叉的两条根在交点处走向连续，配对即把交叉的根"穿过去"还原整根）；
   不成对的多余臂 = 该根的起点/终止端；
6. 从 叶端/未配对臂 起步沿配对关系串成完整轨迹：每条轨迹 = 1 条根，
   长度 = 路径上各链边长之和（像素欧氏，与 RSML 口径一致）；
   轨迹同时保留像素序列，可按弧长抽稀导出为 RSML 折线（见 extract_root_paths）。

说明：无先验的近似拆分（相切/粘连时走向可能误配），误差在 test.py 汇总对比
中体现；阈值参数化便于调优。
"""
import numpy as np
from PIL import Image
from skimage.morphology import disk, erosion, skeletonize

_EPS = 1e-9


def _step_len(a, b) -> float:
    return 1.4142135623730951 if (a[0] != b[0] and a[1] != b[1]) else 1.0


def _unit(dy, dx):
    n = (dy * dy + dx * dx) ** 0.5
    return (dy / n, dx / n) if n > _EPS else (0.0, 1.0)


def _prune(adj, spur_s):
    """从叶端向内剥除长度 < spur_s 的末梢（原地修改 adj）。"""
    pruned = True
    while pruned:
        pruned = False
        for leaf in [p for p in adj if len(adj[p]) == 1]:
            if leaf not in adj:
                continue
            path = [leaf]
            cur, prev, length = leaf, None, 0.0
            while True:
                nbrs = [n for n in adj[cur] if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                length += adj[cur][nxt]
                if length >= spur_s or len(adj.get(nxt, ())) > 2:
                    break
                path.append(nxt)
                prev, cur = cur, nxt
            if length < spur_s:
                for p in path:
                    if p not in adj:
                        continue
                    for q in adj[p]:
                        if q in adj:
                            adj[q].pop(p, None)
                    del adj[p]
                pruned = True
    for p in [p for p in adj if not adj[p]]:
        del adj[p]


def _components(adj):
    seen, comps = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            p = stack.pop()
            comp.append(p)
            for q in adj[p]:
                if q not in seen:
                    seen.add(q)
                    stack.append(q)
        comps.append(comp)
    return comps


def _merge_junctions(nodes, adj):
    """把邻近(<=6px)的分叉点归并为一个代表点，返回 原节点->代表点 映射。

    只合并分叉点(度>=3)，绝不合并叶端（相邻两条根的端部可能只有几像素远）。
    """
    juncs = [p for p in nodes if len(adj[p]) >= 3]
    parent = {p: p for p in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(juncs):
        for b in juncs[i + 1:]:
            if abs(a[0] - b[0]) <= 6 and abs(a[1] - b[1]) <= 6:
                union(a, b)
    return {p: find(p) for p in nodes}


def _strands_of_component(comp_nodes, adj, spur_s, min_len):
    """对一个连通块拆根。

    返回 [[长度(像素), 像素轨迹[(y,x), ...]], ...]；轨迹首点是一个根端。
    """
    # ---- 全部为度2节点 -> 纯环，整块 1 条根 ----
    if all(len(adj[p]) == 2 for p in comp_nodes):
        p0 = comp_nodes[0]
        prev, cur = p0, next(iter(adj[p0]))
        total = 0.0
        px = [p0]
        while cur != p0:
            total += adj[prev][cur]
            px.append(cur)
            a, b = tuple(adj[cur])
            nxt = a if a != prev else b
            prev, cur = cur, nxt
        total += adj[prev][cur]
        return [[total, px]] if total >= min_len else []

    # ---- 链收缩：节点 = 叶端/分叉点；边 = 带像素序列的链 ----
    node_set = {p for p in comp_nodes if len(adj[p]) != 2}
    # 同一条链若从两端各走一遍会登记成两条，故只从"字典序较小"端点登记
    edges = []  # (a, b, w, d_a, d_b, chain)  chain: a -> b 的骨架像素(含两端)
    for p in node_set:
        for nxt0 in adj[p]:
            prev, cur = p, nxt0
            w = adj[p][nxt0]
            chain = [p, nxt0]
            while len(adj[cur]) == 2:
                a, b = tuple(adj[cur])
                nxt = a if a != prev else b
                w += adj[cur][nxt]
                prev, cur = cur, nxt
                chain.append(cur)
            end = cur
            if end == p or end < p:
                continue  # 自环或由另一端登记
            da = _unit(chain[1][0] - p[0], chain[1][1] - p[1])       # p 端方向
            db = _unit(chain[-2][0] - end[0], chain[-2][1] - end[1])  # end 端方向
            edges.append((p, end, w, da, db, chain))

    # ---- 邻近分叉点合并 ----
    rep_of = _merge_junctions(node_set, adj)
    arms = {}  # 代表节点 -> [{end, w, dout, px}]
    for (a, b, w, da, db, chain) in edges:
        ra, rb = rep_of[a], rep_of[b]
        if ra == rb:
            continue  # 簇内自环，长度极小，忽略
        arms.setdefault(ra, []).append(
            {"end": rb, "w": w, "dout": da, "px": chain})
        arms.setdefault(rb, []).append(
            {"end": ra, "w": w, "dout": db, "px": chain[::-1]})

    # ---- 分叉点续接配对 ----
    # 第一轮：只配"方向连贯"(接近直通)的对，避免把垂直粘连误连；
    # 第二轮：剩余臂两两按最连贯方向补配（剩 0/1 条为止），
    #         保证轨迹能贯通到真正的根端，计数贴近 叶端数/2。
    paired = {}
    for j, jarms in arms.items():
        if len(jarms) < 3:
            continue
        idx = list(range(len(jarms)))
        use = {}

        def best_pair(idx):
            best = None
            for ai, i in enumerate(idx):
                for aj in range(ai + 1, len(idx)):
                    j2 = idx[aj]
                    d, e = jarms[i]["dout"], jarms[j2]["dout"]
                    cost = 1.0 + d[0] * e[0] + d[1] * e[1]
                    if best is None or cost < best[0]:
                        best = (cost, i, j2)
            return best

        while len(idx) >= 2:
            cand = best_pair(idx)
            if cand is None or cand[0] > 0.8:
                break  # 第一轮：非直通不再配
            _, bi, bj = cand
            use[bi], use[bj] = bj, bi
            idx.remove(bi)
            idx.remove(bj)
        while len(idx) >= 2:  # 第二轮：无条件补配到 0/1 条
            _, bi, bj = best_pair(idx)
            use[bi], use[bj] = bj, bi
            idx.remove(bi)
            idx.remove(bj)
        if use:
            paired[j] = use

    # ---- 沿配对串轨迹 ----
    seen_arms = set()
    results = []  # [[长度, 像素轨迹], ...]

    def traverse(start_node, start_k):
        """从某臂起步串一条轨迹，返回 (长度, 像素轨迹)；遇到已消费臂返回 None。"""
        length = 0.0
        px = []
        node, k = start_node, start_k
        while True:
            if (node, k) in seen_arms:
                return (length, px) if length > 0 else None
            if node not in arms or k >= len(arms[node]):
                return None
            seen_arms.add((node, k))
            arm = arms[node][k]
            seg = arm["px"]
            px.extend(seg if not px else seg[1:])
            length += arm["w"]
            n2 = arm["end"]
            k2 = None
            for i2, a2 in enumerate(arms.get(n2, ())):
                if a2["end"] == node:
                    k2 = i2
                    break
            if k2 is None:
                return length, px
            juse = paired.get(n2)
            if juse and k2 in juse:
                seen_arms.add((n2, k2))  # 到达侧臂已消费
                node, k = n2, juse[k2]
                continue
            seen_arms.add((n2, k2))
            return length, px

    starts = []
    for p, parms in arms.items():
        if len(parms) == 1:
            starts.append((p, 0))  # 叶端
    for p, parms in arms.items():
        if len(parms) >= 3:
            juse = paired.get(p, {})
            for k in range(len(parms)):
                if k not in juse:
                    starts.append((p, k))  # 未配对的起点臂
    started_set = set(starts)
    for s in starts:
        r = traverse(*s)
        if r is not None and r[0] >= min_len:
            results.append([r[0], r[1]])
    # 兜底：剩余未消费臂（环等）也串起来，避免丢长度
    for p, parms in arms.items():
        for k in range(len(parms)):
            if (p, k) not in seen_arms and (p, k) not in started_set:
                r = traverse(p, k)
                if r is not None and r[0] >= min_len:
                    results.append([r[0], r[1]])

    # 计数归一：本块根数 = ceil(叶端数/2)（每根两端在图上分开、互不粘连时严格成立，
    # GT 统计验证 23 根 -> 23)。轨迹多于该值时，把最短轨迹按"端点最近"原则
    # 拼接回现有轨迹（既保住总长，又让每条折线几何上连续完整）。
    n_leaves = sum(1 for p, v in arms.items() if len(v) == 1)
    results.sort(key=lambda t: t[0], reverse=True)
    if n_leaves >= 2:
        k = (n_leaves + 1) // 2
        while len(results) > k:
            piece = results.pop()  # 最短的一条
            ppts = piece[1]
            p_ends = (ppts[0], ppts[-1])
            best = None
            for idx, (_, pts) in enumerate(results):
                for end_i in (0, 1):
                    for piece_end in (0, 1):
                        e1, e2 = pts[0 if end_i == 0 else -1], p_ends[piece_end]
                        d = (e1[0] - e2[0]) ** 2 + (e1[1] - e2[1]) ** 2
                        if best is None or d < best[0]:
                            best = (d, idx, end_i, piece_end)
            _, idx, end_i, piece_end = best
            L, pts = results[idx]
            if end_i == 1:  # 接到 pts 尾部
                seg = ppts if piece_end == 0 else ppts[::-1]
                results[idx] = [L + piece[0], pts + seg]
            else:           # 接到 pts 头部
                seg = ppts if piece_end == 1 else ppts[::-1]
                results[idx] = [L + piece[0], seg + pts]
        results.sort(key=lambda t: t[0], reverse=True)
    return results


def _decimate(pts, spacing):
    """像素轨迹 -> 折线点 [(x, y), ...]：每约 spacing 像素取一点（同 RSML 控制点间距）。"""
    line = [pts[0][::-1]]
    acc = 0.0
    for prev, cur in zip(pts[:-1], pts[1:]):
        acc += _step_len(prev, cur)
        if acc >= spacing:
            line.append(cur[::-1])
            acc = 0.0
    end = pts[-1][::-1]
    if line[-1] != end:
        line.append(end)
    if len(line) < 2:
        line.append(end)
    return [(int(x), int(y)) for (x, y) in line]


def analyze_mask_ex(mask: np.ndarray, spur: float = 30.0, min_len: float = 20.0,
                    erode_iters: int = 1, with_paths: bool = False,
                    spacing: float = 50.0) -> dict:
    """mask: (h, w) bool 原图分辨率二值掩码。

    返回 {"count", "lengths"(降序), "total"}；with_paths=True 时附加
    "paths": [[(x, y), ...], ...]（与 lengths 一一对应、同序的抽稀折线）。
    """
    empty = {"count": 0, "lengths": [], "total": 0.0}
    if mask is None or mask.ndim != 2 or not mask.any():
        return {**empty, "paths": []} if with_paths else empty

    m = mask
    for _ in range(erode_iters):  # 消除线宽厚度伪影
        m = erosion(m, footprint=disk(1))
    if not m.any():
        return {**empty, "paths": []} if with_paths else empty

    skel = skeletonize(m)
    ys, xs = np.nonzero(skel)
    pts = {(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())}
    if not pts:
        return {**empty, "paths": []} if with_paths else empty

    # 带权 8 邻域邻接表
    adj = {p: {} for p in pts}
    for (y, x) in pts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in pts:
                    w = _step_len((y, x), q)
                    adj[(y, x)][q] = w
                    adj[q][(y, x)] = w

    _prune(adj, spur)
    if not adj:
        return {**empty, "paths": []} if with_paths else empty

    entries = []
    for comp in _components(adj):
        entries.extend(_strands_of_component(comp, adj, spur, min_len))
    if not entries:
        return {**empty, "paths": []} if with_paths else empty
    entries.sort(key=lambda t: t[0], reverse=True)
    lengths = [t[0] for t in entries]
    out = {"count": len(lengths), "lengths": lengths,
           "total": float(sum(lengths))}
    if with_paths:
        out["paths"] = [_decimate(t[1], spacing) for t in entries]
    return out


def analyze_mask(mask: np.ndarray, spur: float = 30.0, min_len: float = 20.0,
                 erode_iters: int = 1) -> dict:
    """兼容入口：只要统计量（count / lengths / total）。"""
    return analyze_mask_ex(mask, spur, min_len, erode_iters, with_paths=False)


def extract_root_paths(mask: np.ndarray, spur: float = 30.0, min_len: float = 20.0,
                       erode_iters: int = 1, spacing: float = 50.0):
    """返回逐根折线 [[(x, y), ...], ...]（按长度降序，与 analyze_mask 的长度一一对应）。

    spacing: 折线点抽稀间距(px)，默认 50，与标注 RSML 的
    controlpointseparation="50" 一致。可直接喂给 common.rsml_export.write_rsml。
    """
    return analyze_mask_ex(mask, spur, min_len, erode_iters,
                           with_paths=True, spacing=spacing)["paths"]
