#Bellow how to run the scrept

#python3 extract1_intermsg.py \
#  --root "/home/instantf2md/F2MD/f2md-results/LuSTNanoScenario-ITSG5" \
#  --out  "/home/instantf2md/F2MD/f2md-results/features_intermessage_v2.csv" \
#  --version v2
#
#
import os, sys, json, math, argparse
from glob import glob

def ang_norm(d):
    while d > math.pi: d -= 2*math.pi
    while d < -math.pi: d += 2*math.pi
    return d

def mag(x, y): return math.hypot(x, y)

def heading_angle(hvec):
    if not isinstance(hvec, list) or len(hvec) < 2:
        return None
    return math.atan2(hvec[1], hvec[0])

def safe_get(v, i, default=0.0):
    try:
        return float(v[i])
    except:
        return float(default)

def extract_records(obj):
    bp = obj.get("BsmPrint", {})
    meta = bp.get("Metadata", {})
    bsms = bp.get("BSMs", []) or []
    return meta, bsms

def yield_json_files_from_dirs(dirs):
    for d in dirs:
        for f in sorted(glob(os.path.join(d, "*"))):
            if os.path.isdir(f):
                continue
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    txt = fh.read().strip()
                    if not txt:
                        continue
                    yield f, json.loads(txt)
            except Exception:
                # يمكن أن توجد ملفات غير JSON أو تالفة؛ نتجاهلها
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="جذر النتائج (إمّا مجلد الأب الذي يحوي MDBsms_* أو مجلد MDBsms_V1_*/V2_* نفسه)")
    ap.add_argument("--out", required=True, help="مسار ملف CSV الناتج")
    ap.add_argument("--version", default="v1", choices=["v1","v2"], help="اختر أي نسخة تريد استخراجها")
    ap.add_argument("--min-dt", type=float, default=1e-6, help="أصغر فرق زمني لتفادي القسمة على صفر")
    args = ap.parse_args()

    want_v = "V1" if args.version.lower()=="v1" else "V2"

    # حدد المجلدات المرشحة التي سنقرأ منها
    root_basename = os.path.basename(os.path.normpath(args.root))
    if root_basename.startswith("MDBsms_") and os.path.isdir(args.root):
        cand_dirs = [args.root]
    else:
        cand_dirs = glob(os.path.join(args.root, "**", "MDBsms_*"), recursive=True)

    # صفّي على النسخة المطلوبة فقط (V1 أو V2)
    bsm_dirs = [d for d in cand_dirs if f"_{want_v}_" in os.path.basename(d)]

    # إذا المستخدم أعطى مجلد MDBsms_V1/2 صحيح لكنه لا يحتوي العلامة بالشكل المتوقع، لا نمنعه:
    if not bsm_dirs and root_basename.startswith("MDBsms_"):
        bsm_dirs = [args.root]

    if not bsm_dirs:
        print(f"[warn] لم يتم العثور على مجلدات MDBsms_* مطابقة لـ {want_v} تحت: {args.root}")
        print("       تأكد من المسار. إن أردت المسار الأب أعطه مجلد السيناريو مثل: .../LuSTNanoScenario-ITSG5")
        open(args.out, "w", encoding="utf-8").close()
        return

    # تجميع الرسائل حسب (receiverPseudo, senderPseudo)
    buckets = {}
    total = 0

    for fpath, obj in yield_json_files_from_dirs(bsm_dirs):
        meta, bsms = extract_records(obj)
        recv = meta.get("receiverPseudo")
        genT = meta.get("generationTime")
        attack_meta = meta.get("attackType", meta.get("mbType", "Genuine"))

        for b in bsms:
            sender = b.get("Pseudonym") or b.get("RealId")
            if sender is None or recv is None:
                continue

            t = float(b.get("CreationTime", 0.0))
            pos = b.get("Pos", [0,0,0]);    x, y = safe_get(pos,0), safe_get(pos,1)
            spd = b.get("Speed", [0,0,0]);  vx, vy = safe_get(spd,0), safe_get(spd,1)
            acc = b.get("Accel", [0,0,0]);  ax, ay = safe_get(acc,0), safe_get(acc,1)
            hd  = b.get("Heading", [1,0,0]); ang = heading_angle(hd)

            pc = b.get("PosConfidence", [0,0,0]); pcx, pcy = safe_get(pc,0), safe_get(pc,1)
            sc = b.get("SpeedConfidence",[0,0,0]); scx, scy = safe_get(sc,0), safe_get(sc,1)
            ac = b.get("AccelConfidence",[0,0,0]); acx, acy = safe_get(ac,0), safe_get(ac,1)
            hc = b.get("HeadingConfidence",[0,0,0]); hcx, hcy = safe_get(hc,0), safe_get(hc,1)

            label = 0 if (b.get("AttackType","Genuine") == "Genuine" and (attack_meta or "Genuine") == "Genuine") else 1

            rec = dict(
                receiver_pseudo=int(recv),
                sender_pseudo=int(sender),
                creation_time=t,
                x=x, y=y, vx=vx, vy=vy, ax=ax, ay=ay,
                heading=ang if ang is not None else 0.0,
                pos_conf_x=pcx, pos_conf_y=pcy,
                spd_conf_x=scx, spd_conf_y=scy,
                acc_conf_x=acx, acc_conf_y=acy,
                head_conf_x=hcx, head_conf_y=hcy,
                label=label,
                mb_version=args.version,
                meta_generation_time=genT if genT is not None else t
            )
            buckets.setdefault((int(recv), int(sender)), []).append(rec)
            total += 1

    cols = [
        "receiver_pseudo","sender_pseudo","t_prev","t_curr","dt",
        "x_prev","y_prev","x_curr","y_curr","dx","dy","dist",
        "speed_prev","speed_curr","dv","jerk",
        "acc_prev","acc_curr","dacc",
        "heading_prev","heading_curr","dtheta","heading_rate",
        "rate_msgs_per_s",
        "pos_conf_x_curr","pos_conf_y_curr","spd_conf_x_curr","spd_conf_y_curr",
        "acc_conf_x_curr","acc_conf_y_curr","head_conf_x_curr","head_conf_y_curr",
        "label","mb_version"
    ]
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(",".join(cols)+"\n")
        pairs = 0
        for key, lst in buckets.items():
            lst.sort(key=lambda r: r["creation_time"])
            for i in range(1, len(lst)):
                a, b = lst[i-1], lst[i]
                dt = max(args.min_dt, b["creation_time"] - a["creation_time"])
                dx, dy = (b["x"]-a["x"]), (b["y"]-a["y"])
                dist = math.hypot(dx, dy)

                sp_prev = mag(a["vx"], a["vy"])
                sp_curr = mag(b["vx"], b["vy"])
                dv = sp_curr - sp_prev
                jerk = dv / dt

                acc_prev = mag(a["ax"], a["ay"])
                acc_curr = mag(b["ax"], b["ay"])
                dacc = acc_curr - acc_prev

                ang_prev = a["heading"]
                ang_curr = b["heading"]
                dtheta = ang_norm(ang_curr - ang_prev)
                heading_rate = dtheta / dt

                rate = 1.0 / dt

                row = [
                    b["receiver_pseudo"], b["sender_pseudo"],
                    f'{a["creation_time"]:.6f}', f'{b["creation_time"]:.6f}', f'{dt:.6f}',
                    f'{a["x"]:.6f}', f'{a["y"]:.6f}', f'{b["x"]:.6f}', f'{b["y"]:.6f}',
                    f'{dx:.6f}', f'{dy:.6f}', f'{dist:.6f}',
                    f'{sp_prev:.6f}', f'{sp_curr:.6f}', f'{dv:.6f}', f'{jerk:.6f}',
                    f'{acc_prev:.6f}', f'{acc_curr:.6f}', f'{dacc:.6f}',
                    f'{ang_prev:.6f}', f'{ang_curr:.6f}', f'{dtheta:.6f}', f'{heading_rate:.6f}',
                    f'{rate:.6f}',
                    f'{b["pos_conf_x"]:.6f}', f'{b["pos_conf_y"]:.6f}',
                    f'{b["spd_conf_x"]:.6f}', f'{b["spd_conf_y"]:.6f}',
                    f'{b["acc_conf_x"]:.6f}', f'{b["acc_conf_y"]:.6f}',
                    f'{b["head_conf_x"]:.6f}', f'{b["head_conf_y"]:.6f}',
                    b["label"], b["mb_version"]
                ]
                out.write(",".join(map(str,row))+"\n")
                pairs += 1

    print(f"Read {total} BSMs -> wrote {pairs} inter-message rows to {args.out}")

if __name__ == "__main__":
    main()
