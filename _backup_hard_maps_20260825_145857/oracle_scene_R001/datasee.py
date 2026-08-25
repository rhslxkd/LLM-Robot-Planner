import glob, os, numpy as np

for label, d in [("GT", "data/oracle_scene_R001/dial_gt"), ("NA", "data/oracle_scene_R001/dial_na")]:
    print(f"\n=== {label}: {d} ===")
    if not os.path.isdir(d):
        print("  디렉토리 없음")
        continue
    print("  파일 목록:", os.listdir(d))
    states_files = glob.glob(os.path.join(d, "*_states.npy"))
    if not states_files:
        print("  *_states.npy 없음 -- 실행이 끝까지 안 갔을 가능성")
        continue
    for sf in states_files:
        arr = np.load(sf, allow_pickle=True)
        print(f"  {os.path.basename(sf)} shape={arr.shape} dtype={arr.dtype}")
        try:
            first = arr[0] if arr.ndim > 1 else arr[0]
            last = arr[-1] if arr.ndim > 1 else arr[-1]
            print("  first row (앞 10개 값):", np.array(first).flatten()[:10])
            print("  last  row (앞 10개 값):", np.array(last).flatten()[:10])
            print("  NaN 포함 여부:", bool(np.isnan(np.array(arr, dtype=float)).any()))
        except Exception as e:
            print("  요약 실패:", e)