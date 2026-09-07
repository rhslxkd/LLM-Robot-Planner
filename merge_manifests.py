import pandas as pd

v2 = pd.read_csv("data/random_batch_manifest_v2.csv")
v3 = pd.read_csv("data/random_batch_manifest_v3.csv")
retry = pd.read_csv("data/random_batch_manifest_v3_retry.csv")

print(f"v2: {len(v2)}, v3: {len(v3)}, retry: {len(retry)}")

# retry 결과를 scene 기준으로 v3에 반영 (dial_mpc_ok, elapsed_s만 덮어씀)
v3 = v3.set_index("scene")
retry_idx = retry.set_index("scene")
common = v3.index.intersection(retry_idx.index)
print(f"retry로 갱신되는 씬 수: {len(common)} (기대값: 74)")

v3.loc[common, "dial_mpc_ok"] = retry_idx.loc[common, "dial_mpc_ok"]
v3.loc[common, "elapsed_s"] = retry_idx.loc[common, "elapsed_s"]
v3 = v3.reset_index()

combined = pd.concat([v2, v3], ignore_index=True)
combined.to_csv("data/random_batch_manifest_combined.csv", index=False)

ok_mask = combined["dial_mpc_ok"].astype(str) == "True"
print(f"combined 총 씬: {len(combined)}")
print(f"generator_passed=True: {(combined['generator_passed'].astype(str) == 'True').sum()}")
print(f"dial_mpc_ok=True: {ok_mask.sum()}")
print("저장: data/random_batch_manifest_combined.csv")
