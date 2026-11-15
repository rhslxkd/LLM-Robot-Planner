// Go2BevExporter.cs ─ Unity 실내 씬 → (BEV 이미지, 로봇 높이 점유 지도, 메타) 내보내기
// =====================================================================================
// 설치: 이 파일을 Unity 프로젝트의 Assets/Editor/ 아래에 넣는다. 메뉴 Tools > Go2 BEV Exporter.
//
// 씬 준비
//   1) 빈 GameObject 두 개를 만들고 이름을 Go2Start, Go2Goal로 짓는다.
//      - Go2Start: 바닥 위 로봇 시작 위치. 파란 축(z, forward)이 로봇이 바라보는 방향.
//      - Go2Goal : 목표 위치 (높이는 무시).
//      - 한 씬에서 여러 쌍을 뽑으려면 Go2Start_01 / Go2Goal_01, Go2Start_02 / Go2Goal_02 ... 처럼
//        접미사를 맞춰서 만든다. 쌍마다 별도 실험 씬으로 내보내진다.
//   2) 가구/벽에 Collider가 없어도 "Auto MeshColliders"가 켜져 있으면 내보내는 동안만 임시로 붙였다 뗀다.
//   3) 천장이 있는 씬이면 Camera Height를 천장 아래(가구 위)로 둔다. 카메라 뒤쪽(위쪽)은 렌더되지 않는다.
//
// 출력: <프로젝트>/Go2Export/<scene_name><접미사>/
//   bev.png        IMG_W×IMG_H 정사영 top-down. 화면 오른쪽 = 로봇 전방(+x), 화면 위 = 로봇 왼쪽(+y),
//                  로봇 시작점이 픽셀 robot_px에 찍힘 (oracle_gen.py와 같은 규약)
//   occupancy.png  1 px = cell_m. 흰색 = 점유. PNG 맨 위 행 = y_max, 맨 왼쪽 열 = x_min
//   meta.json      스케일·범위·목표점 등 (Python indoor/build_scene.py가 읽음)
//
// 좌표 변환 (Unity 왼손 좌표계 → 로봇/MuJoCo 오른손 좌표계)
//   fwd  = Go2Start.forward를 수평으로 투영,  left = Cross(fwd, up)
//   로봇 x = dot(p - start, fwd),  로봇 y = dot(p - start, left),  로봇 z = p.y - start.y
// =====================================================================================
#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditorInternal;
using UnityEngine;

[Serializable]
public class Go2ExportMeta
{
    public string scene_name;
    public string unity_scene;
    public string exported_at;
    public int img_w;
    public int img_h;
    public float ppm;
    public float[] robot_px;
    public float cell_m;
    public float x_min, x_max, y_min, y_max;   // 점유 지도가 덮는 범위 (로봇 좌표, m)
    public int occ_w, occ_h;
    public float[] band;                         // [low, high] 로봇 높이 단면 (바닥 기준, m)
    public float[] goal;                         // 로봇 좌표 (m)
    public float camera_height;
    public float[] start_unity_pos;
    public float[] start_unity_fwd;
    public int n_auto_colliders;
}

public class Go2BevExporter : EditorWindow
{
    // ── 실험 규약 (oracle_gen.py / main_court.py와 동일하게 유지) ─────────────
    int imgW = 1263, imgH = 1080;
    float ppm = 90f;
    Vector2 robotPx = new Vector2(421f, 540f);
    float cellM = 0.05f;
    float bandLow = 0.05f, bandHigh = 0.45f;
    float cameraHeight = 2.0f;
    int obstacleMask = ~0;
    bool autoMeshColliders = true;
    string sceneName = "indoor_000";
    string outRoot = "Go2Export";

    Vector3 origin, fwd, left;

    [MenuItem("Tools/Go2 BEV Exporter")]
    static void Open() => GetWindow<Go2BevExporter>("Go2 BEV Exporter");

    void OnGUI()
    {
        EditorGUILayout.LabelField("Image (oracle_gen.py 규약)", EditorStyles.boldLabel);
        imgW = EditorGUILayout.IntField("Width (px)", imgW);
        imgH = EditorGUILayout.IntField("Height (px)", imgH);
        ppm = EditorGUILayout.FloatField("Pixels per meter", ppm);
        robotPx = EditorGUILayout.Vector2Field("Robot pixel (u, v)", robotPx);
        EditorGUILayout.HelpBox($"보이는 범위: 전방 {(imgW - robotPx.x) / ppm:F1} m, 후방 {robotPx.x / ppm:F1} m, " +
                                $"좌우 ±{imgH / 2f / ppm:F1} m", MessageType.None);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Occupancy (GT)", EditorStyles.boldLabel);
        cellM = EditorGUILayout.FloatField("Cell size (m)", cellM);
        bandLow = EditorGUILayout.FloatField("Band low (m)", bandLow);
        bandHigh = EditorGUILayout.FloatField("Band high (m)", bandHigh);
        obstacleMask = EditorGUILayout.MaskField("Obstacle layers", obstacleMask, InternalEditorUtility.layers);
        autoMeshColliders = EditorGUILayout.Toggle("Auto MeshColliders", autoMeshColliders);

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Render", EditorStyles.boldLabel);
        cameraHeight = EditorGUILayout.FloatField("Camera height (m)", cameraHeight);

        EditorGUILayout.Space();
        sceneName = EditorGUILayout.TextField("Scene name", sceneName);
        outRoot = EditorGUILayout.TextField("Output folder", outRoot);

        var pairs = FindPairs();
        EditorGUILayout.HelpBox($"찾은 Start/Goal 쌍: {pairs.Count}개", pairs.Count > 0 ? MessageType.Info : MessageType.Warning);
        GUI.enabled = pairs.Count > 0;
        if (GUILayout.Button($"Export {pairs.Count} pair(s)")) ExportAll(pairs);
        GUI.enabled = true;
    }

    // Go2Start[_suffix] ↔ Go2Goal[_suffix]
    static List<(string suffix, Transform start, Transform goal)> FindPairs()
    {
        var all = FindObjectsByType<Transform>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
        var goals = new Dictionary<string, Transform>();
        foreach (var t in all)
            if (t.name.StartsWith("Go2Goal")) goals[t.name.Substring("Go2Goal".Length)] = t;
        var pairs = new List<(string, Transform, Transform)>();
        foreach (var t in all)
        {
            if (!t.name.StartsWith("Go2Start")) continue;
            var suffix = t.name.Substring("Go2Start".Length);
            if (goals.TryGetValue(suffix, out var g)) pairs.Add((suffix, t, g));
        }
        pairs.Sort((a, b) => string.CompareOrdinal(a.Item1, b.Item1));
        return pairs;
    }

    void ExportAll(List<(string suffix, Transform start, Transform goal)> pairs)
    {
        var added = new List<Collider>();
        var hidden = new List<(Renderer r, bool on)>();
        var disabled = new List<(Collider c, bool on)>();
        try
        {
            // 마커 오브젝트 자체가 렌더/점유에 섞이지 않게 숨김
            foreach (var p in pairs)
                foreach (var t in new[] { p.start, p.goal })
                {
                    foreach (var r in t.GetComponentsInChildren<Renderer>()) { hidden.Add((r, r.enabled)); r.enabled = false; }
                    foreach (var c in t.GetComponentsInChildren<Collider>()) { disabled.Add((c, c.enabled)); c.enabled = false; }
                }
            if (autoMeshColliders) added = AddTemporaryMeshColliders();
            Physics.SyncTransforms();

            for (int i = 0; i < pairs.Count; i++)
            {
                var p = pairs[i];
                var name = sceneName + p.suffix;
                EditorUtility.DisplayProgressBar("Go2 BEV Exporter", name, (float)i / pairs.Count);
                ExportOne(name, p.start, p.goal, added.Count);
            }
            Debug.Log($"[Go2BevExporter] {pairs.Count}개 내보냄 → {Path.GetFullPath(outRoot)}");
        }
        finally
        {
            foreach (var c in added) if (c != null) DestroyImmediate(c);
            foreach (var (r, on) in hidden) if (r != null) r.enabled = on;
            foreach (var (c, on) in disabled) if (c != null) c.enabled = on;
            EditorUtility.ClearProgressBar();
        }
    }

    static List<Collider> AddTemporaryMeshColliders()
    {
        var added = new List<Collider>();
        foreach (var mf in FindObjectsByType<MeshFilter>(FindObjectsInactive.Exclude, FindObjectsSortMode.None))
        {
            if (mf.sharedMesh == null || mf.GetComponent<Collider>() != null) continue;
            if (mf.GetComponent<Renderer>() == null || !mf.GetComponent<Renderer>().enabled) continue;
            var mc = mf.gameObject.AddComponent<MeshCollider>();
            mc.sharedMesh = mf.sharedMesh;
            mc.hideFlags = HideFlags.DontSave;
            added.Add(mc);
        }
        return added;
    }

    Vector3 ToWorld(float x, float y, float z) => origin + fwd * x + left * y + Vector3.up * z;

    Vector2 ToRobot(Vector3 p)
    {
        var d = p - origin;
        return new Vector2(Vector3.Dot(d, fwd), Vector3.Dot(d, left));
    }

    void ExportOne(string name, Transform start, Transform goal, int nAuto)
    {
        origin = start.position;
        fwd = Vector3.ProjectOnPlane(start.forward, Vector3.up).normalized;
        left = Vector3.Cross(fwd, Vector3.up);          // Unity 왼손계에서 (0,0,1)×(0,1,0) = (-1,0,0) = 왼쪽

        var dir = Path.Combine(outRoot, name);
        Directory.CreateDirectory(dir);

        // ── 1) BEV 렌더 ───────────────────────────────────────────────
        var camGo = new GameObject("_Go2BevCam") { hideFlags = HideFlags.HideAndDontSave };
        var cam = camGo.AddComponent<Camera>();
        cam.orthographic = true;
        cam.orthographicSize = imgH / (2f * ppm);
        cam.aspect = (float)imgW / imgH;
        cam.nearClipPlane = 0.01f;
        cam.farClipPlane = cameraHeight + 5f;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        float cx = (imgW / 2f - robotPx.x) / ppm;        // 화면 중앙의 로봇 좌표
        float cy = (robotPx.y - imgH / 2f) / ppm;
        camGo.transform.position = ToWorld(cx, cy, cameraHeight);
        camGo.transform.rotation = Quaternion.LookRotation(Vector3.down, left);  // 화면 위 = 로봇 왼쪽, 화면 오른쪽 = 전방

        var rt = new RenderTexture(imgW, imgH, 24, RenderTextureFormat.ARGB32) { antiAliasing = 4 };
        var tex = new Texture2D(imgW, imgH, TextureFormat.RGB24, false);
        try
        {
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, imgW, imgH), 0, 0);
            tex.Apply();
            File.WriteAllBytes(Path.Combine(dir, "bev.png"), tex.EncodeToPNG());
        }
        finally
        {
            RenderTexture.active = null;
            cam.targetTexture = null;
            DestroyImmediate(rt);
            DestroyImmediate(tex);
            DestroyImmediate(camGo);
        }

        // ── 2) 로봇 높이 단면 점유 지도 ────────────────────────────────
        float xMin = -robotPx.x / ppm, xMaxView = (imgW - robotPx.x) / ppm;
        float yMin = (robotPx.y - imgH) / ppm, yMaxView = robotPx.y / ppm;
        int nx = Mathf.CeilToInt((xMaxView - xMin) / cellM);
        int ny = Mathf.CeilToInt((yMaxView - yMin) / cellM);
        var occ = new Texture2D(nx, ny, TextureFormat.RGB24, false);
        var orient = Quaternion.LookRotation(fwd, Vector3.up);
        var half = new Vector3(cellM / 2f, (bandHigh - bandLow) / 2f, cellM / 2f);
        float zMid = (bandLow + bandHigh) / 2f;
        int nOcc = 0;
        for (int r = 0; r < ny; r++)
        {
            float y = yMin + (r + 0.5f) * cellM;
            for (int c = 0; c < nx; c++)
            {
                float x = xMin + (c + 0.5f) * cellM;
                bool hit = Physics.CheckBox(ToWorld(x, y, zMid), half, orient, obstacleMask, QueryTriggerInteraction.Ignore);
                occ.SetPixel(c, r, hit ? Color.white : Color.black);   // Texture 원점 = 좌하단 → PNG 맨 위 = y_max
                if (hit) nOcc++;
            }
        }
        occ.Apply();
        File.WriteAllBytes(Path.Combine(dir, "occupancy.png"), occ.EncodeToPNG());
        DestroyImmediate(occ);

        // ── 3) 메타 ───────────────────────────────────────────────────
        var g = ToRobot(goal.position);
        var meta = new Go2ExportMeta
        {
            scene_name = name,
            unity_scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().path,
            exported_at = DateTime.Now.ToString("s"),
            img_w = imgW, img_h = imgH, ppm = ppm,
            robot_px = new[] { robotPx.x, robotPx.y },
            cell_m = cellM,
            x_min = xMin, x_max = xMin + nx * cellM, y_min = yMin, y_max = yMin + ny * cellM,
            occ_w = nx, occ_h = ny,
            band = new[] { bandLow, bandHigh },
            goal = new[] { g.x, g.y },
            camera_height = cameraHeight,
            start_unity_pos = new[] { origin.x, origin.y, origin.z },
            start_unity_fwd = new[] { fwd.x, fwd.y, fwd.z },
            n_auto_colliders = nAuto,
        };
        File.WriteAllText(Path.Combine(dir, "meta.json"), JsonUtility.ToJson(meta, true));
        if (nOcc == 0)
            Debug.LogWarning($"[Go2BevExporter] {name}: 점유 셀 0개 — Collider/레이어/Band 높이를 확인하세요.");
    }
}
#endif
