# Go2Export 씬 점검 (2026-11-16, 점유 높이 띠 0.05~1.0 m)

책상·식탁 상판까지 장애물로 넣은 지도 기준. 쌍 조건: 시작·목표 여유 ≥ 0.6 m, 같은 연결 영역, 직선이 (0.441 m 팽창) 장애물에 막힘, A* 경로 존재, 최단 경로 주변 0.6 m를 막고도 1.5배 이내 두 번째 경로, 목표가 이미지 안.

| 씬 | 용도 | 시작/목표 여유 (m) | 직선 (m) | A* (m) | 두 번째/최단 | 판정 |
|---|---|---|---|---|---|---|
| replica_apartment_0_p0 | 평가 | 0.87 / 0.60 | 8.3 | 9.7 | 1.06 | OK |
| replica_apartment_0_p2 | 평가 | 0.87 / 0.94 | 7.1 | 8.6 | 1.01 | OK |
| replica_apartment_2_p0 | 평가 | 0.70 / 0.60 | 5.8 | 6.3 | 1.19 | OK |
| replica_frl_apartment_0_p0 | 평가 | 0.68 / 0.64 | 5.3 | 6.1 | 1.10 | OK |
| replica_frl_apartment_1_p0 | 개발용(평가 제외) | 0.56 / 0.65 | 5.7 | 6.4 | 1.10 | ❌ |
| replica_frl_apartment_3_p0 | 평가 | 0.66 / 0.63 | 7.8 | 8.9 | 1.05 | OK |
| replica_frl_apartment_3_p1 | 평가 | 0.63 / 0.90 | 6.5 | 7.3 | 1.10 | OK |
| replica_frl_apartment_4_p0 | 평가 | 0.60 / 0.70 | 4.0 | 4.8 | 1.10 | OK |
| replica_frl_apartment_5_p0 | 평가 | 0.81 / 0.95 | 4.1 | 4.8 | 1.29 | OK |
| replica_office_4_p0 | 개발용(평가 제외) | 0.67 / 0.90 | 4.6 | 5.6 | inf | ❌ |

변경 이력 (0.45 m 띠 → 1.0 m 띠)
- office_4 p0·p1, frl_apartment_5 p0: 두 번째 경로가 책상 밑을 지나는 길이었음 → 조건 불만족. office_4는 새 지도에서 조건을 만족하는 쌍이 없음(seed 8개). office_4 p1 제외.
- frl_apartment_5 p0: 새 지도에서 다시 고른 쌍으로 교체. frl_apartment_4 p0 추가(새 지도에서 유일하게 조건을 만족한 추가 후보; apartment_1, frl_2, hotel_0, office_0~3, room_0~2는 0쌍).
- frl_apartment_0 p0 목표, frl_apartment_3 p0 목표 / p1 시작: 여유가 0.54~0.58 m로 줄어 5~10 cm 이동.
- 개발용 2쌍은 이미지(시작점)를 그대로 둠: 프롬프트 기준 설정에만 쓰고 평가에 쓰지 않음.
- 평가 8쌍 = 공간 3개(apartment_0, apartment_2, FRL apartment), 배치 6개(apartment_0, apartment_2, frl_0, frl_3, frl_4, frl_5).
