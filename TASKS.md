# 현재 작업 분담 (2026-07)

데이터/feature 설명은 [data/README.md](data/README.md) 참고. 아래는 그 데이터로 지금 각자 뭘 하는지에 대한 내용.

## 최적화 (동원) — Phase 3
3. ML 쪽 예측 결과 나오면 헷지 결정에 연결할 준비, 최적화 타겟 명확히

## ML (3명) — 모두 동일한 데이터/split/지표 사용, 모델만 다르게 !!

**은지 — Random Walk + Ridge (기준선)**
- RW(변화 없음으로 예측)로 train/val/test 방향적중률·RMSE 계산 → 팀 전체의 비교 기준점
- Ridge 회귀 학습, val로 정규화 강도 튜닝, test는 마지막에 한 번만 평가

**혜원 — Lasso + ElasticNet (feature 선별)**
- Lasso로 26개 feature 중 실제로 살아남는 게 몇 개인지 확인 → 팀에 "어떤 변수가 진짜 유의미한지" 근거 제공
- ElasticNet과 비교해서 정규화 방식에 따른 성능 차이 확인

**준혁 — LightGBM (비선형/과적합 검증)**
- 과적합이 있다면 정규화(얕은 트리, early stopping 등)로 완화 시도
- 최종적으로 복잡한 모델이 Ridge/Lasso 대비 실제로 나은지 결론

## 마무리
세 명 결과를 한 표로 모아 RW 대비 방향적중률/RMSE 비교 → 제일 성능 좋은 모델(들)을 최적화 담당에게 전달.
