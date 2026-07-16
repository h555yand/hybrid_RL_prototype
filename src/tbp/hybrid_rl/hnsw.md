Проанализировал код. Вот почему Q-store не улучшается на вазе и что можно сделать.

## Проблемы

### 1. Нормализация заморожена от кружки/чашки

```python
store._norm_frozen = True
store._freeze_done = True
```

При загрузке Q-store нормализация **замораживается**. `state_mean` и `state_std` от кружки/чашки. Состояния вазы нормализуются **неправильно** — другая геометрия, другие кривизны, другие расстояния.

В `_run_adaptive` мы не размораживаем нормализацию (в отличие от `_run_train` где `unfreeze_normalization=True`).

**Fix:** разморозить нормализацию при adaptive:

```python
# В _run_adaptive, после загрузки контроллера:
controller.q_store_free._norm_frozen = False
controller.q_store_free._freeze_done = False
controller.q_store_surface._norm_frozen = False
controller.q_store_surface._freeze_done = False
```

### 2. Старые точки доминируют через kNN

Q-store содержит ~270K free + ~137K surface точек от кружки/чашки. На вазе kNN находит соседей из этих точек. Новые точки от вазы вставляются, но их мало (~1000 за 1000 эпизодов при `insert_threshold=0.5`).

При `k_neighbors=7` — 7 соседей, из которых 6-7 от кружки и 0-1 от вазы. Интерполяция даёт Q-values от кружки.

**Fix варианты:**

a) **Уменьшить k_neighbors** для adaptive — `k=3` вместо 7. Меньше соседей → меньше влияние старых точек.

b) **Recency-weighted interpolation** — добавить вес по `last_step`. Свежие точки (от вазы) получают больший вес:

```python
# В get_q_values, после kernel weights:
for i, label in enumerate(labels):
    point = self.points[label]
    recency = 1.0 / (self.global_step - point.last_step + 1)
    recency_weight = min(recency * 1000, 1.0)  # normalize
    weights[i] *= recency_weight
```

c) **Уменьшить insert_threshold** — 0.5 → 0.3. Больше новых точек вставляется вместо обновления старых.

### 3. Confidence-weighted interpolation подавляет новые знания

```python
q_values *= confidence
```

`confidence = min(weight_sum / k, 1.0)`. Для новых точек вазы — соседи далеко → weight_sum маленький → confidence маленький → Q-values масштабируются вниз. Q-store "не уверен" в новых знаниях и занижает их.

**Fix:** убрать confidence scaling для update (оставить для get_q_values):

Или использовать confidence только для арбитража, не для Q-values.

### 4. Q-update обновляет чужие точки

Когда агент на вазе в состоянии S, ближайший сосед — точка от кружки. `update_q_value` обновляет **эту точку кружки** с reward от вазы. Q-values точки кружки портятся — она теперь содержит смесь опыта кружки и вазы.

```python
if distances[0] < self.insert_threshold ** 2:
    point = self.points[labels[0]]
    point.q_values[action] += alpha * (td_target - point.q_values[action])
```

**Fix:** вставлять новую точку вместо обновления старой, если старая точка "слишком старая" (от другого объекта):

```python
if distances[0] < self.insert_threshold ** 2:
    point = self.points[labels[0]]
    age = self.global_step - point.last_step
    if age > 10000:  # точка от другого объекта
        # Вставить новую вместо обновления старой
        self._insert_point(state, norm_state, action, td_target, alpha)
    else:
        point.q_values[action] += alpha * (td_target - point.q_values[action])
```

## Рекомендация: начать с fix 1 (разморозка нормализации)

Это самое простое и может дать значительный эффект. Нормализация пересчитается на данных вазы, HNSW индекс перестроится, расстояния станут корректными.

Хочешь применить fix 1 и перезапустить?




Отличные вопросы. Давай разберём.

## 1) Зачем freeze нормализации?

Нормализация пересчитывается на первых `norm_warmup_steps=5000` шагах. Потом замораживается чтобы:

- **Стабильность HNSW расстояний.** Если mean/std меняются — расстояния между точками меняются — kNN находит других соседей — Q-values скачут.
- **Не нужен rebuild.** При изменении нормализации нужно пересчитать `norm_state` для всех точек и перестроить индекс. Дорого при 400K точек.

Freeze правильный для одного объекта. Для multi-mesh — нормализация от первого объекта может не подходить для последующих. Но мы размораживаем при `unfreeze_normalization=True` в train.

**Для adaptive:** нормализация от кружки/чашки. State vector содержит `local_pos_error`, `rot_error`, `local_normal`, `curvature`, `alignment`, `distance`, `depth`. Эти величины имеют **похожие** диапазоны на вазе и кружке — позиция в mm, углы в градусах, нормали [-1,1]. Нормализация от кружки **должна подходить** для вазы.

**Вывод:** freeze нормализации скорее всего **не проблема** для вазы.

## 2) Старые точки — что плохого?

Ты прав. State vector **не зависит от объекта**:

```
local_pos_error  [3D]  — относительное положение цели
rot_error        [3D]  — ошибка ориентации
local_normal     [3D]  — нормаль поверхности в локальных координатах
mean_curvature   [1D]  — кривизна
gauss_curvature  [1D]  — гауссова кривизна
on_object        [1D]  — на поверхности или нет
alignment        [1D]  — dot(direction_to_goal, normal)
distance         [1D]  — расстояние до цели
norm_depth       [1D]  — глубина до поверхности
```

Если агент на вазе в состоянии "на цилиндрической стенке, цель в 20mm впереди, alignment=0.8" — это **то же самое** состояние что на кружке. Q-values от кружки **должны быть правильными**.

Отличия вазы:
- Горлышко — маленький радиус, высокая кривизна
- Переход тело→горлышко — резкое изменение кривизны

Но эти отличия **уже закодированы** в state через `mean_curvature` и `gauss_curvature`. kNN найдёт точки с похожей кривизной.

**Вывод:** старые точки от кружки **полезны** на вазе. Проблема не в том что они старые, а в чём-то другом.

## 3) Как обновляется Q-store в adaptive?

В `_run_adaptive`:

```python
action_index, source = manager.get_action(state, pose, sensor)
_st, done = controller.update_only(pose, sensor, action_index)
```

`update_only` получает `action_index` — это действие выбранное **арбитратором** (может быть от SAC). Внутри:

```python
self._last_action = action_index
...
prev_store.update_q_value(
    self._prev_state, self._last_action, td_target, lr
)
```

Q-store обновляет Q-value для действия которое **SAC выбрал**. Это off-policy Q-learning — корректно по теории. Q-store учит: "в этом состоянии действие X (выбранное SAC) дало reward Y".

Но проблема: SAC выбирает **непрерывные** параметры, которые дискретизируются через `sac_to_discrete`. SAC может выбрать MoveLinear с шагом 12.3mm → дискретизируется в `free_forward` (action 8). Q-store обновляет Q(state, free_forward). Но Q-store сам выбирает `free_forward` с фиксированным шагом 8mm — **другой** результат чем SAC с 12.3mm.

**Вывод:** Q-store учится на действиях SAC, но не может их воспроизвести с той же точностью. Это фундаментальное ограничение дискретных действий.

## 4) State не зависит от объекта — тогда почему Q-store не улучшается?

Ты прав — state универсальный. Тогда Q-store от кружки **должен** работать на вазе для похожих состояний. И он работает — `q_success_rate = 0.38` стабильно.

**Почему не растёт выше 0.38-0.47?**

Возможные причины:

**a) Дискретные действия.** Q-store выбирает из 21 дискретного действия. SAC выбирает непрерывные параметры. На вазе (узкое горлышко) нужна точность — дискретные шаги 3mm/8mm слишком грубые.

**b) Q-store обновляет Q-values для SAC-действий.** Когда SAC доминирует (66%), Q-store обновляет Q-values для действий SAC. Но когда Q-store сам выбирает — он выбирает дискретные версии этих действий. Результат отличается → Q-values неточные.

**c) Exploration ограничен.** `epsilon=0.1` — Q-store в 90% случаев выбирает greedy. Мало исследования новых действий на вазе.

**d) Q-store используется только 34% шагов.** Остальные 66% — SAC. Q-store получает мало обратной связи от своих собственных решений.

**Главный вывод:** Q-store не улучшается не из-за бага, а из-за **фундаментального ограничения** — дискретные действия не могут конкурировать с непрерывными на задачах требующих точности. Q-store стабильно даёт 38% — это его потолок с дискретными шагами.

Для улучшения Q-store нужно менять архитектуру — например Q-store выбирает **тип** действия, а параметры берутся из SAC. Но это большое изменение.

Что думаешь?


Хороший вопрос. Давай разберём.

## Когда нормализация от старого объекта подходит?

State vector:
```
local_pos_error  [3D]  — mm, диапазон ±120mm
rot_error        [3D]  — degrees, диапазон ±180°
local_normal     [3D]  — unit vector, диапазон [-1, 1]
mean_curvature   [1D]  — зависит от объекта
gauss_curvature  [1D]  — зависит от объекта
on_object        [1D]  — 0 или 1
alignment        [1D]  — [-1, 1]
distance         [1D]  — mm, диапазон 0-120mm
norm_depth       [1D]  — [0, 1]
```

Из 15 измерений **13 имеют одинаковые диапазоны** на любом объекте. Только `mean_curvature` и `gauss_curvature` зависят от геометрии.

Кружка: curvature ~5-7 (цилиндр radius=30mm).
Ваза горлышко: curvature ~15-20 (radius=12mm).
Ваза тело: curvature ~3-4 (radius=35mm).

Если нормализация от кружки: `std_curvature ≈ 5`. Ваза горлышко: `(20 - mean) / 5 = 3-4 sigma` — выброс. kNN расстояние будет большим из-за кривизны → соседи не найдутся → Q-store вернёт нули.

## Когда размораживать?

**Автоматически определить** — сравнить текущие состояния с нормализацией:

```python
def _should_unfreeze(self, state: np.ndarray) -> bool:
    """Check if current state is outside normalization range."""
    norm_state = self._normalize(state)
    # Если нормализованное состояние > 3 sigma по любому измерению
    return np.any(np.abs(norm_state) > 3.0)
```

Но это сложно — один выброс не значит что нужно размораживать.

**Проще:** считать статистику отклонений и размораживать если много выбросов:

```python
def _check_normalization_health(self, state: np.ndarray):
    """Track how well current normalization fits new data."""
    norm_state = self._normalize(state)
    max_deviation = float(np.max(np.abs(norm_state)))
    self._deviation_history.append(max_deviation)
    
    if len(self._deviation_history) >= 100:
        mean_dev = np.mean(self._deviation_history)
        if mean_dev > 3.0 and self._norm_frozen:
            logger.info(
                "Normalization mismatch detected: "
                "mean_deviation=%.2f, unfreezing",
                mean_dev,
            )
            self._norm_frozen = False
            self._freeze_done = False
```

## Рекомендация

Для текущего эксперимента — **не размораживать**. 13 из 15 измерений одинаковые, кривизна отличается но не критично. Q-store работает на 38% — это не из-за нормализации.

Для будущего — добавить `_check_normalization_health` как диагностику. Если mean deviation > 3.0 — логировать warning. Решение о разморозке — вручную или автоматически.

Что думаешь? Двигаемся к следующему вопросу или хочешь реализовать health check?


Отличный вопрос. Давай разберём.

## Как считается reward в Q-learning vs SAC

### Q-learning (в `_compute_reward` контроллера):

```python
reward += progress / surface_step * reward_progress  # progress/3 * 3 = progress
reward += reward_step_penalty                         # -0.5
reward += reward_goal_reached                         # +60 при успехе
reward += reward_surface_violation                    # -12 при коллизии
reward += reward_timeout                              # -12 при timeout
```

Типичный reward за шаг: progress 3mm → +3.0 - 0.5 = **+2.5**
Типичный reward за успешный эпизод (30 шагов): +60 + 30×2.5 = **+135**

### SAC (в `compute_reward` PSACTrainer):

```python
reward += progress / surface_step * reward_progress  # то же самое
reward += reward_step * sub_steps                     # -0.5 * sub_steps (SMDP)
reward += reward_goal                                 # +60
reward += reward_collision                            # -12
reward += reward_timeout                              # -12
```

**Те же формулы!** Reward одинаковый для Q-learning и SAC.

## Но Q-values разные

### Q-learning Q-values:

Q(s,a) = expected cumulative reward. При gamma=0.95 и 30 шагов до цели:

```
Q ≈ 60 + sum(2.5 * 0.95^t for t in range(30)) ≈ 60 + 40 = 100
```

Q-values в Q-store: `q_magnitude_mean ≈ 3-5`, `q_spread ≈ 5-8`. Это **маленькие** значения — Q-store интерполирует через kNN, confidence scaling уменьшает.

### SAC Q-values:

SAC critic предсказывает Q-values другого масштаба — зависит от обучения. Но SAC Q-values **не используются** в Q-store update.

## Как Q-store обновляется в adaptive

```python
# В update_only:
if done:
    td_target = reward  # терминальный: просто reward
else:
    next_q = next_store.get_q_values(state)
    td_target = reward + self.gamma * np.max(next_q)

prev_store.update_q_value(
    self._prev_state, self._last_action, td_target, lr
)
```

`td_target` зависит от `next_q` — Q-values из **Q-store**, не из SAC critic. SAC critic не участвует в Q-store update.

**Проблема:** `self._last_action` — действие от SAC (дискретизированное). Q-store обновляет Q(state, sac_action) с td_target на основе **своих** Q-values. Это корректный off-policy Q-learning.

## Тогда почему Q-store не растёт?

Reward одинаковый, td_target корректный. Проблема не в масштабе reward.

**Реальная проблема:** Q-store обновляет Q-value для действия SAC. Но когда Q-store сам выбирает — он выбирает **дискретную** версию. Результат отличается:

```
SAC выбирает: MoveLinear(12.3mm) → progress 12mm → reward +12
Q-store запоминает: Q(state, free_forward) += α * (12 - Q)

Потом Q-store выбирает: free_forward(8mm) → progress 8mm → reward +8
Q-store ожидал reward 12, получил 8 → Q снижается
```

Q-values осциллируют — SAC даёт высокий reward, Q-store не может воспроизвести → Q снижается → SAC снова даёт высокий → Q повышается → цикл.

## Что можно сделать

### Вариант A: Q-store обновляет только свои действия

```python
# В update_only, добавить проверку:
if source == "q_store":  # только если Q-store выбрал действие
    prev_store.update_q_value(...)
```

Q-store учится только на своём опыте. Не портится от SAC-действий.

**Минус:** Q-store учится медленнее — только 34% шагов.

### Вариант B: Q-store обновляет все, но с разным alpha

```python
if source == "q_store":
    lr = self.alpha  # 0.1 — полный
else:
    lr = self.alpha * 0.3  # 0.03 — осторожнее для чужих действий
```

### Вариант C: Не обновлять Q для действий которые Q-store не может воспроизвести

Проверять — совпадает ли SAC-действие с тем что Q-store рекомендовал:

```python
q_action = np.argmax(store.get_q_values(state))
if action_index == q_action:
    # Q-store согласен — обновлять с полным alpha
    lr = self.alpha
else:
    # Q-store не согласен — обновлять осторожно
    lr = self.alpha * 0.1
```

Что думаешь? Какой вариант ближе?


Давай разберём. У Q-store есть 21 действие:

```
8 surface directions (3mm шаг)
free_forward (8mm)
free_forward_small (2mm)
free_backward (8mm)
look_up/down (5°)
turn_left/right (5°)
rotate_sensor ±
orient_horizontal/vertical
detach
detach_edge
```

## Что нужно для успеха на вазе

1. **Добраться до цели** — ползание + detach + полёт
2. **Попасть в 2mm** — финальная точность

## Анализ из логов adaptive

Из ep 2000:
```
q_success_rate: 0.38
sac_success_rate: 0.54
q_store_rate: 30.6%
```

Q-store выбирается 31% шагов и успешен в 38% эпизодов. SAC — 69% шагов и 54% успеха.

**Но мы не знаем:**
- В каких **ситуациях** Q-store выбирается (близко/далеко, на поверхности/в воздухе)?
- Какие **действия** Q-store выбирает vs SAC?
- Где Q-store **ошибается** — коллизия, timeout, или не попадает в 2mm?

## Что нужно для диагностики

Добавить в арбитратор per-source action tracking:

```python
# В decide(), после выбора:
if q_score >= sac_score:
    self.q_chosen_actions[q_name] += 1
    # Уже есть
```

Это уже есть. Но нам нужно **per-source termination tracking** — почему Q-store эпизоды неуспешны:

- Q-store эпизод → timeout (не дошёл за 150 шагов)
- Q-store эпизод → collision (врезался)
- Q-store эпизод → success но не попал в 2mm

## Гипотезы почему Q-store не улучшается

### Гипотеза A: Q-store не попадает в 2mm (финальная точность)

`surface_step=3mm`, `goal_threshold=2mm`. Агент ползает шагами 3mm — может проскочить мимо цели. `free_forward_small=2mm` — но это для воздуха, не для поверхности.

**Проверка:** посмотреть `final_distance` для Q-store эпизодов. Если часто 2-5mm — проблема в точности.

### Гипотеза B: Q-store выбирает неоптимальные повороты

Q-store поворачивает на 5° за шаг. SAC поворачивает на точный угол. На вазе (горлышко) нужны точные повороты — Q-store тратит 5-10 шагов на наведение, SAC — 1-2.

**Проверка:** посмотреть долю look/turn в Q-store vs SAC действиях.

### Гипотеза C: Q-store спамит detach

Мы видели detach spam раньше. Q-store может тратить шаги на бесполезные detach.

**Проверка:** посмотреть долю detach в Q-store chosen actions.

### Гипотеза D: Q-store выбирается в сложных ситуациях

Арбитратор выбирает Q-store когда `q_confidence > 0.2 AND q_spread > 0.5 AND q_score > sac_score`. Это может быть в ситуациях где Q-store **уверен но неправ** — знакомое состояние от кружки, но на вазе нужно другое действие.

## Как проверить

Добавить в `_run_adaptive` tracking per-source:

```python
# После определения success/collision/timeout:
if dominant_source == "q_store":
    q_terminations[termination] = q_terminations.get(termination, 0) + 1
elif dominant_source == "sac":
    sac_terminations[termination] = sac_terminations.get(termination, 0) + 1
```

И в snapshot:

```python
"per_source_terminations": {
    "q_store": dict(q_terminations),
    "sac": dict(sac_terminations),
}
```

Также полезно: `q_chosen_actions` и `sac_chosen_actions` из арбитратора — уже есть в `get_stats()`.

Хочешь добавить per-source termination tracking и перезапустить? Или сначала посмотреть `q_chosen_top` и `sac_chosen_top` из текущих данных?

### Что это покажет

| Метрика | Гипотеза | Что ищем |
|---|---|---|
| `near_miss_count` (2-5mm) | A: не попадает в 2mm | Много near miss → проблема точности |
| `q_chosen_actions` | B: неоптимальные повороты | Много look/turn → тратит шаги |
| `q_chosen_actions` detach % | C: detach spam | Высокий detach % → спам |
| `q_terminations` | D: сложные ситуации | Много collision → Q ошибается |
| `mean_final_distance` | Общая | Q vs SAC — кто ближе к цели |
| `mean_episode_steps` | Эффективность | Q vs SAC — кто быстрее |


Отличные данные! Теперь видно причины.

## Per-source анализ (ep 100)

### Q-store vs SAC

| Метрика | Q-store | SAC |
|---|---|---|
| Эпизодов | 32 | 68 |
| Success | 14 (44%) | 26 (38%) |
| Collision | 7 (22%) | **32 (47%)** |
| Timeout | 11 (34%) | 10 (15%) |
| Mean final distance | **8.3mm** | 19.9mm |
| Mean episode steps | **107.8** | 49.5 |
| Near miss (2-5mm) | **0** | 1 |

### Ключевые находки

**1. Q-store ближе к цели но не попадает в 2mm**

`mean_final_distance: 8.3mm` — Q-store доводит до 8mm но не может попасть в 2mm. **Zero near misses** — Q-store даже не приближается к 2-5mm зоне. Проблема не в финальной точности, а в том что Q-store **застревает** на 8mm.

**Гипотеза A подтверждена частично** — но проблема не в шаге 3mm vs 2mm threshold, а в том что Q-store не может навигировать последние 8mm.

**2. SAC далеко от цели но быстрый**

`mean_final_distance: 19.9mm` — SAC не доходит до цели. Но `mean_episode_steps: 49.5` — SAC быстро завершает эпизоды (коллизия или timeout). **47% коллизий** — SAC врезается на вазе.

**3. Q-store тратит шаги на повороты**

Q-store chosen actions:
```
move_tangentially: 24%
look_up: 17%
look_down: 15%
free_forward_small: 13%
turn_left: 13%
```

**45% шагов — повороты** (look + turn). Q-store тратит почти половину шагов на наведение. SAC: look+turn = ~6%.

**Гипотеза B подтверждена** — Q-store неэффективен из-за дискретных поворотов 5°.

**4. SAC больше ползает и использует detach**

SAC chosen: `move_tangentially: 71%`, `detach: 5%`. Q-store: `move_tangentially: 24%`. SAC ползает эффективнее — непрерывные углы.

**5. Q-store предлагает move_tangentially 46% но выбирается только 24%**

`proposed: move_tangentially 46%` vs `chosen: 24%`. Арбитратор выбирает Q-store когда Q-score > SAC-score. Но Q-store часто предлагает ползание, а SAC предлагает что-то другое с более высоким score → SAC побеждает на ползании.

## Вывод

**Главная проблема Q-store: дискретные повороты.** 45% шагов на look/turn по 5° — неэффективно. SAC делает точный поворот за 1 шаг. Q-store тратит 5-10 шагов на то же самое.

**Вторая проблема: Q-store застревает на 8mm.** Не может сделать финальный подход. Возможно из-за тех же поворотов — не может точно навестись на цель с 8mm.

## Анализ ep 500 — подтверждение гипотез

### Per-source сравнение

| Метрика | Q-store | SAC |
|---|---|---|
| Эпизодов | 141 | 359 |
| Success rate | 37% (52/141) | 41% (146/359) |
| Collision rate | **27%** | **40%** |
| Timeout rate | **36%** | 19% |
| Mean final distance | **9.4mm** | 16.6mm |
| Near miss (2-5mm) | **8** | 6 |
| Mean episode steps | **102.7** | 55.9 |

### Подтверждённые гипотезы

**Гипотеза B: Q-store тратит шаги на повороты — ПОДТВЕРЖДЕНА**

Q-store chosen actions:
```
move_tangentially: 22%
look_up+down: 32%
turn_left+right: 27%
```

**59% шагов — повороты.** Q-store тратит 60 из 103 шагов на повороты по 5°. SAC: look+turn = ~10%.

**Гипотеза A: Q-store не попадает в 2mm — ПОДТВЕРЖДЕНА**

`mean_final_distance: 9.4mm`. 8 near misses (2-5mm) из 141 эпизодов = 5.7%. Q-store доводит до ~9mm но не может финишировать.

**Гипотеза D: Q-store collision rate ниже SAC**

Q-store: 27% collision vs SAC: 40%. Q-store **безопаснее** — дискретные шаги осторожнее. Но timeout 36% vs SAC 19% — Q-store медленнее.

### Главный вывод

**Q-store проблема — не в Q-values, а в дискретных действиях.** Q-store знает куда идти (mean_final_distance снижается), но не может эффективно исполнить:

1. **59% шагов на повороты** — 5° за шаг, нужно 10-20 шагов для разворота
2. **Не может финишировать** — surface_step=3mm при threshold=2mm, нет точного наведения
3. **Timeout 36%** — тратит 103 шага и не успевает

SAC решает это непрерывными параметрами — точный поворот за 1 шаг, точный шаг любого размера.

### Для улучшения Q-store

Нужно не менять HNSW/kNN, а **изменить как Q-store использует действия**:

1. **Q-store выбирает тип, SAC выбирает параметры** — гибридный подход
2. **Или увеличить rotation_step** для Q-store — 10-15° вместо 5°
3. **Или добавить macro-action "повернуться к цели"** — один шаг вместо 10

Жду ep 1000 или хочешь обсудить улучшения?


## Финальный анализ adaptive на вазе (2000 эпизодов)

### Per-source итоги

| Метрика | Q-store | SAC |
|---|---|---|
| Эпизодов | 468 (23%) | 1532 (77%) |
| Success rate | **39%** (184/468) | **44%** (678/1532) |
| Collision rate | **21%** | **32%** |
| Timeout rate | **40%** | **24%** |
| Mean final distance | **9.2mm** | **19.4mm** |
| Near miss (2-5mm) | 20 (4.3%) | 35 (2.3%) |
| Mean episode steps | **103** | **63** |
| Повороты (look+turn) % | **58%** | **14%** |

### Подтверждённые выводы

**1. Q-store ближе к цели но не финиширует**

Q-store: final_distance=9.2mm. SAC: 19.4mm. Q-store доводит **в 2 раза ближе**, но не попадает в 2mm. 20 near misses из 468 эпизодов (4.3%) — редко приближается к 2-5mm.

**2. Q-store тратит 58% шагов на повороты**

```
look_up: 18% + look_down: 16% + turn_left: 13% + turn_right: 11% = 58%
move_tangentially: 22%
```

SAC: повороты 14%, ползание 62%. Q-store в **4 раза** больше крутится.

**3. Q-store безопаснее но медленнее**

Collision: Q 21% vs SAC 32%. Timeout: Q **40%** vs SAC 24%. Q-store осторожен (мало коллизий) но не успевает за 150 шагов (103 шага в среднем, 40% timeout).

**4. SAC collision rate высокий на вазе**

SAC: 32% collision, 489 из 1532 эпизодов. На тренировочных объектах SAC collision <8%. Ваза — новая геометрия, SAC врезается.

### Главный вывод для Q-store

**Проблема Q-store — не HNSW, не kNN, не Q-values. Проблема — дискретные повороты 5°.**

Q-store знает куда идти (final_distance 9.2mm — ближе чем SAC). Но тратит 60 шагов из 103 на повороты. Если бы Q-store мог повернуться за 1 шаг — success rate был бы значительно выше.

### Рекомендации для улучшения Q-store

**Быстрая победа:** Q-store выбирает **тип** действия, SAC выбирает **параметры**. Q-store говорит "поверни налево", SAC выбирает "на 23.7°". Лучшее из обоих миров.

Хочешь обсудить эту архитектуру?

