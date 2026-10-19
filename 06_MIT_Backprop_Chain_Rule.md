# Конспект 6. Backpropagation и chain rule

**Источник:** MIT OpenCourseWare 18.065 — Lecture 27, *Backpropagation: Find Partial Derivatives*  
<https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/resources/lecture-27-backpropagation-find-partial-derivatives/>

## 1. Что такое backpropagation

Backpropagation — алгоритм эффективного вычисления производных функции потерь по всем параметрам сети.

MIT описывает его через три идеи:

1. вычисление представляется как **computational graph**;
2. у каждого шага есть локальная производная;
3. chain rule переносит производную от выхода к входу в reverse mode.

Backpropagation не является отдельным способом обновлять веса. Он только вычисляет градиенты. После него нужен optimizer или ручное правило обновления.

---

## 2. Computational graph

Пусть

$$
z=f(u),
\qquad
u=g(x).
$$

Тогда

$$
z=f(g(x)).
$$

Граф:

```text
x ──g──> u ──f──> z
```

В нейросети узлами графа являются:

- матричные умножения;
- сложения bias;
- ReLU;
- Dropout;
- softmax;
- cross-entropy;
- L2-регуляризация.

Forward вычисляет значения слева направо. Backward вычисляет производные справа налево.

---

## 3. Chain rule для скаляров

Если

$$
z=f(u),
\qquad
u=g(x),
$$

то

$$
\boxed{
\frac{dz}{dx}
=
\frac{dz}{du}
\frac{du}{dx}
}
$$

Пример:

$$
u=x^2,
\qquad
z=\sin u.
$$

Тогда

$$
\frac{dz}{dx}
=
\cos(u)\cdot2x
=
2x\cos(x^2).
$$

Backprop просто применяет этот принцип много раз к большому графу.

---

## 4. Локальный и входящий градиенты

Для узла

$$
y=f(x)
$$

из правой части графа приходит upstream gradient:

$$
\frac{\partial L}{\partial y}.
$$

Узел знает свою локальную производную:

$$
\frac{\partial y}{\partial x}.
$$

Он вычисляет downstream gradient:

$$
\boxed{
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
}
$$

Удобная формулировка:

```text
downstream gradient = upstream gradient × local gradient
```

---

## 5. Reverse mode

Loss — один скаляр, а параметров очень много. Поэтому выгодно начать с

$$
\frac{\partial L}{\partial L}=1
$$

и двигаться назад, вычисляя градиенты всех предков.

Это **reverse-mode differentiation**.

Forward mode был бы удобнее, если входной параметр один, а выходов много. Для нейросетей ситуация обратная: параметров много, loss один, поэтому reverse mode намного эффективнее.

---

## 6. Почему backprop быстрее прямого повторения chain rule

Если отдельно выписывать

$$
\frac{\partial L}{\partial\theta_1},
\quad
\frac{\partial L}{\partial\theta_2},
\quad\ldots,
$$

в формулах много общих множителей. Backprop вычисляет общие промежуточные производные один раз и переиспользует их.

Например, величина

$$
\frac{\partial L}{\partial H_1}
$$

нужна сразу для всех параметров первого слоя. Нет смысла повторно вычислять её для каждого элемента $W_1$.

MIT связывает этот reuse с идеей dynamic programming.

---

## 7. Правило для сложения

Если

$$
y=x_1+x_2,
$$

то

$$
\frac{\partial y}{\partial x_1}=1,
\qquad
\frac{\partial y}{\partial x_2}=1.
$$

Поэтому входящий градиент копируется в обе ветви:

$$
dx_1=dy,
\qquad
dx_2=dy.
$$

Если переменная используется в нескольких ветвях графа, её градиенты **суммируются**.

---

## 8. Правило для умножения скаляров

Если

$$
y=x_1x_2,
$$

то

$$
\frac{\partial y}{\partial x_1}=x_2,
\qquad
\frac{\partial y}{\partial x_2}=x_1.
$$

Следовательно:

$$
dx_1=dy\,x_2,
$$

$$
dx_2=dy\,x_1.
$$

В backward нужны значения $x_1,x_2$ из forward, поэтому они должны быть сохранены.

---

## 9. Правило для матричного слоя

Пусть

$$
Z=XW+b.
$$

Из следующего слоя приходит

$$
dZ=\frac{\partial L}{\partial Z}.
$$

Тогда:

$$
\boxed{dX=dZW^T}
$$

$$
\boxed{dW=X^TdZ}
$$

$$
\boxed{db=\sum_{i=1}^{N}dZ_i}
$$

### Проверка форм

Если

$$
X:N\times D,
\quad
W:D\times H,
\quad
dZ:N\times H,
$$

то

$$
dX:(N\times H)(H\times D)=N\times D,
$$

$$
dW:(D\times N)(N\times H)=D\times H.
$$

Форма градиента всегда совпадает с формой соответствующей переменной.

---

## 10. Почему не строят полный Jacobian

Для векторной функции

$$
\mathbf y=f(\mathbf x)
$$

Jacobian содержит все частные производные:

$$
J_{ij}=\frac{\partial y_i}{\partial x_j}.
$$

Но полный Jacobian может быть огромным. В backprop обычно нужен не сам Jacobian, а произведение:

$$
\frac{\partial L}{\partial\mathbf y}J.
$$

Это vector-Jacobian product. Его можно вычислить напрямую матричными формулами, не создавая гигантскую промежуточную матрицу.

---

## 11. Backward через ReLU

Forward:

$$
H=\max(0,Z).
$$

Локальная производная:

$$
\frac{\partial H}{\partial Z}=\mathbb1(Z>0).
$$

Backward:

$$
\boxed{
dZ=dH\odot\mathbb1(Z>0)}.
$$

Интерпретация: ReLU либо пропускает upstream gradient без изменения, либо блокирует его.

---

## 12. Backward через softmax + cross-entropy

Вместо отдельного построения Jacobian softmax используют объединённый результат:

$$
\boxed{
dS=\frac{P-Y}{N}}.
$$

Это пример того, почему полезно аналитически объединять соседние операции графа: итоговая производная намного проще промежуточных выражений.

---

## 13. Двухслойная сеть как graph

Forward:

```text
X
│
├─ Linear(W1,b1) → Z1
│
├─ ReLU → H1
│
├─ Linear(W2,b2) → S
│
├─ Softmax → P
│
└─ CrossEntropy(y) → L
```

Backward:

```text
1 = dL/dL
      ↓
dS = (P-Y)/N
      ↓
dW2, db2, dH1
      ↓
dZ1 = dH1 * I(Z1>0)
      ↓
dW1, db1, dX
```

---

## 14. Полный вывод для двухслойной сети

### Forward

$$
Z_1=XW_1+b_1,
$$

$$
H_1=\operatorname{ReLU}(Z_1),
$$

$$
S=H_1W_2+b_2,
$$

$$
P=\operatorname{softmax}(S).
$$

### Backward

$$
dS=\frac{P-Y}{N},
$$

$$
dW_2=H_1^TdS,
$$

$$
db_2=\sum_i dS_i,
$$

$$
dH_1=dSW_2^T,
$$

$$
dZ_1=dH_1\odot\mathbb1(Z_1>0),
$$

$$
dW_1=X^TdZ_1,
$$

$$
db_1=\sum_i dZ_{1,i}.
$$

При L2:

$$
dW_l\leftarrow dW_l+\lambda W_l.
$$

---

## 15. Backprop для трёхслойной сети

Forward:

$$
Z_1=XW_1+b_1,
\quad H_1=\operatorname{ReLU}(Z_1),
$$

$$
Z_2=H_1W_2+b_2,
\quad H_2=\operatorname{ReLU}(Z_2),
$$

$$
S=H_2W_3+b_3.
$$

Backward идёт строго в обратном порядке:

```text
S → W3,b3,H2
H2 → ReLU2 → Z2
Z2 → W2,b2,H1
H1 → ReLU1 → Z1
Z1 → W1,b1,X
```

Если поменять порядок или использовать уже обновлённые веса, градиенты перестанут соответствовать одному forward.

---

## 16. Cache как часть ручного слоя

Хороший интерфейс слоя:

```python
def linear_forward(x, W, b):
    out = x @ W + b
    cache = (x, W)
    return out, cache


def linear_backward(dout, cache):
    x, W = cache
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db
```

Для ReLU:

```python
def relu_forward(z):
    return np.maximum(0, z), z


def relu_backward(dh, z):
    return dh * (z > 0)
```

В лабораторной нельзя использовать готовые layers, но можно самостоятельно организовать код такими функциями или классами.

---

## 17. Почему нельзя обновлять веса во время backward

Допустим, сначала вычислили $dW_3$ и сразу изменили $W_3$. Тогда при расчёте

$$
dH_2=dSW_3^T
$$

будут использованы уже новые веса, хотя forward выполнялся со старыми.

Правильно:

1. выполнить forward;
2. вычислить loss;
3. выполнить полный backward;
4. сохранить все gradients;
5. затем одновременно обновить параметры.

---

## 18. Gradient accumulation в ветвящемся графе

Если переменная $x$ влияет на loss через две ветви:

$$
L=L_1(x)+L_2(x),
$$

то

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L_1}{\partial x}
+
\frac{\partial L_2}{\partial x}.
$$

Это важно для:

- residual connections;
- shared parameters;
- нескольких loss-компонент;
- L2-регуляризации.

Градиент data loss и градиент regularization складываются по тому же правилу.

---

## 19. Backprop и автоматическое дифференцирование

Современные библиотеки автоматически строят вычислительный граф и применяют reverse mode. Но лабораторная запрещает autograd, поэтому нужно вручную реализовать ту же логику:

- сохранить промежуточные значения;
- знать локальные производные;
- передавать upstream gradient;
- суммировать вклад ветвей;
- проверять формы.

То есть задача лабораторной — не придумать другой алгоритм, а самостоятельно воспроизвести основы reverse-mode differentiation.

---

## 20. Вопросы для самопроверки

1. Чем backpropagation отличается от gradient descent?
2. Что такое upstream gradient?
3. Почему переменные forward надо сохранять?
4. Почему backprop идёт от loss к входу?
5. Почему reverse mode выгоден для scalar loss и большого числа параметров?
6. Откуда берутся формулы $dX=dZW^T$ и $dW=X^TdZ$?
7. Почему градиенты ветвей суммируются?
8. Почему не нужно строить полный Jacobian?
9. Почему параметры обновляют только после полного backward?
10. Как проверить формы всех градиентов?

## Краткий устный ответ

> Backpropagation — это reverse-mode применение chain rule к вычислительному графу. Каждый узел получает upstream gradient, умножает его на локальную производную и передаёт дальше. Общие промежуточные производные вычисляются один раз и переиспользуются, поэтому backprop эффективнее отдельного вывода каждой частной производной. Он вычисляет градиенты, а gradient descent уже использует их для обновления весов.
